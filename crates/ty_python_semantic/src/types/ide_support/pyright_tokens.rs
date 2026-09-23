//! Semantic facts in the shape used by pyright's semantic token walker.
//!
//! basedpyright classifies every name in a file from two inputs: the declarations of the symbol
//! that the name refers to, and the type that the name evaluates to. Both inputs are modelled on
//! pyright's type system: declarations have a small set of kinds (variable, parameter, class, ...),
//! and types fall into categories (class, function, module, ...) with flags such as "instantiable"
//! or "special form". This module computes the equivalent facts from ty's semantic model, so that
//! the semantic token provider in `ty_ide` can reproduce basedpyright's token types and modifiers.

use std::cell::OnceCell;

use ruff_db::parsed::{ParsedModuleRef, parsed_module};
use ruff_python_ast as ast;
use ruff_python_ast::helpers::ReturnStatementVisitor;
use ruff_python_ast::name::Name;
use ruff_python_ast::visitor::Visitor;
use ruff_text_size::{Ranged, TextRange, TextSize};
use ty_module_resolver::file_to_module;
use ty_python_core::definition::{Definition, DefinitionKind, ParameterDefinitionNodeKind};
use ty_python_core::place::{PlaceExprRef, ScopedPlaceId};
use ty_python_core::scope::{NodeWithScopeKind, NodeWithScopeRef, ScopeId, ScopeKind};
use ty_python_core::{ProgramFile, attribute_scopes, place_table, semantic_index, use_def_map};

use crate::types::class::{ClassLiteral, KnownClass, StaticClassLiteral};
use crate::types::definition_resolution;
use crate::types::enums::is_enum_class_by_inheritance;
use crate::types::function::FunctionType;
use crate::types::ide_support::{
    CallSignatureDetails, definitions_for_keyword_argument, resolved_call_signature,
};
use crate::types::infer::{
    infer_complete_scope_types, infer_definition_types, original_class_type,
};
use crate::types::signatures::CallableSignature;
use crate::types::special_form::SpecialFormType;
use crate::types::subclass_of::SubclassOfInner;
use crate::types::{
    BoundTypeVarInstance, ClassBase, DynamicType, IntersectionType, KnownInstanceType, KnownUnion,
    MemberLookupPolicy, Type, TypeQualifiers, TypeVarKind, UnionType, binding_type,
    inferred_declaration,
};
use crate::{Db, FxIndexSet, HasType, ResolvedDefinition, SemanticModel};

/// The kind of a declaration, following pyright's `DeclarationType`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PyrightDeclarationKind {
    Variable,
    Parameter,
    TypeParameter,
    TypeAlias,
    Function,
    Class,
    /// A special form declared in `typing.pyi` or `typing_extensions.pyi`, such as `Protocol`.
    SpecialBuiltInClass,
    /// An import that could not be resolved to a declaration, or that resolved to a module.
    Alias,
}

/// A declaration of the symbol that a name refers to, with the facts pyright's walker reads.
#[derive(Debug, Clone, Copy)]
#[expect(
    clippy::struct_excessive_bools,
    reason = "the fields mirror the facts that pyright's walker reads from a declaration"
)]
pub struct PyrightDeclaration<'db> {
    pub kind: PyrightDeclarationKind,
    pub definition: Option<Definition<'db>>,
    /// The declaration lives in `builtins.pyi` (or a project `__builtins__.pyi`).
    pub in_builtins_module: bool,
    /// The declaration lives in a project `__builtins__.pyi`.
    pub in_project_builtins_module: bool,
    /// A variable declaration whose name follows the constant naming convention, or that is
    /// annotated with `Final`.
    pub is_constant_or_final: bool,
    /// The declaration is directly inside a class body. Lambdas and comprehensions don't count as
    /// a boundary, but functions do.
    pub in_class_body: bool,
    /// Like `in_class_body`, and the class is an enum class.
    pub in_enum_class_body: bool,
    /// Like `in_class_body`, and the class (including its bases) has a member with the
    /// declaration's name.
    pub enclosing_class_has_member: bool,
    /// The declaration's declared type is a property without a setter.
    pub is_property_without_setter: bool,
    /// Pyright's `hasTypeForDeclaration` (see [`pyright_has_declared_type`]).
    pub has_declared_type: bool,
}

impl PyrightDeclaration<'_> {
    pub fn is_variable(&self) -> bool {
        self.kind == PyrightDeclarationKind::Variable
    }

    /// The declaration of a `__slots__` entry.
    pub fn is_slot(&self) -> bool {
        self.is_variable() && self.definition.is_none() && self.in_class_body
    }

    /// A variable declaration that pyright's binder creates without a ty definition: a
    /// `__slots__` entry (in the class body), a field of a functional named tuple, or a class's
    /// implicit `__doc__` and `__module__` (whose node is the class itself, so they don't count as
    /// declared in the class body).
    fn synthesized_variable(in_class_body: bool, is_constant: bool) -> Self {
        Self {
            kind: PyrightDeclarationKind::Variable,
            definition: None,
            in_builtins_module: false,
            in_project_builtins_module: false,
            is_constant_or_final: is_constant,
            in_class_body,
            in_enum_class_body: false,
            enclosing_class_has_member: in_class_body,
            is_property_without_setter: false,
            // The implicit class attributes and the fields of named tuples are typed, while
            // `__slots__` entries aren't.
            has_declared_type: !in_class_body,
        }
    }

    /// An import that doesn't resolve to a definition, such as a module.
    fn alias(in_builtins_module: bool, in_project_builtins_module: bool) -> Self {
        Self {
            kind: PyrightDeclarationKind::Alias,
            definition: None,
            in_builtins_module,
            in_project_builtins_module,
            is_constant_or_final: false,
            in_class_body: false,
            in_enum_class_body: false,
            enclosing_class_has_member: false,
            is_property_without_setter: false,
            has_declared_type: false,
        }
    }
}

/// Converts resolved definitions into pyright-style declarations, ordered the way pyright's binder
/// creates them.
///
/// Pyright binds a scope's own statements in source order and defers function bodies until the
/// enclosing scope is done. For example, a class attribute declared in the class body comes
/// before one assigned through `self.x = ...` in a method, even if the method is defined first.
pub fn pyright_declarations<'db>(
    model: &SemanticModel<'db>,
    definitions: &[ResolvedDefinition<'db>],
) -> Vec<PyrightDeclaration<'db>> {
    let db = model.db();
    let mut declarations: Vec<PyrightDeclaration<'db>> = definitions
        .iter()
        .map(|resolved| match resolved {
            ResolvedDefinition::Definition(definition) => pyright_declaration(model, *definition),
            ResolvedDefinition::Module(file) => {
                let (in_builtins_module, in_project_builtins_module) =
                    builtins_module_flags(db, *file);
                PyrightDeclaration::alias(in_builtins_module, in_project_builtins_module)
            }
            ResolvedDefinition::FileWithRange(_) => PyrightDeclaration::alias(false, false),
        })
        .collect();

    // Only reorder the declarations of a single file.
    let mut definitions = declarations
        .iter()
        .filter_map(|declaration| declaration.definition);
    let Some(first) = definitions.next() else {
        return declarations;
    };
    if !definitions.all(|definition| definition.file(db) == first.file(db)) {
        return declarations;
    }
    let parsed = parsed_module(db, first.python_file(db)).load(db);
    // Declarations without a definition keep their place after the declaration before them.
    let mut previous = DeclarationOrder::default();
    let mut orders = Vec::with_capacity(declarations.len());
    for declaration in &declarations {
        if let Some(definition) = declaration.definition {
            previous = DeclarationOrder {
                deferred: is_in_deferred_scope(db, definition),
                offset: definition.focus_range(db, &parsed).start(),
            };
        }
        orders.push(previous);
    }
    let mut ordered: Vec<_> = orders.into_iter().zip(declarations.drain(..)).collect();
    ordered.sort_by_key(|(order, _)| *order);
    ordered
        .into_iter()
        .map(|(_, declaration)| declaration)
        .collect()
}

/// The position of a declaration in pyright's binding order.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, PartialOrd, Ord)]
struct DeclarationOrder {
    /// The declaration is in a function body that pyright binds after the enclosing scope.
    deferred: bool,
    offset: TextSize,
}

/// Returns the pyright-style declaration for a single definition.
pub fn pyright_declaration<'db>(
    model: &SemanticModel<'db>,
    definition: Definition<'db>,
) -> PyrightDeclaration<'db> {
    let db = model.db();
    let file = definition.program_file(db);
    let (in_builtins_module, in_project_builtins_module) = builtins_module_flags(db, file);
    let kind = declaration_kind(db, definition);
    let name = definition_name(db, definition);

    let enclosing_class = enclosing_class(db, definition);
    let in_enum_class_body = enclosing_class.is_some_and(|class| is_enum_class(db, model, class));
    // Pyright looks the name up in the class's symbol table, which holds every name bound
    // directly in the class body, even ones ty doesn't treat as members (`_: KW_ONLY`).
    let enclosing_class_has_member = enclosing_class.is_some_and(|class| {
        definition.scope(db).scope(db).kind() == ScopeKind::Class
            || name
                .as_deref()
                .is_some_and(|name| class_has_member(db, model, class, name))
    });

    let is_constant_or_final = kind == PyrightDeclarationKind::Variable
        && (name.as_deref().is_some_and(is_constant_name) || is_final(db, definition));

    let is_property_without_setter = matches!(definition.kind(db), DefinitionKind::Function(_))
        && pyright_definition_type(db, definition)
            .and_then(Type::as_property_instance)
            .is_some_and(|property| property.setter(db).is_none());

    PyrightDeclaration {
        kind,
        definition: Some(definition),
        in_builtins_module,
        in_project_builtins_module,
        is_constant_or_final,
        in_class_body: enclosing_class.is_some(),
        in_enum_class_body,
        enclosing_class_has_member,
        is_property_without_setter,
        has_declared_type: pyright_has_declared_type(db, definition),
    }
}

/// Pyright's `isConstantName`: only ASCII uppercase letters, digits and underscores, and not only
/// underscores. Single-letter names such as `T` count.
fn is_constant_name(name: &str) -> bool {
    !name.is_empty()
        && name
            .bytes()
            .all(|byte| byte.is_ascii_uppercase() || byte.is_ascii_digit() || byte == b'_')
        && !name.bytes().all(|byte| byte == b'_')
}

fn builtins_module_flags(db: &dyn Db, file: ProgramFile<'_>) -> (bool, bool) {
    let Some(module) = file_to_module(db, file.resolver_file(db)) else {
        return (false, false);
    };
    let name = module.name(db).as_str();
    let project_builtins = name.rsplit('.').next() == Some("__builtins__");
    (name == "builtins" || project_builtins, project_builtins)
}

/// Names that pyright's binder declares as special built-in classes in the typing stubs.
const SPECIAL_BUILTIN_CLASSES: &[&str] = &[
    "Tuple",
    "Generic",
    "Protocol",
    "Callable",
    "Type",
    "ClassVar",
    "Final",
    "Literal",
    "TypedDict",
    "Union",
    "Optional",
    "Annotated",
    "TypeAlias",
    "Concatenate",
    "TypeGuard",
    "Unpack",
    "Self",
    "NoReturn",
    "Never",
    "LiteralString",
    "OrderedDict",
    "TypeIs",
];

fn declaration_kind<'db>(db: &'db dyn Db, definition: Definition<'db>) -> PyrightDeclarationKind {
    match definition.kind(db) {
        DefinitionKind::Class(_) => PyrightDeclarationKind::Class,
        DefinitionKind::Function(_) => PyrightDeclarationKind::Function,
        DefinitionKind::TypeAlias(_) => PyrightDeclarationKind::TypeAlias,
        DefinitionKind::TypeVar(_)
        | DefinitionKind::ParamSpec(_)
        | DefinitionKind::TypeVarTuple(_) => PyrightDeclarationKind::TypeParameter,
        DefinitionKind::Parameter(_) | DefinitionKind::LambdaParameter(_) => {
            PyrightDeclarationKind::Parameter
        }
        DefinitionKind::Import(_)
        | DefinitionKind::ImportFrom(_)
        | DefinitionKind::ImportFromSubmodule(_)
        | DefinitionKind::StarImport(_) => PyrightDeclarationKind::Alias,
        DefinitionKind::AnnotatedAssignment(_) if is_special_builtin_class(db, definition) => {
            PyrightDeclarationKind::SpecialBuiltInClass
        }
        DefinitionKind::NamedExpression(_)
        | DefinitionKind::Assignment(_)
        | DefinitionKind::AnnotatedAssignment(_)
        | DefinitionKind::AugmentedAssignment(_)
        | DefinitionKind::DictKeyAssignment(_)
        | DefinitionKind::For(_)
        | DefinitionKind::Comprehension(_)
        | DefinitionKind::WithItem(_)
        | DefinitionKind::MatchPattern(_)
        | DefinitionKind::ExceptHandler(_)
        | DefinitionKind::LoopHeader(_)
        | DefinitionKind::NestedBindings(_) => PyrightDeclarationKind::Variable,
    }
}

fn is_special_builtin_class<'db>(db: &'db dyn Db, definition: Definition<'db>) -> bool {
    let file = definition.program_file(db);
    if !file.file(db).is_stub(db) {
        return false;
    }
    let Some(module) = file_to_module(db, file.resolver_file(db)) else {
        return false;
    };
    matches!(module.name(db).as_str(), "typing" | "typing_extensions")
        && definition_name(db, definition)
            .is_some_and(|name| SPECIAL_BUILTIN_CLASSES.contains(&name.as_str()))
}

/// The name that a definition binds: the symbol name, or the attribute name for `self.x = ...`.
fn definition_name<'db>(db: &'db dyn Db, definition: Definition<'db>) -> Option<String> {
    let table = place_table(db, definition.scope(db));
    match table.place(definition.place(db)) {
        PlaceExprRef::Symbol(symbol) => Some(symbol.name().to_string()),
        PlaceExprRef::Member(member) => member.as_instance_attribute().map(str::to_string),
    }
}

fn is_final<'db>(db: &'db dyn Db, definition: Definition<'db>) -> bool {
    matches!(definition.kind(db), DefinitionKind::AnnotatedAssignment(_))
        && inferred_declaration(db, definition)
            .declared()
            .is_some_and(|declared| declared.qualifiers().contains(TypeQualifiers::FINAL))
}

/// Whether pyright's binder defers the definition: it sits in a function or lambda body while
/// binding an attribute of the instance or class (`self.x = ...`).
fn is_in_deferred_scope<'db>(db: &'db dyn Db, definition: Definition<'db>) -> bool {
    matches!(definition.place(db), ScopedPlaceId::Member(_))
        && matches!(
            definition.scope(db).scope(db).kind(),
            ScopeKind::Function | ScopeKind::Lambda
        )
}

/// Pyright's `getEnclosingClass(declarationNode, stopAtFunction=true)`: the class whose body
/// contains the definition, looking through lambdas and comprehensions but not functions.
fn enclosing_class<'db>(db: &'db dyn Db, definition: Definition<'db>) -> Option<ClassLiteral<'db>> {
    let file = definition.program_file(db);
    let index = semantic_index(db, file);
    // Parameters are bound in the function's own scope, but pyright looks at the parameter's
    // parent, which is the function node itself.
    if matches!(definition.kind(db), DefinitionKind::Parameter(_)) {
        return None;
    }
    for (_, scope) in index.ancestor_scopes(definition.file_scope(db)) {
        match scope.kind() {
            ScopeKind::Class => {
                let NodeWithScopeKind::Class(class) = scope.node() else {
                    return None;
                };
                let parsed = parsed_module(db, file.python_file(db)).load(db);
                let class_definition = index.try_definition(class.node(&parsed))?;
                return original_class_type(db, class_definition);
            }
            ScopeKind::Lambda | ScopeKind::Comprehension | ScopeKind::TypeParams => {}
            ScopeKind::Function | ScopeKind::Module | ScopeKind::TypeAlias => return None,
        }
    }
    None
}

fn is_enum_class<'db>(
    db: &'db dyn Db,
    model: &SemanticModel<'db>,
    class: ClassLiteral<'db>,
) -> bool {
    match class {
        ClassLiteral::Static(class) => {
            is_enum_class_by_inheritance(db, &model.program_environment(), class)
        }
        ClassLiteral::DynamicEnum(_) => true,
        ClassLiteral::Dynamic(_)
        | ClassLiteral::DynamicNamedTuple(_)
        | ClassLiteral::DynamicTypedDict(_) => false,
    }
}

fn class_has_member<'db>(
    db: &'db dyn Db,
    model: &SemanticModel<'db>,
    class: ClassLiteral<'db>,
    name: &str,
) -> bool {
    let env = model.program_environment();
    !class
        .class_member(db, &env, name, MemberLookupPolicy::default())
        .place
        .is_undefined()
        || !class
            .instance_member(db, &env, None, name)
            .place
            .is_undefined()
}

/// Where a name appears, which changes the type pyright evaluates it to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PyrightTypeContext {
    /// A name that is read in a value expression.
    Value,
    /// A name that is bound by an assignment or similar statement.
    StoreTarget,
    /// A name inside an annotation or another type expression. ty records the type of the
    /// annotated value there (for example, an instance of `int` for `int`), while pyright records
    /// the class itself.
    TypeExpression,
    /// A name imported by `from module import name`.
    ImportedName,
    /// A value expression whose special forms pyright doesn't convert to their runtime objects:
    /// the value of an implicit type alias (`JSON = Any`) and the type arguments of a generic
    /// class subscripted in a value expression (`list[Any]`). Unlike in a type expression, names
    /// keep the type of their value.
    TypeArgument,
}

/// Pyright's type categories.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PyrightTypeCategory {
    Unknown,
    Any,
    Never,
    Function,
    Overloaded,
    Class,
    Module,
    TypeVar,
    Union,
}

/// A function type's facts that pyright's walker reads.
#[derive(Debug, Clone, Copy, Default)]
#[expect(
    clippy::struct_excessive_bools,
    reason = "the fields mirror the facts that pyright's walker reads from a function type"
)]
pub struct PyrightFunction<'db> {
    /// The function is defined in the `builtins` module (including methods of builtin classes).
    pub in_builtins_module: bool,
    pub is_static_method: bool,
    /// The function is bound to an instance or class (pyright's `isMethodType`).
    pub is_method_type: bool,
    /// The function is defined in a class body.
    pub has_method_class: bool,
    /// The function's own definition, if it has one.
    pub declaration: Option<Definition<'db>>,
}

/// A type in pyright's terms.
#[derive(Debug, Clone, Copy)]
#[expect(
    clippy::struct_excessive_bools,
    reason = "the fields mirror the facts that pyright's walker reads from a type"
)]
pub struct PyrightType<'db> {
    pub category: PyrightTypeCategory,
    pub is_instantiable: bool,
    pub is_instance: bool,
    /// For `Any`: the type stands for the special form `Any` itself rather than a value of type
    /// `Any`.
    pub is_special_form: bool,
    /// Pyright's `isAnyOrUnknown`, which also holds for unions of only `Any` and unknown types.
    pub is_any_or_unknown: bool,
    /// For classes: the class (or the class of the instance) is an enum class.
    pub is_enum_class: bool,
    /// Every member of the type is a function or overloaded function.
    pub all_functions: bool,
    /// Every member of the type is a function bound to an object.
    pub all_method_types: bool,
    /// For type variables: the type variable is pyright's synthesized `Self`.
    pub is_synthesized_type_var: bool,
    /// For instantiable type variables: the bound is a class.
    pub type_var_bound_is_class: bool,
    /// For unions: some member is a class object.
    pub union_contains_class_object: bool,
    pub function: Option<PyrightFunction<'db>>,
}

impl<'db> PyrightType<'db> {
    fn new(category: PyrightTypeCategory, is_instantiable: bool, is_instance: bool) -> Self {
        Self {
            category,
            is_instantiable,
            is_instance,
            is_special_form: false,
            is_any_or_unknown: matches!(
                category,
                PyrightTypeCategory::Any | PyrightTypeCategory::Unknown
            ),
            is_enum_class: false,
            all_functions: matches!(
                category,
                PyrightTypeCategory::Function | PyrightTypeCategory::Overloaded
            ),
            all_method_types: false,
            is_synthesized_type_var: false,
            type_var_bound_is_class: false,
            union_contains_class_object: false,
            function: None,
        }
    }

    /// An instance of one of the type variables that pyright synthesizes for the parameters of
    /// pseudo-generic classes (see [`pyright_is_pseudo_generic_attribute`]).
    pub fn synthesized_type_var_instance() -> Self {
        Self {
            is_synthesized_type_var: true,
            ..Self::new(PyrightTypeCategory::TypeVar, false, true)
        }
    }

    fn unknown() -> Self {
        Self::new(PyrightTypeCategory::Unknown, true, true)
    }

    fn class(is_instantiable: bool, is_enum_class: bool) -> Self {
        Self {
            is_enum_class,
            ..Self::new(
                PyrightTypeCategory::Class,
                is_instantiable,
                !is_instantiable,
            )
        }
    }

    fn function(function: PyrightFunction<'db>, overloaded: bool, is_instantiable: bool) -> Self {
        let category = if overloaded {
            PyrightTypeCategory::Overloaded
        } else {
            PyrightTypeCategory::Function
        };
        Self {
            all_method_types: function.is_method_type,
            function: Some(function),
            ..Self::new(category, is_instantiable, !is_instantiable)
        }
    }

    pub fn is_class(&self) -> bool {
        self.category == PyrightTypeCategory::Class
    }

    pub fn is_instantiable_class(&self) -> bool {
        self.is_class() && self.is_instantiable
    }
}

/// Classifies a type the way pyright's type system would describe it.
pub fn pyright_type<'db>(
    model: &SemanticModel<'db>,
    ty: Type<'db>,
    context: PyrightTypeContext,
) -> PyrightType<'db> {
    pyright_type_impl(model, ty, context, 0)
}

/// How many recursive type aliases [`pyright_type`] unfolds before it gives up. The body of a
/// recursive alias can refer to the alias again (`X = int | list[X]`).
const MAX_RECURSIVE_UNFOLDS: u8 = 2;

fn pyright_type_impl<'db>(
    model: &SemanticModel<'db>,
    ty: Type<'db>,
    context: PyrightTypeContext,
    unfolds: u8,
) -> PyrightType<'db> {
    let db = model.db();
    let type_expression = context == PyrightTypeContext::TypeExpression;
    let pyright_type = |model: &SemanticModel<'db>, ty: Type<'db>, context: PyrightTypeContext| {
        pyright_type_impl(model, ty, context, unfolds)
    };

    match ty {
        // Pyright expands a recursive type alias (`MarkerList = List[Union["MarkerList", ...]]`)
        // to its value.
        Type::Recursive(recursive) => {
            match recursive
                .unfold(db, &model.program_environment())
                .into_unfolded()
            {
                Some(unfolded)
                    if unfolds < MAX_RECURSIVE_UNFOLDS
                        && !matches!(unfolded, Type::Recursive(_)) =>
                {
                    pyright_type_impl(model, unfolded, context, unfolds + 1)
                }
                _ => PyrightType::unknown(),
            }
        }
        // Pyright narrows an unknown value to the narrowing type itself (`isinstance(x, F)` gives
        // `F`), and narrowing `None | Unknown` with `is not None` leaves `Unknown`. ty keeps the
        // unknown part as an element of an intersection (`Unknown & F`, `Unknown & ~None`).
        Type::Intersection(intersection)
            if let Some(narrowed) = pyright_intersection_without_dynamic(db, intersection) =>
        {
            pyright_type(model, narrowed, context)
        }
        // A value of type `Any`; the special form `Any` itself is `Type::SpecialForm`.
        Type::Dynamic(DynamicType::Any) => PyrightType::new(PyrightTypeCategory::Any, true, true),
        Type::Dynamic(_) | Type::Divergent(_) | Type::RecursiveVar(_) => PyrightType::unknown(),
        Type::Never => PyrightType::new(PyrightTypeCategory::Never, true, true),
        Type::FunctionLiteral(function) => {
            let (overloads, _) = function.overloads_and_implementation(db);
            PyrightType::function(
                function_facts(db, function, false),
                !overloads.is_empty(),
                false,
            )
        }
        Type::BoundMethod(method) => match method.function(db) {
            Some(function) => {
                let (overloads, _) = function.overloads_and_implementation(db);
                let mut facts = function_facts(db, function, function.name(db) != "__new__");
                // Pyright synthesizes the accessor methods of a property object (`x.setter`),
                // so they don't come from the `builtins` module.
                if matches!(method.self_instance(db), Type::PropertyInstance(_)) {
                    facts.in_builtins_module = false;
                }
                PyrightType::function(facts, !overloads.is_empty(), false)
            }
            None => PyrightType::function(
                PyrightFunction {
                    is_method_type: true,
                    ..PyrightFunction::default()
                },
                false,
                false,
            ),
        },
        Type::KnownBoundMethod(_) | Type::WrapperDescriptor(_) => PyrightType::function(
            PyrightFunction {
                is_method_type: true,
                ..PyrightFunction::default()
            },
            false,
            false,
        ),
        Type::DataclassDecorator(_) | Type::DataclassTransformer(_) => {
            PyrightType::function(PyrightFunction::default(), false, false)
        }
        Type::Callable(callable) => {
            // ty widens functions stored in undeclared class attributes (`d = print`,
            // `concat = "".join`) to callables, which keep the function's definition. Pyright keeps
            // the function type, whose module decides `defaultLibrary` and `builtin`.
            let in_builtins_module = callable
                .signatures(db)
                .overloads
                .first()
                .and_then(|signature| signature.definition)
                .is_some_and(|definition| builtins_module_flags(db, definition.program_file(db)).0);
            PyrightType::function(
                PyrightFunction {
                    in_builtins_module,
                    ..PyrightFunction::default()
                },
                false,
                type_expression,
            )
        }
        Type::ModuleLiteral(_) => PyrightType::new(PyrightTypeCategory::Module, true, false),
        Type::ClassLiteral(class) => PyrightType::class(true, is_enum_class(db, model, class)),
        Type::GenericAlias(alias) => PyrightType::class(
            true,
            is_enum_class(db, model, ClassLiteral::Static(alias.origin(db))),
        ),
        Type::SubclassOf(subclass_of) => match subclass_of.subclass_of() {
            SubclassOfInner::Class(class) => {
                PyrightType::class(true, is_enum_class(db, model, class.class_literal(db)))
            }
            // Pyright keeps `type[Any]` an instance of `type` rather than a class object.
            SubclassOfInner::Dynamic(_) => PyrightType::class(false, false),
            SubclassOfInner::Protocol(_) => PyrightType::class(true, false),
            SubclassOfInner::TypeVar(type_var) => {
                let typevar = type_var.typevar(db);
                PyrightType {
                    is_synthesized_type_var: matches!(typevar.kind(db), TypeVarKind::TypingSelf),
                    type_var_bound_is_class: true,
                    ..PyrightType::new(PyrightTypeCategory::TypeVar, true, false)
                }
            }
        },
        Type::NominalInstance(_)
        | Type::ProtocolInstance(_)
        | Type::TypedDict(_)
        | Type::NewTypeInstance(_)
        | Type::LiteralValue(_)
        | Type::PropertyInstance(_)
        | Type::SlotDescriptor(_)
        | Type::EnumComplement(_)
        | Type::BoundSuper(_)
        | Type::TypeIs(_)
        | Type::TypeGuard(_)
        | Type::TypeForm(_)
        | Type::AlwaysTruthy
        | Type::AlwaysFalsy
        | Type::Intersection(_) => {
            let is_enum_class =
                instance_class(db, model, ty).is_some_and(|class| is_enum_class(db, model, class));
            PyrightType::class(type_expression, is_enum_class)
        }
        Type::SpecialForm(special_form) => match (special_form, context) {
            // In other value expressions pyright converts special forms to their runtime objects,
            // which are classes.
            (
                SpecialFormType::Any,
                PyrightTypeContext::TypeExpression
                | PyrightTypeContext::ImportedName
                | PyrightTypeContext::TypeArgument,
            ) => PyrightType {
                is_special_form: true,
                ..PyrightType::new(PyrightTypeCategory::Any, true, true)
            },
            (
                SpecialFormType::NoReturn | SpecialFormType::Never,
                PyrightTypeContext::TypeExpression
                | PyrightTypeContext::ImportedName
                | PyrightTypeContext::TypeArgument,
            ) => PyrightType::new(PyrightTypeCategory::Never, true, true),
            (SpecialFormType::TypingSelf, context)
                if context != PyrightTypeContext::ImportedName =>
            {
                PyrightType {
                    is_synthesized_type_var: true,
                    ..PyrightType::new(PyrightTypeCategory::TypeVar, true, false)
                }
            }
            _ => PyrightType::class(true, false),
        },
        Type::KnownInstance(known) => match known {
            KnownInstanceType::TypeVar(_) => {
                let instantiable = context != PyrightTypeContext::Value;
                PyrightType::new(PyrightTypeCategory::TypeVar, instantiable, !instantiable)
            }
            // `Literal["a"]` is the class of a single literal value for pyright, and
            // `Literal["a", "b"]` a union of them.
            KnownInstanceType::Literal(literal) if !matches!(literal.inner(db), Type::Union(_)) => {
                let is_enum_class = instance_class(db, model, literal.inner(db))
                    .is_some_and(|class| is_enum_class(db, model, class));
                PyrightType::class(true, is_enum_class)
            }
            KnownInstanceType::UnionType(_)
            | KnownInstanceType::Literal(_)
            | KnownInstanceType::LiteralStringAlias(_) => {
                PyrightType::new(PyrightTypeCategory::Union, true, false)
            }
            KnownInstanceType::Callable(_) => {
                PyrightType::function(PyrightFunction::default(), false, true)
            }
            KnownInstanceType::Annotated(_)
            | KnownInstanceType::TypeGenericAlias(_)
            | KnownInstanceType::SubscriptedGeneric(_)
            | KnownInstanceType::SubscriptedProtocol(_)
            | KnownInstanceType::NewType(_) => PyrightType::class(true, false),
            KnownInstanceType::TypeAliasType(alias) => {
                if context == PyrightTypeContext::Value {
                    PyrightType::class(false, false)
                } else {
                    pyright_type(
                        model,
                        alias.value_type(db),
                        PyrightTypeContext::TypeExpression,
                    )
                }
            }
            _ => PyrightType::class(false, false),
        },
        Type::TypeAlias(alias) => pyright_type(model, alias.value_type(db), context),
        Type::TypeVar(bound_type_var) => {
            let typevar = bound_type_var.typevar(db);
            PyrightType {
                is_synthesized_type_var: matches!(typevar.kind(db), TypeVarKind::TypingSelf),
                ..PyrightType::new(
                    PyrightTypeCategory::TypeVar,
                    type_expression,
                    !type_expression,
                )
            }
        }
        // `float` and `complex` in annotations mean `int | float` and `int | float | complex`, but
        // pyright still sees the class named in the annotation.
        Type::Union(union)
            if matches!(
                union.known(db),
                Some(KnownUnion::Float | KnownUnion::Complex)
            ) =>
        {
            PyrightType::class(type_expression, false)
        }
        Type::Union(union) => {
            // Pyright's `combineTypes` drops members that are the same type as another, ignoring
            // the object that a method is bound to (`s.lower` for `s: Literal["a", "b"]`).
            if let Some(method) = identical_bound_methods(db, union) {
                return pyright_type(model, method, context);
            }
            let elements: Vec<PyrightType<'db>> = union
                .elements(db)
                .iter()
                .map(|element| pyright_type(model, *element, context))
                .collect();
            PyrightType {
                category: PyrightTypeCategory::Union,
                is_instantiable: elements.iter().all(|element| element.is_instantiable),
                is_instance: elements.iter().all(|element| element.is_instance),
                is_special_form: false,
                is_any_or_unknown: elements.iter().all(|element| element.is_any_or_unknown),
                is_enum_class: false,
                all_functions: elements.iter().all(|element| element.all_functions),
                all_method_types: elements.iter().all(|element| element.all_method_types),
                is_synthesized_type_var: false,
                type_var_bound_is_class: false,
                union_contains_class_object: elements
                    .iter()
                    .any(PyrightType::is_instantiable_class),
                function: None,
            }
        }
    }
}

/// The first member of a union whose members are all the same method bound to different objects.
fn identical_bound_methods<'db>(db: &'db dyn Db, union: UnionType<'db>) -> Option<Type<'db>> {
    let (first, rest) = union.elements(db).split_first()?;
    let Type::BoundMethod(first_method) = first else {
        return None;
    };
    let definition = first_method.function(db)?.definition(db);
    let signature = first_method.bound_signatures(db)?;
    let is_same = |other: &CallableSignature<'db>| {
        other.overloads.len() == signature.overloads.len()
            && other
                .overloads
                .iter()
                .zip(&signature.overloads)
                .all(|(a, b)| a.parameters() == b.parameters() && a.return_ty == b.return_ty)
    };
    rest.iter()
        .all(|element| {
            matches!(element, Type::BoundMethod(method)
                if method.function(db).map(|function| function.definition(db)) == Some(definition)
                    && method.bound_signatures(db).is_some_and(is_same))
        })
        .then_some(*first)
}

fn function_facts<'db>(
    db: &'db dyn Db,
    function: FunctionType<'db>,
    is_method_type: bool,
) -> PyrightFunction<'db> {
    let definition = function.definition(db);
    let (in_builtins_module, _) = builtins_module_flags(db, definition.program_file(db));
    let is_constructor = function.name(db) == "__new__";
    let has_method_class = definition.scope(db).scope(db).kind() == ScopeKind::Class;
    PyrightFunction {
        in_builtins_module,
        is_static_method: has_method_class && function.is_staticmethod(db) && !is_constructor,
        is_method_type,
        has_method_class,
        declaration: Some(definition),
    }
}

/// The class of an instance type, or of a class object.
fn instance_class<'db>(
    db: &'db dyn Db,
    model: &SemanticModel<'db>,
    ty: Type<'db>,
) -> Option<ClassLiteral<'db>> {
    let env = model.program_environment();
    // ty normalizes `type[object]` to an instance of `type`, so the meta type of an `object`
    // instance loses `object`.
    if let Type::NominalInstance(instance) = ty {
        let class = instance.class(db, &env);
        if class.is_object(db) {
            return Some(class.class_literal(db));
        }
    }
    let meta = match ty {
        Type::ClassLiteral(_) | Type::GenericAlias(_) | Type::SubclassOf(_) => ty,
        _ => ty.to_meta_type(db, &env),
    };
    match meta {
        Type::ClassLiteral(class) => Some(class),
        Type::GenericAlias(alias) => Some(ClassLiteral::Static(alias.origin(db))),
        Type::SubclassOf(subclass_of) => match subclass_of.subclass_of() {
            SubclassOfInner::Class(class) => Some(class.class_literal(db)),
            _ => None,
        },
        _ => None,
    }
}

/// The type pyright has for an intersection that ty uses to represent narrowing.
///
/// Pyright has no intersection types. Narrowing `Unknown` or `Any` with `isinstance`, `TypeIs` or
/// `callable()` gives the narrowed-to type itself, negative narrowing leaves a type unchanged, and
/// `hasattr()` doesn't narrow. So `Unknown & T` and `T & ~U` are `T` for pyright, and
/// `Unknown & ~U` or `Unknown & <protocol from hasattr>` are `Unknown`. Returns `None` for
/// intersections of several classes, which pyright also represents as a single (synthesized)
/// class.
fn pyright_intersection_without_dynamic<'db>(
    db: &'db dyn Db,
    intersection: IntersectionType<'db>,
) -> Option<Type<'db>> {
    let is_transparent = |element: &Type<'db>| match element {
        Type::Dynamic(_) => true,
        Type::ProtocolInstance(protocol) => protocol.class_origin(db).is_none(),
        _ => false,
    };
    let positive = intersection.positive(db);
    let mut rest = positive
        .iter()
        .copied()
        .filter(|element| !is_transparent(element));
    match (rest.next(), rest.next()) {
        (Some(only), None) => Some(only),
        (None, _) => positive
            .iter()
            .copied()
            .find(|element| matches!(element, Type::Dynamic(_))),
        (Some(_), Some(_)) => None,
    }
}

/// Whether ty narrowed `ty` with `is not None`: an intersection with a negative `None` element.
/// Pyright's narrowing of the synthesized type variables of pseudo-generic classes loses them in
/// that case (see [`pyright_is_pseudo_generic_attribute`]).
pub fn pyright_is_narrowed_not_none<'db>(db: &'db dyn Db, ty: Type<'db>) -> bool {
    matches!(ty, Type::Intersection(intersection)
        if intersection.iter_negative(db).any(|negative| negative.is_none(db)))
}

/// The type of reading `name` through `receiver`, for receivers that ty narrowed from an unknown
/// type (`Unknown & F` after `isinstance(x, F)`). ty's attribute type is then unknown too, while
/// pyright reads the attribute from `F`. Returns `None` for other receivers.
pub fn pyright_narrowed_receiver_member_type<'db>(
    model: &SemanticModel<'db>,
    receiver: Type<'db>,
    name: &str,
) -> Option<Type<'db>> {
    let db = model.db();
    let Type::Intersection(intersection) = receiver else {
        return None;
    };
    let narrowed = pyright_intersection_without_dynamic(db, intersection)?;
    if matches!(narrowed, Type::Dynamic(_)) {
        return None;
    }
    narrowed
        .member(db, &model.program_environment(), name)
        .ignore_possibly_undefined()
}

/// The type of reading `name` through `receiver`. Pyright gives an assignment target the
/// member's declared type when the assigned value is `Any` or unknown.
pub fn pyright_member_type<'db>(
    model: &SemanticModel<'db>,
    receiver: Type<'db>,
    name: &str,
) -> Option<Type<'db>> {
    receiver
        .member(model.db(), &model.program_environment(), name)
        .ignore_possibly_undefined()
}

/// The receiver `x` of an attribute access `x.attr`, in the terms of pyright's
/// `_getClassMemberAccessInfo`.
#[derive(Debug, Clone, Copy)]
pub struct PyrightReceiver {
    /// The receiver is a class object rather than an instance.
    pub is_class_object: bool,
    /// The receiver's class defines `__getattribute__` or `__getattr__` (ignoring `object`).
    pub has_magic_get: bool,
    /// The receiver's class defines `__setattr__` (ignoring `object`).
    pub has_magic_set: bool,
}

/// Describes the receiver of an attribute access, or returns `None` if pyright wouldn't treat it
/// as a class (for example, a union, a module or `Any`).
///
/// A type variable receiver is replaced by its upper bound. This includes `self` and `cls`, whose
/// synthesized `Self` type variable is bound to an instance of the enclosing class. Pyright only
/// looks for `__getattr__` and `__setattr__` on receivers that are classes or instances without
/// that replacement, so a type variable receiver never has them.
pub fn pyright_receiver<'db>(
    model: &SemanticModel<'db>,
    lhs: Type<'db>,
) -> Option<PyrightReceiver> {
    let db = model.db();
    let env = model.program_environment();

    let is_type_var = matches!(lhs, Type::TypeVar(_))
        || matches!(lhs, Type::SubclassOf(subclass_of)
            if matches!(subclass_of.subclass_of(), SubclassOfInner::TypeVar(_)));
    let (class, is_class_object) = match lhs {
        // A narrowed receiver such as `x` in `if x: x.attr` (`F & ~AlwaysFalsy`) or after
        // `isinstance(x, F)` for an unknown `x` (`Unknown & F`): pyright sees `F`.
        Type::Intersection(intersection) => {
            let narrowed = pyright_intersection_without_dynamic(db, intersection)
                .or_else(|| intersection.iter_positive(db).next())?;
            return if matches!(narrowed, Type::Intersection(_)) {
                None
            } else {
                pyright_receiver(model, narrowed)
            };
        }
        Type::Recursive(recursive) => {
            let unfolded = recursive.unfold(db, &env).into_unfolded()?;
            return if matches!(unfolded, Type::Recursive(_)) {
                None
            } else {
                pyright_receiver(model, unfolded)
            };
        }
        Type::TypeVar(type_var) => (type_var_bound_class(db, model, type_var)?, false),
        Type::SubclassOf(subclass_of) => match subclass_of.subclass_of() {
            SubclassOfInner::TypeVar(type_var) => {
                (type_var_bound_class(db, model, type_var)?, false)
            }
            SubclassOfInner::Class(class) => (class.class_literal(db), true),
            SubclassOfInner::Dynamic(_) => (
                ClassLiteral::Static(KnownClass::Type.try_to_class_literal(db, &env)?),
                false,
            ),
            SubclassOfInner::Protocol(_) => return None,
        },
        Type::ClassLiteral(class) => (class, true),
        Type::GenericAlias(alias) => (ClassLiteral::Static(alias.origin(db)), true),
        // Pyright evaluates `super()` to an instance (or the class) of the next class in the MRO.
        Type::BoundSuper(bound_super) => {
            return pyright_receiver(model, bound_super.owner(db).owner_type());
        }
        Type::NominalInstance(_)
        | Type::ProtocolInstance(_)
        | Type::TypedDict(_)
        | Type::NewTypeInstance(_)
        | Type::LiteralValue(_)
        | Type::PropertyInstance(_)
        | Type::SlotDescriptor(_)
        | Type::EnumComplement(_)
        | Type::TypeIs(_)
        | Type::TypeGuard(_)
        | Type::SpecialForm(_)
        | Type::KnownInstance(_) => (instance_class(db, model, lhs)?, false),
        _ => return None,
    };

    let has = |name: &str| {
        !is_type_var
            && !class
                .class_member(db, &env, name, MemberLookupPolicy::MRO_NO_OBJECT_FALLBACK)
                .place
                .is_undefined()
    };

    Some(PyrightReceiver {
        is_class_object,
        has_magic_get: has("__getattribute__") || has("__getattr__"),
        has_magic_set: has("__setattr__"),
    })
}

fn type_var_bound_class<'db>(
    db: &'db dyn Db,
    model: &SemanticModel<'db>,
    type_var: BoundTypeVarInstance<'db>,
) -> Option<ClassLiteral<'db>> {
    let bound = type_var
        .typevar(db)
        .upper_bound(db, &model.program_environment())?;
    instance_class(db, model, bound)
}

/// The result of pyright's descriptor analysis for a method declaration: whether it behaves like
/// a property, and the type a read (or write) through it produces.
#[derive(Debug, Clone, Copy)]
pub enum PyrightAccessor<'db> {
    /// An ordinary method.
    Method,
    /// A property or another descriptor.
    Accessor {
        is_readonly: bool,
        /// The getter's return type for reads, or the setter's value type for writes.
        effective_type: Option<Type<'db>>,
    },
}

/// Pyright's analysis of a method declaration's declared (decorated) type in
/// `_getFunctionTokenType`.
pub fn pyright_method_accessor<'db>(
    model: &SemanticModel<'db>,
    declaration: Definition<'db>,
    all_declarations: &[PyrightDeclaration<'db>],
    is_write: bool,
) -> PyrightAccessor<'db> {
    let db = model.db();
    let env = model.program_environment();
    let Some(declared) = pyright_definition_type(db, declaration) else {
        return PyrightAccessor::Method;
    };

    if declared.as_property_instance().is_some() {
        let accessor = all_declarations
            .iter()
            .filter_map(|declaration| declaration.definition)
            .find_map(|definition| {
                let property = pyright_definition_type(db, definition)?.as_property_instance()?;
                if is_write {
                    property.setter(db)
                } else {
                    property.getter(db)
                }
            });
        return PyrightAccessor::Accessor {
            is_readonly: all_declarations
                .iter()
                .all(|declaration| declaration.is_property_without_setter),
            effective_type: accessor.and_then(|accessor| accessor_type(model, accessor, is_write)),
        };
    }

    if !pyright_type(model, declared, PyrightTypeContext::Value).is_class() {
        return PyrightAccessor::Method;
    }

    // Pyright's `staticmethod` and `classmethod` decorators leave an object that isn't a function
    // as it is (`@staticmethod` over `@functools.cache`), while ty wraps it.
    if let Type::KnownInstance(KnownInstanceType::MethodWrapper(wrapper)) = declared
        && matches!(
            wrapper.class(db),
            KnownClass::Staticmethod | KnownClass::Classmethod
        )
    {
        return PyrightAccessor::Method;
    }

    let lookup = |name: &str| {
        declared
            .member_lookup_with_policy(db, &env, name, MemberLookupPolicy::NO_INSTANCE_FALLBACK)
            .place
            .ignore_possibly_undefined()
    };
    let set = lookup("__set__");
    let member = if is_write { set } else { lookup("__get__") };
    let Some(member) = member else {
        return PyrightAccessor::Method;
    };
    PyrightAccessor::Accessor {
        is_readonly: set.is_none(),
        effective_type: accessor_type(model, member, is_write),
    }
}

/// The return type of a getter, or the type of a setter's last parameter. Returns `None` for
/// overloaded accessors, like pyright does.
fn accessor_type<'db>(
    model: &SemanticModel<'db>,
    accessor: Type<'db>,
    is_write: bool,
) -> Option<Type<'db>> {
    let db = model.db();
    let function = match accessor {
        Type::FunctionLiteral(function) => function,
        Type::BoundMethod(method) => method.function(db)?,
        _ => return None,
    };
    let (overloads, _) = function.overloads_and_implementation(db);
    if !overloads.is_empty() {
        return None;
    }
    let signature = function.signature(db).overloads.first()?;
    if is_write {
        let [_, .., value] = signature.parameters().as_slice() else {
            return None;
        };
        Some(value.annotated_type())
    } else {
        inferred_return_type(model, function).or(Some(signature.return_ty))
    }
}

/// The return type pyright infers for calling `callee`, if it is a function without a return
/// annotation (see [`inferred_return_type`]).
pub fn pyright_inferred_call_type<'db>(
    model: &SemanticModel<'db>,
    callee: Type<'db>,
) -> Option<Type<'db>> {
    let function = match callee {
        Type::FunctionLiteral(function) => function,
        Type::BoundMethod(method) => method.function(model.db())?,
        _ => return None,
    };
    let (overloads, _) = function.overloads_and_implementation(model.db());
    if !overloads.is_empty() {
        return None;
    }
    inferred_return_type(model, function)
}

/// Pyright's inferred return type of a function without a return annotation, approximated as the
/// union of the types of its `return` values. Returns `None` for functions with a return
/// annotation, generators, and functions that don't return a value.
fn inferred_return_type<'db>(
    model: &SemanticModel<'db>,
    function: FunctionType<'db>,
) -> Option<Type<'db>> {
    let db = model.db();
    let definition = function.definition(db);
    let DefinitionKind::Function(node) = definition.kind(db) else {
        return None;
    };
    let file = definition.program_file(db);
    let parsed = parsed_module(db, file.python_file(db)).load(db);
    let node = node.node(&parsed);
    if node.returns.is_some() {
        return None;
    }
    let mut returns = ReturnStatementVisitor::default();
    returns.visit_body(&node.body);
    if returns.is_generator || returns.returns.is_empty() {
        return None;
    }
    let scope = semantic_index(db, file)
        .node_scope(NodeWithScopeRef::Function(node))
        .to_scope_id(db, file);
    let inference = infer_complete_scope_types(db, scope);
    let env = model.program_environment();
    let elements: Vec<Type<'db>> = returns
        .returns
        .iter()
        .map(|statement| match statement.value.as_deref() {
            Some(value) => inference.expression_type(value),
            None => Type::none(db, &env),
        })
        .collect();
    Some(UnionType::from_elements(db, &env, elements))
}

/// Whether a function's undecorated type is a static method, for a function definition.
pub fn pyright_function_definition_is_static<'db>(
    db: &'db dyn Db,
    definition: Definition<'db>,
) -> bool {
    let DefinitionKind::Function(function) = definition.kind(db) else {
        return false;
    };
    let parsed = parsed_module(db, definition.python_file(db)).load(db);
    // Pyright only treats `@staticmethod` as a static method inside a class body, and `__new__`
    // as a constructor rather than a static method.
    if function.node(&parsed).name.as_str() == "__new__"
        || definition.scope(db).scope(db).kind() != ScopeKind::Class
    {
        return false;
    }
    // Pyright reads the flag that `@staticmethod` sets on the undecorated function wherever the
    // decorator sits. ty's undecorated type loses the descriptor kind when a transforming decorator
    // such as `@cache` wraps the static method, so read the declaration instead.
    pyright_undecorated_type(db, definition)
        .as_function_literal()
        .is_some_and(|function| function.has_staticmethod_declaration(db))
}

/// The type of a function or class definition before its decorators are applied.
pub fn pyright_undecorated_type<'db>(db: &'db dyn Db, definition: Definition<'db>) -> Type<'db> {
    let inference = infer_definition_types(db, definition);
    inference
        .undecorated_type()
        .unwrap_or_else(|| inference.binding_type(definition))
}

/// Whether a parameter definition is the receiver (`self` or `cls`) of a method, following
/// pyright's `_getParamTokenType`: it must be the first parameter of a `def` directly inside a
/// class body, the `def` must not have PEP 695 type parameters, and the function must not be a
/// static method.
pub fn pyright_parameter_is_method_receiver<'db>(
    db: &'db dyn Db,
    definition: Definition<'db>,
) -> bool {
    let DefinitionKind::Parameter(parameter) = definition.kind(db) else {
        return false;
    };
    let file = definition.program_file(db);
    let index = semantic_index(db, file);
    let parsed = parsed_module(db, file.python_file(db)).load(db);

    let NodeWithScopeKind::Function(function) = index.scope(definition.file_scope(db)).node()
    else {
        return false;
    };
    let function_node = function.node(&parsed);
    if function_node.type_params.is_some() {
        return false;
    }
    let parameter_name_range = match parameter {
        ParameterDefinitionNodeKind::Parameter(parameter) => {
            parameter.node(&parsed).parameter.name.range()
        }
        ParameterDefinitionNodeKind::VariadicPositionalParameter(parameter)
        | ParameterDefinitionNodeKind::VariadicKeywordParameter(parameter) => {
            parameter.node(&parsed).name.range()
        }
    };
    // Pyright's parameter list includes the bare `*` of `def f(*, x)`, which ruff doesn't
    // represent, so keyword-only parameters are never first.
    let parameters = &function_node.parameters;
    let first_range = parameters
        .posonlyargs
        .iter()
        .chain(&parameters.args)
        .map(|parameter| parameter.parameter.name.range())
        .next()
        .or_else(|| parameters.vararg.as_ref().map(|vararg| vararg.name.range()))
        .or_else(|| {
            parameters
                .kwonlyargs
                .is_empty()
                .then(|| parameters.kwarg.as_ref().map(|kwarg| kwarg.name.range()))
                .flatten()
        });
    if first_range != Some(parameter_name_range) {
        return false;
    }

    let Some(function_definition) = index.try_definition(function_node) else {
        return false;
    };
    if function_definition.scope(db).scope(db).kind() != ScopeKind::Class {
        return false;
    }
    pyright_undecorated_type(db, function_definition)
        .as_function_literal()
        .is_some_and(|function| {
            function.name(db) == "__new__" || !function.has_staticmethod_declaration(db)
        })
}

/// The declarations and type of a keyword argument's name in a call, following pyright's
/// `getDeclInfoForNameNode` and the type it records for the name.
#[derive(Debug)]
pub struct PyrightKeywordArgument<'db> {
    pub definitions: Vec<ResolvedDefinition<'db>>,
    pub ty: Option<Type<'db>>,
}

/// Resolves the keyword argument names of a call, in the order of `call.arguments.keywords`.
///
/// Pyright only resolves keyword names for callees that are functions, bound methods,
/// overloads and classes (through `__init__`, or the fields of dataclasses and typed dicts). Its
/// type is the matched parameter's declared type, or the argument value's type when the
/// parameter is unannotated or the callee is unknown.
pub fn pyright_keyword_arguments<'db>(
    model: &SemanticModel<'db>,
    call: &ast::ExprCall,
) -> Vec<PyrightKeywordArgument<'db>> {
    let callee = call.func.inferred_type(model);
    // The call's signature is only resolved once, and only if a keyword needs it.
    let signature = OnceCell::new();
    call.arguments
        .keywords
        .iter()
        .map(|keyword| keyword_argument(model, keyword, call, callee, &signature))
        .collect()
}

fn keyword_argument<'db>(
    model: &SemanticModel<'db>,
    keyword: &ast::Keyword,
    call: &ast::ExprCall,
    callee: Option<Type<'db>>,
    signature: &OnceCell<Option<CallSignatureDetails<'db>>>,
) -> PyrightKeywordArgument<'db> {
    let db = model.db();

    // Pyright evaluates calls that create type variables specially and records nothing for their
    // keyword names (`bound=`, `covariant=`, ...).
    if let Some(Type::ClassLiteral(class)) = callee
        && matches!(
            class.known(db),
            Some(KnownClass::TypeVar | KnownClass::ParamSpec | KnownClass::TypeVarTuple)
        )
    {
        return PyrightKeywordArgument {
            definitions: Vec::new(),
            ty: None,
        };
    }

    // Pyright checks a call of `Any`, `Unknown` or `Never` against a dummy
    // `(*args: Any, **kwargs: Any)` signature, so every keyword name gets the declared `Any` of
    // `**kwargs`, whatever the argument's value is.
    if matches!(callee, Some(Type::Dynamic(_) | Type::Never)) {
        return PyrightKeywordArgument {
            definitions: Vec::new(),
            ty: Some(Type::any()),
        };
    }

    // `functools.partial(f, name=...)`: pyright resolves the keyword against the parameters of
    // `f` (or of its `__init__` if it's a class), but records no type for it.
    if let Some(Type::ClassLiteral(class)) = callee
        && class.known(db) == Some(KnownClass::FunctoolsPartial)
    {
        return PyrightKeywordArgument {
            definitions: partial_keyword_definitions(model, keyword, call),
            ty: None,
        };
    }

    let resolves = matches!(
        callee,
        Some(
            Type::FunctionLiteral(_)
                | Type::BoundMethod(_)
                | Type::ClassLiteral(_)
                | Type::GenericAlias(_)
        )
    );
    let definitions = if resolves {
        definitions_for_keyword_argument(model, keyword, call)
    } else {
        Vec::new()
    };

    // Pyright never records a type for the keyword names of named tuple constructors.
    let is_named_tuple = match callee {
        Some(Type::ClassLiteral(ClassLiteral::Static(class))) => {
            class.has_named_tuple_class_in_mro(db)
        }
        Some(Type::ClassLiteral(ClassLiteral::DynamicNamedTuple(_))) => true,
        _ => false,
    };
    if is_named_tuple {
        return PyrightKeywordArgument {
            definitions,
            ty: None,
        };
    }

    let argument_index = call
        .arguments
        .iter_source_order()
        .position(|argument| argument.range() == keyword.range());
    // The type of the parameter that the keyword matches in the overload that the call resolves
    // to, unless that type is unknown.
    let parameter_type = argument_index.and_then(|index| {
        let details = signature
            .get_or_init(|| resolved_call_signature(model, call))
            .as_ref()?;
        let parameter = (*details.argument_to_displayed_parameter_mapping.get(index)?)?;
        details
            .parameters
            .get(parameter)
            .map(|parameter| parameter.ty)
            .filter(|ty| !ty.is_unknown())
    });
    let ty = parameter_type.or_else(|| keyword.value.inferred_type(model));
    PyrightKeywordArgument { definitions, ty }
}

fn partial_keyword_definitions<'db>(
    model: &SemanticModel<'db>,
    keyword: &ast::Keyword,
    call: &ast::ExprCall,
) -> Vec<ResolvedDefinition<'db>> {
    let db = model.db();
    let env = model.program_environment();
    let Some(name) = keyword.arg.as_ref() else {
        return Vec::new();
    };
    let Some(target) = call
        .arguments
        .args
        .first()
        .and_then(|arg| arg.inferred_type(model))
    else {
        return Vec::new();
    };
    // For classes, pyright only looks at `__init__`.
    let target = match target {
        Type::ClassLiteral(_) | Type::GenericAlias(_) => {
            match target
                .member_lookup_with_policy(
                    db,
                    &env,
                    "__init__",
                    MemberLookupPolicy::MRO_NO_OBJECT_FALLBACK,
                )
                .place
                .ignore_possibly_undefined()
            {
                Some(init) => init,
                None => return Vec::new(),
            }
        }
        _ => target,
    };
    let Some(callables) = target.try_upcast_to_callable(db, &env) else {
        return Vec::new();
    };
    callables
        .signatures(db)
        .filter_map(|signature| {
            let (_, parameter) = signature.parameters().keyword_by_name(name.as_str())?;
            parameter.definition().map(ResolvedDefinition::Definition)
        })
        .collect()
}

/// All declarations of the symbol that `definition` binds, in pyright's order.
///
/// Pyright's class symbols also hold the attributes assigned through `self.name = ...` in the
/// class's methods, so for a definition in a class body those assignments are included too.
pub fn pyright_symbol_declarations<'db>(
    model: &SemanticModel<'db>,
    definition: Definition<'db>,
) -> Vec<PyrightDeclaration<'db>> {
    let definitions: Vec<_> = pyright_symbol_definitions(model.db(), definition)
        .into_iter()
        .map(ResolvedDefinition::Definition)
        .collect();
    pyright_declarations(model, &definitions)
}

/// The definitions of the symbol that `definition` binds, starting with `definition` itself.
pub fn pyright_symbol_definitions<'db>(
    db: &'db dyn Db,
    definition: Definition<'db>,
) -> Vec<Definition<'db>> {
    let scope = definition.scope(db);
    let ScopedPlaceId::Symbol(symbol) = definition.place(db) else {
        return vec![definition];
    };
    let use_def = use_def_map(db, scope);
    let mut definitions = FxIndexSet::default();
    definitions.insert(definition);
    definitions.extend(
        use_def
            .reachable_symbol_bindings(symbol)
            .filter_map(|binding| binding.binding.definition()),
    );
    definitions.extend(
        use_def
            .reachable_symbol_declarations(symbol)
            .filter_map(|declaration| declaration.declaration.definition()),
    );
    if scope.scope(db).kind() == ScopeKind::Class {
        let name = place_table(db, scope).symbol(symbol).name();
        definitions.extend(instance_attribute_definitions(db, scope, name));
    }
    // ty's loop headers and nested bindings are internal definitions that pyright doesn't have.
    definitions
        .into_iter()
        .filter(|definition| definition.kind(db).is_user_visible())
        .collect()
}

/// The bindings and declarations of `self.name` in the methods of the class with body `scope`.
fn instance_attribute_definitions<'db>(
    db: &'db dyn Db,
    class_scope: ScopeId<'db>,
    name: &str,
) -> Vec<Definition<'db>> {
    let index = semantic_index(db, class_scope.program_file(db));
    let mut definitions = Vec::new();
    for function_scope in attribute_scopes(db, class_scope) {
        let Some(member) = index
            .place_table(function_scope)
            .member_id_by_instance_attribute_name(name)
        else {
            continue;
        };
        let use_def = index.use_def_map(function_scope);
        definitions.extend(
            use_def
                .reachable_member_declarations(member)
                .filter_map(|declaration| declaration.declaration.definition()),
        );
        definitions.extend(
            use_def
                .reachable_member_bindings(member)
                .filter_map(|binding| binding.binding.definition()),
        );
    }
    definitions.retain(|definition| definition.kind(db).is_user_visible());
    definitions
}

/// Whether pyright may treat `definition` as a type alias: an explicit `X: TypeAlias = ...`, or an
/// assignment that pyright's binder considers a possible implicit type alias
/// (`isPossibleTypeAliasDeclaration`). That is a single name target outside of functions and
/// loops, which is the only declaration of its symbol, with a value that has the syntax of a type
/// expression. Pyright also requires that the value evaluates to a type, which the caller checks.
pub fn pyright_is_type_alias_declaration<'db>(
    db: &'db dyn Db,
    definition: Definition<'db>,
) -> bool {
    let file = definition.program_file(db);
    let parsed = parsed_module(db, file.python_file(db)).load(db);
    match definition.kind(db) {
        DefinitionKind::AnnotatedAssignment(assignment) => {
            assignment.value(&parsed).is_some()
                && match assignment.annotation(&parsed) {
                    ast::Expr::Name(name) => name.id.as_str() == "TypeAlias",
                    ast::Expr::Attribute(attribute) => attribute.attr.as_str() == "TypeAlias",
                    _ => false,
                }
        }
        DefinitionKind::Assignment(assignment) => {
            let target = assignment.target(&parsed);
            assignment.unpack().is_none()
                && target.is_name_expr()
                && is_type_alias_expression(assignment.value(&parsed), false)
                && !matches!(
                    definition.scope(db).scope(db).kind(),
                    ScopeKind::Function | ScopeKind::Lambda
                )
                && !is_within_loop(db, definition, &parsed, target.range())
                && pyright_symbol_definitions(db, definition).len() == 1
        }
        _ => false,
    }
}

/// Pyright's `_isLegalTypeAliasExpressionForm`: whether an expression has the syntax of a type.
fn is_type_alias_expression(expr: &ast::Expr, allow_string: bool) -> bool {
    match expr {
        ast::Expr::Name(_) | ast::Expr::NoneLiteral(_) => true,
        ast::Expr::StringLiteral(_) => allow_string,
        ast::Expr::BinOp(binary) => {
            binary.op == ast::Operator::BitOr
                && is_type_alias_expression(&binary.left, true)
                && is_type_alias_expression(&binary.right, true)
        }
        ast::Expr::Subscript(subscript) => is_type_alias_expression(&subscript.value, allow_string),
        ast::Expr::Attribute(attribute) => is_type_alias_expression(&attribute.value, allow_string),
        _ => false,
    }
}

/// Whether the statement at `range` is inside a `for` or `while` loop of the definition's scope.
fn is_within_loop<'db>(
    db: &'db dyn Db,
    definition: Definition<'db>,
    parsed: &ParsedModuleRef,
    range: TextRange,
) -> bool {
    fn contains_loop(body: &[ast::Stmt], range: TextRange) -> bool {
        let Some(statement) = body
            .iter()
            .find(|statement| statement.range().contains_range(range))
        else {
            return false;
        };
        match statement {
            ast::Stmt::For(_) | ast::Stmt::While(_) => true,
            ast::Stmt::If(if_stmt) => {
                contains_loop(&if_stmt.body, range)
                    || if_stmt
                        .elif_else_clauses
                        .iter()
                        .any(|clause| contains_loop(&clause.body, range))
            }
            ast::Stmt::With(with_stmt) => contains_loop(&with_stmt.body, range),
            ast::Stmt::Try(try_stmt) => {
                contains_loop(&try_stmt.body, range)
                    || try_stmt.handlers.iter().any(|handler| {
                        let ast::ExceptHandler::ExceptHandler(handler) = handler;
                        contains_loop(&handler.body, range)
                    })
                    || contains_loop(&try_stmt.orelse, range)
                    || contains_loop(&try_stmt.finalbody, range)
            }
            ast::Stmt::Match(match_stmt) => match_stmt
                .cases
                .iter()
                .any(|case| contains_loop(&case.body, range)),
            _ => false,
        }
    }

    match definition.scope(db).node(db) {
        NodeWithScopeKind::Module => contains_loop(parsed.suite(), range),
        NodeWithScopeKind::Class(class) => contains_loop(&class.node(parsed).body, range),
        _ => false,
    }
}

/// Whether an attribute is assigned from a parameter that pyright types with a synthesized type
/// variable because the class is "pseudo-generic".
///
/// Pyright treats a class as pseudo-generic when it isn't generic, isn't defined in a stub, has a
/// single `__init__` with more than one parameter and none of those parameters is annotated.
/// The parameters without default values then get a synthesized type variable instead of an
/// unknown type. For example, `self.path` below has a known (if generic) type in pyright:
///
/// ```python
/// class File:
///     def __init__(self, path, mode="rb"):
///         self.path = path
/// ```
pub fn pyright_is_pseudo_generic_attribute<'db>(
    model: &SemanticModel<'db>,
    declarations: &[PyrightDeclaration<'db>],
) -> bool {
    let db = model.db();
    declarations
        .iter()
        .filter_map(|declaration| declaration.definition)
        .any(|definition| assigns_pseudo_generic_parameter(db, definition))
}

fn assigns_pseudo_generic_parameter<'db>(db: &'db dyn Db, definition: Definition<'db>) -> bool {
    let DefinitionKind::Assignment(assignment) = definition.kind(db) else {
        return false;
    };
    if !matches!(definition.place(db), ScopedPlaceId::Member(_)) {
        return false;
    }
    let file = definition.program_file(db);
    if file.file(db).is_stub(db) {
        return false;
    }
    let index = semantic_index(db, file);
    let parsed = parsed_module(db, file.python_file(db)).load(db);
    let NodeWithScopeKind::Function(function) = index.scope(definition.file_scope(db)).node()
    else {
        return false;
    };
    let init = function.node(&parsed);
    if init.name.as_str() != "__init__" {
        return false;
    }
    let ast::Expr::Name(value) = assignment.value(&parsed) else {
        return false;
    };

    let parameters = &init.parameters;
    if parameters.len() < 2
        || parameters
            .iter_source_order()
            .any(|parameter| parameter.as_parameter().annotation.is_some())
    {
        return false;
    }
    let is_default_less_parameter = parameters
        .posonlyargs
        .iter()
        .chain(&parameters.args)
        .chain(&parameters.kwonlyargs)
        .skip(1)
        .any(|parameter| {
            parameter.default.is_none() && parameter.parameter.name.as_str() == value.id.as_str()
        });
    if !is_default_less_parameter {
        return false;
    }

    // The enclosing class must not be generic, and this `__init__` must be the only typed
    // declaration of the class's `__init__` symbol (which includes definitions nested in `if`
    // or `try` statements of the class body).
    let Some(init_definition) = index.try_definition(init) else {
        return false;
    };
    let NodeWithScopeKind::Class(class_node) = init_definition.scope(db).node(db) else {
        return false;
    };
    let Some(ClassLiteral::Static(class)) = index
        .try_definition(class_node.node(&parsed))
        .and_then(|class_definition| original_class_type(db, class_definition))
    else {
        return false;
    };
    class.generic_context(db).is_none()
        && own_member_definitions(db, class, "__init__")
            .into_iter()
            .filter(|definition| pyright_has_declared_type(db, *definition))
            .eq([init_definition])
}

/// The type that a definition binds, or the declared type of a declaration without a value
/// (`x: int` outside a stub). Returns `None` for a declaration whose annotation ty rejects.
pub fn pyright_definition_type<'db>(
    db: &'db dyn Db,
    definition: Definition<'db>,
) -> Option<Type<'db>> {
    let file = definition.program_file(db);
    let parsed = parsed_module(db, file.python_file(db)).load(db);
    if definition
        .kind(db)
        .category(file.file(db).is_stub(db), &parsed)
        .is_binding()
    {
        Some(binding_type(db, definition))
    } else {
        inferred_declaration(db, definition)
            .declared()
            .map(|declared| declared.inner_type())
    }
}

/// The declarations of the attribute in `x.name`, the way pyright looks them up.
///
/// For classes and instances, pyright first searches the whole MRO for a class whose member has
/// a declaration with a type annotation, and returns only the annotated declarations. Only if
/// there is none does it take the first class in the MRO that defines the member at all. A class
/// member's declarations include both the class-body ones and the `self.name = ...` assignments
/// in methods. For example, `y.value` below resolves to the annotated declaration in `Base`:
///
/// ```python
/// class Base:
///     value: str
///
/// class Derived(Base):
///     def __init__(self):
///         self.value = "x"
///
/// y = Derived()
/// y.value
/// ```
pub fn pyright_attribute_declarations<'db>(
    model: &SemanticModel<'db>,
    receiver: Type<'db>,
    name: &str,
) -> Vec<PyrightDeclaration<'db>> {
    let db = model.db();
    let env = model.program_environment();
    let elements = match receiver {
        Type::Union(union) => union.elements(db).to_vec(),
        receiver => vec![receiver],
    };

    // Pyright concatenates the declarations found for each member of a union receiver. Members
    // that are instances of the same class (`Literal["a", "b"]`) find the same declarations.
    let mut declarations = Vec::new();
    let mut lookup_classes = FxIndexSet::default();
    for element in elements {
        // Pyright finds no declarations through the `Any` and unknown members of a union.
        if matches!(element, Type::Dynamic(_) | Type::Divergent(_)) {
            continue;
        }
        let lookup = member_lookup_class(db, model, element);
        if lookup.is_some() && !lookup_classes.insert(lookup) {
            continue;
        }
        let found = match lookup {
            Some(lookup) => class_member_declarations(model, lookup, name),
            None => Vec::new(),
        };
        if found.is_empty() {
            // Modules, and members that only the metaclass defines.
            declarations.extend(pyright_declarations(
                model,
                &definition_resolution::definitions_for_attribute(db, &env, element, name),
            ));
        } else {
            declarations.extend(found);
        }
    }
    declarations
}

/// The synthesized declaration of a field of a functional named tuple
/// (`NT = namedtuple("NT", ["x"])`), for `ty` being the named tuple class or an instance of it.
pub fn pyright_functional_named_tuple_field<'db>(
    model: &SemanticModel<'db>,
    ty: Type<'db>,
    name: &str,
) -> Option<PyrightDeclaration<'db>> {
    let db = model.db();
    let class = match ty {
        Type::ClassLiteral(class) => class,
        _ => instance_class(db, model, ty)?,
    };
    let ClassLiteral::DynamicNamedTuple(named_tuple) = class else {
        return None;
    };
    named_tuple
        .field(db, &Name::new(name))
        .map(|_| PyrightDeclaration::synthesized_variable(false, false))
}

/// Where pyright looks up the members of a type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct MemberLookup<'db> {
    /// The class whose MRO pyright searches: the class of an instance, the class itself for a
    /// class object, and the upper bound of a type variable.
    class: ClassLiteral<'db>,
    /// The members are looked up on a class object rather than an instance.
    is_class_object: bool,
    /// For `super()`, the class after which the search of the MRO starts.
    after: Option<ClassLiteral<'db>>,
}

fn member_lookup_class<'db>(
    db: &'db dyn Db,
    model: &SemanticModel<'db>,
    ty: Type<'db>,
) -> Option<MemberLookup<'db>> {
    let instance = |class| MemberLookup {
        class,
        is_class_object: false,
        after: None,
    };
    let class_object = |class| MemberLookup {
        class,
        is_class_object: true,
        after: None,
    };
    match ty {
        Type::TypeVar(type_var) => type_var_bound_class(db, model, type_var).map(instance),
        Type::SubclassOf(subclass_of) => match subclass_of.subclass_of() {
            SubclassOfInner::TypeVar(type_var) => {
                type_var_bound_class(db, model, type_var).map(class_object)
            }
            SubclassOfInner::Class(class) => Some(class_object(class.class_literal(db))),
            // Pyright looks up the members of `type[Any]` on `type` itself.
            SubclassOfInner::Dynamic(_) => KnownClass::Type
                .try_to_class_literal(db, &model.program_environment())
                .map(|class| instance(ClassLiteral::Static(class))),
            SubclassOfInner::Protocol(_) => None,
        },
        Type::ClassLiteral(class) => Some(class_object(class)),
        Type::GenericAlias(alias) => Some(class_object(ClassLiteral::Static(alias.origin(db)))),
        // `super()` searches the MRO of the owner's class after the class it's called in.
        Type::BoundSuper(bound_super) => {
            let owner = member_lookup_class(db, model, bound_super.owner(db).owner_type())?;
            let pivot = bound_super.pivot_class(db).into_class()?.class_literal(db);
            Some(MemberLookup {
                after: Some(pivot),
                ..owner
            })
        }
        // A narrowed receiver (`F & ~AlwaysFalsy`, `Unknown & F`) is looked up on `F`.
        Type::Intersection(intersection) => {
            let narrowed = pyright_intersection_without_dynamic(db, intersection)
                .or_else(|| intersection.iter_positive(db).next())?;
            if matches!(narrowed, Type::Intersection(_)) {
                None
            } else {
                member_lookup_class(db, model, narrowed)
            }
        }
        Type::ModuleLiteral(_) | Type::Dynamic(_) | Type::Union(_) => None,
        _ => instance_class(db, model, ty).map(instance),
    }
}

/// Pyright's `lookUpClassMember` (or `lookUpObjectMember`) followed by `_addSymbolDeclInfo` with
/// `preferTypedDeclarations`, for a member of a class or of an instance of it.
fn class_member_declarations<'db>(
    model: &SemanticModel<'db>,
    lookup: MemberLookup<'db>,
    name: &str,
) -> Vec<PyrightDeclaration<'db>> {
    let db = model.db();
    let to_declarations = |definitions: Vec<Definition<'db>>| {
        let definitions: Vec<_> = definitions
            .into_iter()
            .map(ResolvedDefinition::Definition)
            .collect();
        pyright_declarations(model, &definitions)
    };
    let typed = |definitions: &[Definition<'db>]| -> Vec<Definition<'db>> {
        definitions
            .iter()
            .copied()
            .filter(|definition| pyright_has_declared_type(db, *definition))
            .collect()
    };
    let static_mro = |class: ClassLiteral<'db>| {
        class
            .iter_mro(db)
            .filter_map(ClassBase::into_class)
            .filter_map(move |class| class.static_class_literal(db).map(|(literal, _)| literal))
    };
    let class = lookup.class;

    if let Some(field) =
        pyright_functional_named_tuple_field(model, Type::ClassLiteral(class), name)
    {
        return vec![field];
    }

    // Pyright first looks for an *instance* member in the MRO of a metaclass other than `type`,
    // for class objects and instances alike, and ignoring `DeclaredTypesOnly`. For example,
    // `cls._latest` finds the `cls._latest = token` assignment in a method of the metaclass
    // before the class's own `_latest = None`, and `x.__doc__` for an `abc.ABC` subclass finds
    // `object.__doc__: str | None` through `ABCMeta`.
    if lookup.after.is_none()
        && let Type::ClassLiteral(metaclass) = class.metaclass(db)
        && !metaclass.is_known(db, KnownClass::Type)
        && let Some(definitions) = static_mro(metaclass)
            .map(|ancestor| own_member_definitions(db, ancestor, name))
            .find(|definitions| {
                definitions
                    .iter()
                    .any(|definition| is_instance_member_declaration(db, *definition))
            })
    {
        let typed_definitions = typed(&definitions);
        return to_declarations(if typed_definitions.is_empty() {
            definitions
        } else {
            typed_definitions
        });
    }

    // Pyright's binder gives every class implicit, typed `__doc__` and `__module__` members, and a
    // `__qualname__` that only the class object has, so the typed lookup stops at the receiver's
    // own class. Declarations in the class body follow the implicit one, but a lookup on the
    // class object skips those that declare instance members (`__module__: str` in `object`).
    if lookup.after.is_none()
        && (matches!(name, "__doc__" | "__module__")
            || (name == "__qualname__" && lookup.is_class_object))
    {
        let mut declarations = vec![PyrightDeclaration::synthesized_variable(false, false)];
        if let ClassLiteral::Static(class) = class {
            let own = own_member_definitions(db, class, name)
                .into_iter()
                .filter(|definition| {
                    !lookup.is_class_object || !is_instance_member_declaration(db, *definition)
                })
                .collect();
            declarations.extend(to_declarations(own));
        }
        return declarations;
    }

    // `DeclaredTypesOnly`: the first class whose member has a typed declaration. Otherwise the
    // first class that has the member at all, including through `__slots__`.
    let mut ancestors = static_mro(class);
    if let Some(after) = lookup.after {
        ancestors
            .by_ref()
            .find(|ancestor| ClassLiteral::Static(*ancestor) == after);
    }
    let mut fallback = None;
    for ancestor in ancestors {
        let definitions = own_member_definitions(db, ancestor, name);
        let typed_definitions = typed(&definitions);
        if !typed_definitions.is_empty() {
            return to_declarations(typed_definitions);
        }
        if fallback.is_none() {
            let has_slot = class_slots_contain(db, ancestor, name);
            if !definitions.is_empty() || has_slot {
                fallback = Some((definitions, has_slot));
            }
        }
    }
    let Some((definitions, has_slot)) = fallback else {
        return Vec::new();
    };
    // Pyright's binder declares each `__slots__` entry after the class body and before the
    // assignments in methods.
    let mut declarations = to_declarations(definitions);
    if has_slot {
        let position = declarations
            .iter()
            .position(|declaration| !declaration.in_class_body)
            .unwrap_or(declarations.len());
        declarations.insert(
            position,
            PyrightDeclaration::synthesized_variable(true, is_constant_name(name)),
        );
    }
    declarations
}

/// Whether pyright marks a class member declared by `definition` as an instance member: an
/// attribute assigned through `self` in a method, or a variable annotated in the class body that
/// is neither a `ClassVar` nor a `Final` with a value.
fn is_instance_member_declaration<'db>(db: &'db dyn Db, definition: Definition<'db>) -> bool {
    if matches!(definition.place(db), ScopedPlaceId::Member(_)) {
        return true;
    }
    let DefinitionKind::AnnotatedAssignment(assignment) = definition.kind(db) else {
        return false;
    };
    let qualifiers = inferred_declaration(db, definition)
        .declared()
        .map_or(TypeQualifiers::empty(), |declared| declared.qualifiers());
    if qualifiers.contains(TypeQualifiers::CLASS_VAR) {
        return false;
    }
    let parsed = parsed_module(db, definition.python_file(db)).load(db);
    !(qualifiers.contains(TypeQualifiers::FINAL) && assignment.value(&parsed).is_some())
}

/// Whether the last `__slots__` assignment in the class body lists `name` (as a string, or in a
/// list, tuple or dict of strings).
fn class_slots_contain<'db>(db: &'db dyn Db, class: StaticClassLiteral<'db>, name: &str) -> bool {
    let class_scope = class.body_scope(db);
    let NodeWithScopeKind::Class(class_node) = class_scope.node(db) else {
        return false;
    };
    let parsed = parsed_module(db, class_scope.program_file(db).python_file(db)).load(db);
    let slots = class_node
        .node(&parsed)
        .body
        .iter()
        .filter_map(|statement| match statement {
            ast::Stmt::Assign(assign)
                if matches!(assign.targets.as_slice(), [ast::Expr::Name(target)] if target.id.as_str() == "__slots__") =>
            {
                Some(&*assign.value)
            }
            ast::Stmt::AnnAssign(assign)
                if matches!(&*assign.target, ast::Expr::Name(target) if target.id.as_str() == "__slots__") =>
            {
                assign.value.as_deref()
            }
            _ => None,
        })
        .next_back();
    let is_name = |expr: &ast::Expr| {
        expr.as_string_literal_expr()
            .is_some_and(|string| string.value.to_str() == name)
    };
    match slots {
        Some(slot @ ast::Expr::StringLiteral(_)) => is_name(slot),
        Some(ast::Expr::List(list)) => list.elts.iter().any(is_name),
        Some(ast::Expr::Tuple(tuple)) => tuple.elts.iter().any(is_name),
        Some(ast::Expr::Dict(dict)) => dict
            .items
            .iter()
            .any(|item| item.key.as_ref().is_some_and(is_name)),
        _ => false,
    }
}

/// The definitions of a member in one class: the class-body ones, then the instance attributes
/// assigned in the class's methods.
fn own_member_definitions<'db>(
    db: &'db dyn Db,
    class: StaticClassLiteral<'db>,
    name: &str,
) -> Vec<Definition<'db>> {
    let mut definitions = Vec::new();
    let mut push = |definition: Definition<'db>| {
        if definition.kind(db).is_user_visible() && !definitions.contains(&definition) {
            definitions.push(definition);
        }
    };
    let class_scope = class.body_scope(db);
    if let Some(symbol) = place_table(db, class_scope).symbol_id(name) {
        let use_def = use_def_map(db, class_scope);
        for declaration in use_def.reachable_symbol_declarations(symbol) {
            if let Some(definition) = declaration.declaration.definition() {
                push(definition);
            }
        }
        for binding in use_def.reachable_symbol_bindings(symbol) {
            if let Some(definition) = binding.binding.definition() {
                push(definition);
            }
        }
    }
    for definition in instance_attribute_definitions(db, class_scope, name) {
        push(definition);
    }
    definitions
}

/// Pyright's `hasTypeForDeclaration`: functions, classes, type aliases, type parameters, and
/// annotated variables and parameters have a declared type. Imports don't.
pub fn pyright_has_declared_type<'db>(db: &'db dyn Db, definition: Definition<'db>) -> bool {
    match definition.kind(db) {
        DefinitionKind::Function(_)
        | DefinitionKind::Class(_)
        | DefinitionKind::TypeAlias(_)
        | DefinitionKind::TypeVar(_)
        | DefinitionKind::ParamSpec(_)
        | DefinitionKind::TypeVarTuple(_)
        | DefinitionKind::AnnotatedAssignment(_) => true,
        DefinitionKind::Parameter(parameter) => {
            let parsed = parsed_module(db, definition.python_file(db)).load(db);
            match parameter {
                ParameterDefinitionNodeKind::Parameter(parameter) => {
                    parameter.node(&parsed).parameter.annotation.is_some()
                }
                ParameterDefinitionNodeKind::VariadicPositionalParameter(parameter)
                | ParameterDefinitionNodeKind::VariadicKeywordParameter(parameter) => {
                    parameter.node(&parsed).annotation.is_some()
                }
            }
        }
        _ => false,
    }
}

/// Whether pyright evaluates the subscript of `base` in a value expression as type arguments,
/// for a generic class (`list[int]`). A class whose metaclass defines `__getitem__` (like an
/// enum class) is called instead, and other classes take values.
pub fn pyright_is_generic_class_subscript<'db>(
    model: &SemanticModel<'db>,
    base: Type<'db>,
) -> bool {
    let db = model.db();
    let env = model.program_environment();
    match base {
        Type::GenericAlias(_) => true,
        Type::ClassLiteral(class) => {
            if let Type::ClassLiteral(metaclass) = class.metaclass(db)
                && !metaclass.is_known(db, KnownClass::Type)
                && !metaclass
                    .class_member(
                        db,
                        &env,
                        "__getitem__",
                        MemberLookupPolicy::MRO_NO_OBJECT_FALLBACK,
                    )
                    .place
                    .is_undefined()
            {
                return false;
            }
            matches!(class, ClassLiteral::Static(class) if class.generic_context(db).is_some())
        }
        _ => false,
    }
}

/// The type pyright infers for a `__slots__` entry without an assignment: the type of the string
/// that names it.
pub fn pyright_slot_type<'db>(model: &SemanticModel<'db>) -> Type<'db> {
    KnownClass::Str.to_instance(model.db(), &model.program_environment())
}

/// The first reachable definition of the symbol that `name` refers to, found through the
/// enclosing scopes of the current file without code flow, the way pyright looks symbols up in
/// code that it doesn't bind. Returns `None` for names that aren't defined in the file (such as
/// builtins).
pub fn pyright_symbol_definition<'db>(
    model: &SemanticModel<'db>,
    name: &ast::ExprName,
) -> Option<Definition<'db>> {
    let db = model.db();
    let index = semantic_index(db, model.program_file());
    let use_scope = model.scope(name.into())?;
    let mut is_global = false;
    for (scope_id, scope) in index.ancestor_scopes(use_scope) {
        // Class bodies aren't visible from nested scopes.
        if (is_global && scope.kind() != ScopeKind::Module)
            || (scope_id != use_scope && scope.kind() == ScopeKind::Class)
        {
            continue;
        }
        let table = index.place_table(scope_id);
        let Some(symbol_id) = table.symbol_id(name.id.as_str()) else {
            continue;
        };
        let symbol = table.symbol(symbol_id);
        if symbol.is_global() {
            is_global = true;
            continue;
        }
        if !symbol.is_local() {
            continue;
        }
        let use_def = index.use_def_map(scope_id);
        return use_def
            .reachable_symbol_bindings(symbol_id)
            .filter_map(|binding| binding.binding.definition())
            .chain(
                use_def
                    .reachable_symbol_declarations(symbol_id)
                    .filter_map(|declaration| declaration.declaration.definition()),
            )
            .find(|definition| definition.kind(db).is_user_visible());
    }
    None
}

/// The union of `elements`, the way pyright combines the types of a symbol's declarations.
pub fn pyright_union<'db>(
    model: &SemanticModel<'db>,
    elements: impl IntoIterator<Item = Type<'db>>,
) -> Type<'db> {
    UnionType::from_elements(model.db(), &model.program_environment(), elements)
}

/// Whether `ty` is `type[Any]` or `type[Unknown]`.
pub fn pyright_is_dynamic_class_object(ty: Type<'_>) -> bool {
    matches!(ty, Type::SubclassOf(subclass_of)
        if matches!(subclass_of.subclass_of(), SubclassOfInner::Dynamic(_)))
}
