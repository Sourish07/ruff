//! This module walks the AST and collects a set of "semantic tokens" for a file
//! or a range within a file. Each semantic token provides a "token type" and zero
//! or more "modifiers". This information can be used by an editor to provide
//! color coding based on semantic meaning.
//!
//! The classification follows basedpyright's semantic token provider, so that editors highlight
//! code the same way with either language server. basedpyright decides the token for every name
//! from the declarations of the symbol it refers to and from the type it evaluates to; the
//! functions below mirror that decision procedure (`_visitNameWithDeclarations`,
//! `_getVariableTokenType`, `_getFunctionTokenType`, `_getClassTokenType` and
//! `_getParamTokenType` in basedpyright's `semanticTokensWalker.ts`). The pyright-shaped facts
//! that feed it come from `ty_python_semantic::types::ide_support`.
//!
//! Like basedpyright, the provider only emits tokens for names, for the `@` of decorators and
//! for the `type` keyword of type alias statements. Strings, numbers, keywords and operators are
//! left to the editor's syntax highlighting.

mod reachability;

use reachability::StaticConditions;

use std::cell::RefCell;
use std::ops::Deref;
use std::rc::Rc;

use ruff_db::files::File;
use rustc_hash::FxHashMap;

use bitflags::bitflags;
use ruff_db::parsed::parsed_module;
use ruff_db::source::source_text;
use ruff_python_ast::name::Name;
use ruff_python_ast::visitor::source_order::{
    SourceOrderVisitor, TraversalSignal, walk_expr, walk_pattern, walk_stmt,
};
use ruff_python_ast::{self as ast, AnyNodeRef, Expr, ExprContext, ExprRef, Stmt, TypeParam};
use ruff_text_size::{Ranged, TextRange, TextSize};
use ty_python_core::definition::{Definition, DefinitionKind, ParameterDefinitionNodeKind};
use ty_python_core::place::ScopedPlaceId;
use ty_python_core::scope::{FileScopeId, ScopeId, ScopeKind};
use ty_python_core::{ProgramFile, semantic_index};
use ty_python_semantic::types::ide_support::pyright_tokens::{
    PyrightAccessor, PyrightDeclaration, PyrightDeclarationKind, PyrightFunction,
    PyrightKeywordArgument, PyrightType, PyrightTypeCategory, PyrightTypeContext,
    pyright_attribute_declarations, pyright_call_return_type, pyright_declaration,
    pyright_declarations, pyright_declared_type, pyright_definition_type,
    pyright_function_definition_is_static, pyright_functional_named_tuple_field,
    pyright_has_declared_type, pyright_hasattr_receiver, pyright_inferred_call_type,
    pyright_is_dynamic_class_object, pyright_is_explicit_any, pyright_is_generic_class_subscript,
    pyright_is_in_typing_stub, pyright_is_narrowed_not_none, pyright_is_pseudo_generic_attribute,
    pyright_is_type_alias_declaration, pyright_is_type_form_variable_type,
    pyright_keyword_arguments, pyright_member_type, pyright_method_accessor,
    pyright_narrowed_receiver_member_type, pyright_parameter_is_method_receiver, pyright_receiver,
    pyright_slot_type, pyright_symbol_declarations, pyright_symbol_definition, pyright_type,
    pyright_undecorated_type, pyright_union, pyright_widen_literal,
};
use ty_python_semantic::types::ide_support::{
    CallArgumentForm, UnreachableRange, call_argument_forms, unreachable_ranges,
};
use ty_python_semantic::types::{SpecialFormType, Type};
use ty_python_semantic::{
    HasDefinition, HasType, ImportAliasResolution, ResolvedDefinition, SemanticModel,
    definitions_for_imported_symbol, definitions_for_name,
};

use crate::Db;

/// Semantic token types supported by the language server.
///
/// The order matches basedpyright's legend, followed by the types that only ty uses.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SemanticTokenType {
    // This enum must be kept in sync with `all` and `as_lsp_concept` below.
    Namespace,
    Type,
    Class,
    Enum,
    TypeParameter,
    Parameter,
    Variable,
    Property,
    EnumMember,
    Function,
    Method,
    Keyword,
    Decorator,
    SelfParameter,
    ClsParameter,
    /// A string in the metadata of `Annotated[...]`, such as the shape of a jaxtyping array
    /// (`Float[Tensor, "batch seq"]`). basedpyright has no token for these, and editors color
    /// them like the type around them.
    String,
}

impl SemanticTokenType {
    /// Returns all supported semantic token types as enum variants.
    pub const fn all() -> [SemanticTokenType; 16] {
        [
            SemanticTokenType::Namespace,
            SemanticTokenType::Type,
            SemanticTokenType::Class,
            SemanticTokenType::Enum,
            SemanticTokenType::TypeParameter,
            SemanticTokenType::Parameter,
            SemanticTokenType::Variable,
            SemanticTokenType::Property,
            SemanticTokenType::EnumMember,
            SemanticTokenType::Function,
            SemanticTokenType::Method,
            SemanticTokenType::Keyword,
            SemanticTokenType::Decorator,
            SemanticTokenType::SelfParameter,
            SemanticTokenType::ClsParameter,
            SemanticTokenType::String,
        ]
    }

    /// Converts this semantic token type to its LSP string representation.
    /// Some of these are standardized terms in the LSP specification,
    /// while others are specific to the ty language server. It's important
    /// to use the standardized ones where possible because clients can
    /// use these for standardized color coding and syntax highlighting.
    /// For details, refer to this LSP specification:
    /// <https://microsoft.github.io/language-server-protocol/specifications/lsp/3.17/specification/#semanticTokenTypes>
    pub const fn as_lsp_concept(&self) -> &'static str {
        match self {
            SemanticTokenType::Namespace => "namespace",
            SemanticTokenType::Type => "type",
            SemanticTokenType::Class => "class",
            SemanticTokenType::Enum => "enum",
            SemanticTokenType::TypeParameter => "typeParameter",
            SemanticTokenType::Parameter => "parameter",
            SemanticTokenType::Variable => "variable",
            SemanticTokenType::Property => "property",
            SemanticTokenType::EnumMember => "enumMember",
            SemanticTokenType::Function => "function",
            SemanticTokenType::Method => "method",
            SemanticTokenType::Keyword => "keyword",
            SemanticTokenType::Decorator => "decorator",
            SemanticTokenType::SelfParameter => "selfParameter",
            SemanticTokenType::ClsParameter => "clsParameter",
            SemanticTokenType::String => "string",
        }
    }
}

bitflags! {
    /// Semantic token modifiers using bit flags.
    ///
    /// Bit `i` corresponds to entry `i` of [`SemanticTokenModifier::all_names`].
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    pub struct SemanticTokenModifier: u32 {
        const DECLARATION = 1 << 0;
        const DEFINITION = 1 << 1;
        const READONLY = 1 << 2;
        const STATIC = 1 << 3;
        const ASYNC = 1 << 4;
        const DEFAULT_LIBRARY = 1 << 5;
        /// A symbol from the `builtins` module.
        const BUILTIN = 1 << 6;
        /// A class or instance attribute (including methods and properties).
        const CLASS_MEMBER = 1 << 7;
        /// A parameter, or the name of a keyword argument, regardless of its token type.
        const PARAMETER = 1 << 8;
    }
}

impl SemanticTokenModifier {
    /// Returns all supported token modifiers for LSP capabilities.
    /// Some of these are standardized terms in the LSP specification,
    /// while others may be specific to the ty language server. It's
    /// important to use the standardized ones where possible because
    /// clients can use these for standardized color coding and syntax
    /// highlighting. For details, refer to this LSP specification:
    /// <https://microsoft.github.io/language-server-protocol/specifications/lsp/3.17/specification/#semanticTokenModifiers>
    pub fn all_names() -> Vec<&'static str> {
        vec![
            "declaration",
            "definition",
            "readonly",
            "static",
            "async",
            "defaultLibrary",
            "builtin",
            "classMember",
            "parameter",
        ]
    }
}

/// A semantic token with its position and classification.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SemanticToken {
    pub range: TextRange,
    pub token_type: SemanticTokenType,
    pub modifiers: SemanticTokenModifier,
}

impl Ranged for SemanticToken {
    fn range(&self) -> TextRange {
        self.range
    }
}

/// The result of semantic tokenization.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SemanticTokens {
    tokens: Vec<SemanticToken>,
}

impl SemanticTokens {
    /// Create a new `SemanticTokens` instance.
    fn new(tokens: Vec<SemanticToken>) -> Self {
        Self { tokens }
    }
}

impl Deref for SemanticTokens {
    type Target = [SemanticToken];

    fn deref(&self) -> &Self::Target {
        &self.tokens
    }
}

/// Generates semantic tokens for a Python file within the specified range.
/// Pass None to get tokens for the entire file.
pub fn semantic_tokens(
    db: &dyn Db,
    file: ProgramFile<'_>,
    range: Option<TextRange>,
) -> SemanticTokens {
    let parsed = parsed_module(db, file.python_file(db)).load(db);
    let model = SemanticModel::new(db, file);

    let mut visitor = SemanticTokenVisitor::new(&model, range);
    visitor.visit_body(parsed.suite());

    let mut tokens = visitor.tokens;
    // Declaration tokens are emitted before the decorators that precede them. A stable sort keeps
    // tokens that start at the same offset in the order they were emitted.
    tokens.sort_by_key(SemanticToken::start);
    SemanticTokens::new(tokens)
}

/// Where a name occurs, which affects its classification.
#[derive(Debug, Clone, Copy)]
struct NameContext<'db> {
    /// The name is the name of a keyword argument, as in `f(name=...)`.
    is_keyword_argument: bool,
    /// For the attribute name in `x.name`, the type of `x`.
    receiver: Option<Type<'db>>,
    /// The name is written to, following pyright's `isWriteAccess`.
    is_write: bool,
}

impl NameContext<'_> {
    fn new() -> Self {
        Self {
            is_keyword_argument: false,
            receiver: None,
            is_write: false,
        }
    }
}

/// A name's evaluated type together with its pyright classification.
type NameType<'db> = Option<(Type<'db>, PyrightType<'db>)>;

/// A token type with its modifiers, or `None` if the name gets no token.
type Classification = Option<(SemanticTokenType, SemanticTokenModifier)>;

bitflags! {
    /// Where the visitor is in the tree, as far as it changes how names are classified.
    #[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
    struct VisitFlags: u8 {
        /// Inside an annotation or another type expression.
        const TYPE_FORM = 1 << 0;
        /// Inside a value expression whose special forms pyright doesn't convert to runtime
        /// objects (see [`PyrightTypeContext::TypeArgument`]).
        const TYPE_ARGUMENT = 1 << 1;
        /// String literals are parsed as annotations. Pyright only parses the strings of
        /// annotations, and not inside `Literal[...]`, `Annotated[...]` or call arguments.
        const PARSE_STRINGS = 1 << 2;
        /// Inside an expression that pyright's `isWriteAccess` counts as written: an assignment,
        /// `for` or `del` target, or a `with` item, except for the receivers of attributes.
        const WRITE = 1 << 3;
        /// Inside a branch of a conditional expression that pyright's binder skips because its
        /// condition is statically known: names have no declarations and no type.
        const SKIPPED_BRANCH = 1 << 4;
        /// Inside an expression that pyright doesn't evaluate, but binds: a branch of a
        /// conditional expression in code that is unreachable after type analysis, or the body of
        /// a lambda in a skipped branch. Names have declarations but no type.
        const UNTYPED = 1 << 5;
        /// In a skipped or untyped expression, the receiver of an attribute, the callee of a call
        /// or the condition of a conditional expression, which pyright evaluates on its own.
        const BASE = 1 << 6;
        /// In a skipped or untyped expression, inside the arguments of a call that pyright
        /// doesn't evaluate, so that nothing in them is evaluated.
        const OPAQUE = 1 << 7;
    }
}

impl VisitFlags {
    /// Whether pyright doesn't evaluate the types of expressions here.
    fn is_unevaluated(self) -> bool {
        self.intersects(Self::SKIPPED_BRANCH | Self::UNTYPED)
    }

    /// The flags for a receiver, callee or condition, which pyright evaluates on its own in an
    /// expression that it doesn't evaluate otherwise.
    fn base(self) -> Self {
        if self.is_unevaluated() && !self.contains(Self::OPAQUE) {
            self | Self::BASE
        } else {
            self - Self::BASE
        }
    }
}

/// Whether one of `ranges` (sorted by start and not overlapping) contains `range`.
fn ranges_contain(ranges: &[TextRange], range: TextRange) -> bool {
    let index = ranges.partition_point(|candidate| candidate.end() <= range.start());
    ranges
        .get(index)
        .is_some_and(|candidate| candidate.contains_range(range))
}

/// How pyright evaluates a branch of a conditional expression.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BranchMode {
    Evaluated,
    /// The binder skips the branch because the condition is statically known.
    Skipped,
    /// The branch is bound, but not evaluated.
    Untyped,
}

/// How pyright and ty treat the code at a location.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Reachability {
    Reachable,
    /// ty finds the code unreachable and has no types for it, but pyright's binder binds it
    /// (for example, the code after a call to a `NoReturn` function).
    Unevaluated,
    /// Pyright's binder doesn't bind the code, so names there have no declarations.
    Unbound,
}

/// Facts that many names of a file share, computed on demand.
#[derive(Default)]
struct Caches<'db> {
    /// The ranges that pyright's binder doesn't bind in other files.
    other_files_unbound: FxHashMap<File, Rc<[TextRange]>>,
    /// The declarations of each symbol, in pyright's order.
    symbol_declarations: FxHashMap<(ScopeId<'db>, ScopedPlaceId), Rc<[PyrightDeclaration<'db>]>>,
    /// The effective types of names in unreachable code, by the first definition of the symbol,
    /// whether the name is bound, and the number of the symbol's definitions that count.
    effective_types: FxHashMap<(Definition<'db>, bool, usize), Option<Type<'db>>>,
    /// The types of attributes and calls in unreachable code.
    unevaluated_types: FxHashMap<TextRange, Option<Type<'db>>>,
    /// The declarations for lists of definitions.
    declarations: FxHashMap<Vec<ResolvedDefinition<'db>>, Rc<[PyrightDeclaration<'db>]>>,
    /// The declarations of members, by receiver and member name.
    attribute_declarations: FxHashMap<(Type<'db>, Name), Rc<[PyrightDeclaration<'db>]>>,
}

/// AST visitor that collects semantic tokens.
struct SemanticTokenVisitor<'db> {
    model: &'db SemanticModel<'db>,
    tokens: Vec<SemanticToken>,
    flags: VisitFlags,
    /// The ranges that ty finds unreachable, sorted by start.
    unreachable: &'db [UnreachableRange],
    /// The ranges that pyright's binder doesn't bind, sorted by start.
    unbound: Rc<[TextRange]>,
    conditions: Rc<StaticConditions>,
    caches: RefCell<Caches<'db>>,
    /// The parameters of the lambdas that enclose the visited node inside a string annotation.
    /// ty has no definitions for the nodes of a string annotation, so the visitor resolves these
    /// names itself.
    string_lambda_parameters: Vec<Name>,
    /// The names that the conditions of enclosing `if` and `while` statements pass to `hasattr()`.
    hasattr_names: Vec<Name>,
    range_filter: Option<TextRange>,
}

impl<'db> SemanticTokenVisitor<'db> {
    fn new(model: &'db SemanticModel<'db>, range_filter: Option<TextRange>) -> Self {
        let db = model.db();
        let file = model.program_file();
        let parsed = parsed_module(db, file.python_file(db)).load(db);
        let suite = parsed.suite();
        let conditions = StaticConditions::new(suite, file.python_version(db));
        let unbound = match source_text(db, file.file(db)).as_notebook() {
            // Pyright analyzes each cell of a notebook on its own, so a cell that stops (with
            // `raise SystemExit`, for example) doesn't make the cells after it unreachable.
            Some(notebook) => notebook
                .cell_offsets()
                .windows(2)
                .flat_map(|cell| {
                    let start = suite.partition_point(|statement| statement.start() < cell[0]);
                    let end = suite.partition_point(|statement| statement.start() < cell[1]);
                    reachability::unbound_ranges(&suite[start..end], &conditions)
                })
                .collect(),
            None => reachability::unbound_ranges(suite, &conditions),
        };
        Self::with_reachability(model, range_filter, unbound.into(), Rc::new(conditions))
    }

    fn with_reachability(
        model: &'db SemanticModel<'db>,
        range_filter: Option<TextRange>,
        unbound: Rc<[TextRange]>,
        conditions: Rc<StaticConditions>,
    ) -> Self {
        Self {
            model,
            tokens: Vec::new(),
            flags: VisitFlags::empty(),
            unreachable: unreachable_ranges(model.db(), model.program_file()),
            unbound,
            conditions,
            caches: RefCell::default(),
            string_lambda_parameters: Vec::new(),
            hasattr_names: Vec::new(),
            range_filter,
        }
    }

    fn db(&self) -> &'db dyn ty_python_semantic::Db {
        self.model.db()
    }

    fn reachability(&self, range: TextRange) -> Reachability {
        if ranges_contain(&self.unbound, range) {
            return Reachability::Unbound;
        }
        let index = self
            .unreachable
            .partition_point(|unreachable| unreachable.range.end() <= range.start());
        if self
            .unreachable
            .get(index)
            .is_some_and(|unreachable| unreachable.range.contains_range(range))
        {
            Reachability::Unevaluated
        } else {
            Reachability::Reachable
        }
    }

    /// Whether `definition` is in this file's code that pyright's binder skips.
    fn is_unbound_definition(&self, definition: Definition<'db>) -> bool {
        let db = self.db();
        let file = definition.file(db);
        let parsed = parsed_module(db, definition.python_file(db)).load(db);
        let range = definition.focus_range(db, &parsed).range();
        if file == self.model.file() {
            return ranges_contain(&self.unbound, range);
        }
        let unbound = self
            .caches
            .borrow_mut()
            .other_files_unbound
            .entry(file)
            .or_insert_with(|| {
                let conditions = StaticConditions::new(
                    parsed.suite(),
                    self.model.program_file().python_version(db),
                );
                reachability::unbound_ranges(parsed.suite(), &conditions).into()
            })
            .clone();
        ranges_contain(&unbound, range)
    }

    fn is_outside_range_filter(&self, range: TextRange) -> bool {
        self.range_filter
            .is_some_and(|range_filter| range.intersect(range_filter).is_none())
    }

    fn add_token(
        &mut self,
        ranged: impl Ranged,
        token_type: SemanticTokenType,
        modifiers: SemanticTokenModifier,
    ) {
        let range = ranged.range();

        if range.is_empty() {
            return;
        }

        // Only emit tokens that intersect with the range filter, if one is specified
        if let Some(range_filter) = self.range_filter {
            // Only include ranges that have a non-empty overlap. Adjacent ranges
            // should be excluded.
            if range
                .intersect(range_filter)
                .is_none_or(TextRange::is_empty)
            {
                return;
            }
        }

        self.tokens.push(SemanticToken {
            range,
            token_type,
            modifiers,
        });
    }

    fn add_classified(&mut self, range: TextRange, classification: Classification) {
        if self.flags.contains(VisitFlags::SKIPPED_BRANCH) {
            return;
        }
        if let Some((token_type, modifiers)) = classification {
            self.add_token(range, token_type, modifiers);
        }
    }

    fn name_type(&self, ty: Option<Type<'db>>, context: PyrightTypeContext) -> NameType<'db> {
        let ty = ty?;
        Some((ty, pyright_type(self.model, ty, context)))
    }

    /// The declarations of `definitions` that pyright sees: its binder skips statically
    /// unreachable code in every file (the `else` of `if TYPE_CHECKING:`, for example).
    fn declarations(
        &self,
        definitions: &[ResolvedDefinition<'db>],
    ) -> Vec<PyrightDeclaration<'db>> {
        let definitions: Vec<ResolvedDefinition<'db>> = definitions
            .iter()
            .filter(|resolved| {
                resolved
                    .definition()
                    .is_none_or(|definition| !self.is_unbound_definition(definition))
            })
            .cloned()
            .collect();
        if let Some(declarations) = self.caches.borrow().declarations.get(&definitions) {
            return declarations.to_vec();
        }
        let declarations: Rc<[PyrightDeclaration<'db>]> =
            pyright_declarations(self.model, &definitions).into();
        self.caches
            .borrow_mut()
            .declarations
            .insert(definitions, declarations.clone());
        declarations.to_vec()
    }

    /// The declarations of the member `name` of `receiver` (see
    /// [`pyright_attribute_declarations`]), without those in code that pyright's binder skips.
    fn attribute_declarations(
        &self,
        receiver: Type<'db>,
        name: &str,
    ) -> Vec<PyrightDeclaration<'db>> {
        let key = (receiver, Name::new(name));
        if let Some(declarations) = self.caches.borrow().attribute_declarations.get(&key) {
            return declarations.to_vec();
        }
        let declarations: Rc<[PyrightDeclaration<'db>]> =
            pyright_attribute_declarations(self.model, receiver, name)
                .into_iter()
                .filter(|declaration| {
                    declaration
                        .definition
                        .is_none_or(|definition| !self.is_unbound_definition(definition))
                })
                .collect();
        self.caches
            .borrow_mut()
            .attribute_declarations
            .insert(key, declarations.clone());
        declarations.to_vec()
    }

    fn name_declarations(&self, name: &str, node: AnyNodeRef<'_>) -> Vec<PyrightDeclaration<'db>> {
        self.declarations(&definitions_for_name(
            self.model,
            name,
            node,
            ImportAliasResolution::ResolveAliases,
        ))
    }

    /// pyright's `_visitNameWithDeclarations`.
    fn classify_name(
        &self,
        context: NameContext<'db>,
        declarations: &[PyrightDeclaration<'db>],
        name_type: NameType<'db>,
    ) -> Classification {
        let mut modifiers = SemanticTokenModifier::empty();
        let Some(primary) = declarations.first() else {
            return self.classify_name_with_type(context, declarations, name_type?);
        };

        let token_type = match primary.kind {
            PyrightDeclarationKind::Variable => name_type
                .and_then(|name_type| {
                    self.variable_token_type(
                        context,
                        name_type,
                        declarations,
                        &mut modifiers,
                        false,
                    )
                })
                .unwrap_or(SemanticTokenType::Variable),
            PyrightDeclarationKind::Parameter => match name_type {
                Some(name_type) => self.parameter_token_type(
                    context,
                    primary.definition,
                    name_type,
                    declarations,
                    &mut modifiers,
                ),
                None => SemanticTokenType::Parameter,
            },
            PyrightDeclarationKind::TypeParameter => SemanticTokenType::TypeParameter,
            PyrightDeclarationKind::TypeAlias => {
                if name_type.is_some_and(|(_, ty)| ty.is_class()) {
                    SemanticTokenType::Class
                } else {
                    SemanticTokenType::Type
                }
            }
            PyrightDeclarationKind::Function => {
                let function = name_type.and_then(|(_, ty)| {
                    matches!(
                        ty.category,
                        PyrightTypeCategory::Function | PyrightTypeCategory::Overloaded
                    )
                    .then_some(ty)
                });
                self.function_token_type(
                    context,
                    Some(primary),
                    declarations,
                    function,
                    &mut modifiers,
                )
            }
            PyrightDeclarationKind::Class | PyrightDeclarationKind::SpecialBuiltInClass => {
                match name_type {
                    // A class that is conditionally redefined (for example, as a function) has a
                    // union type in ty; pyright keeps the class.
                    Some((_, ty)) if ty.is_class() || ty.union_contains_class_object => {
                        self.class_token_type(context, ty, declarations, &mut modifiers, true, true)
                    }
                    _ => SemanticTokenType::Type,
                }
            }
            PyrightDeclarationKind::Alias => {
                return self.classify_name_with_type(context, declarations, name_type?);
            }
        };
        Some((token_type, modifiers))
    }

    /// pyright's `_visitNameWithType`, for names without a usable declaration.
    fn classify_name_with_type(
        &self,
        context: NameContext<'db>,
        declarations: &[PyrightDeclaration<'db>],
        (ty, pyright_ty): (Type<'db>, PyrightType<'db>),
    ) -> Classification {
        let mut modifiers = SemanticTokenModifier::empty();
        match pyright_ty.category {
            PyrightTypeCategory::Any => {
                if pyright_ty.is_special_form {
                    return Some((SemanticTokenType::Class, modifiers));
                }
                if context.is_keyword_argument {
                    return Some((
                        SemanticTokenType::Parameter,
                        SemanticTokenModifier::PARAMETER,
                    ));
                }
                return None;
            }
            PyrightTypeCategory::Unknown => return None,
            PyrightTypeCategory::Function | PyrightTypeCategory::Overloaded => {
                let declaration = pyright_ty
                    .function
                    .and_then(|function| function.declaration)
                    .map(|definition| pyright_declaration(self.model, definition));
                let token_type = self.function_token_type(
                    context,
                    declaration.as_ref(),
                    declarations,
                    Some(pyright_ty),
                    &mut modifiers,
                );
                return Some((token_type, modifiers));
            }
            PyrightTypeCategory::Module => return Some((SemanticTokenType::Namespace, modifiers)),
            PyrightTypeCategory::Union if !pyright_ty.is_instance => {
                return Some((SemanticTokenType::Class, modifiers));
            }
            PyrightTypeCategory::Class if !pyright_ty.is_instance => {
                let token_type = self.class_token_type(
                    context,
                    pyright_ty,
                    declarations,
                    &mut modifiers,
                    true,
                    true,
                );
                return Some((token_type, modifiers));
            }
            PyrightTypeCategory::Never
            | PyrightTypeCategory::Union
            | PyrightTypeCategory::Class
            | PyrightTypeCategory::TypeVar => {}
        }

        let token_type = self.variable_token_type(
            context,
            (ty, pyright_ty),
            declarations,
            &mut modifiers,
            false,
        )?;
        Some((token_type, modifiers))
    }

    /// pyright's `_getVariableTokenType`, used for variables and (with `is_parameter`) parameters.
    fn variable_token_type(
        &self,
        context: NameContext<'db>,
        (_, ty): (Type<'db>, PyrightType<'db>),
        declarations: &[PyrightDeclaration<'db>],
        modifiers: &mut SemanticTokenModifier,
        is_parameter: bool,
    ) -> Option<SemanticTokenType> {
        // Names whose type is `Any` or unknown aren't classified, unless they refer to the special
        // form `Any` itself.
        if !ty.is_special_form && ty.is_any_or_unknown {
            return None;
        }

        let mut is_keyword_argument = false;
        if !is_parameter {
            if context.is_keyword_argument {
                is_keyword_argument = true;
                *modifiers |= SemanticTokenModifier::PARAMETER;
            }
            if declarations
                .iter()
                .any(|declaration| declaration.in_project_builtins_module)
            {
                *modifiers |= SemanticTokenModifier::BUILTIN;
            }
            if declarations
                .iter()
                .any(|declaration| declaration.is_variable() && declaration.in_enum_class_body)
            {
                return Some(SemanticTokenType::EnumMember);
            }
        }

        let is_readonly = declarations
            .iter()
            .any(|declaration| declaration.is_variable() && declaration.is_constant_or_final);
        if is_readonly {
            *modifiers |= SemanticTokenModifier::READONLY;
        }

        let is_class_member = !is_parameter
            && self.apply_class_member_access(context, declarations, modifiers, is_readonly);

        if !is_parameter
            && ty.category == PyrightTypeCategory::TypeVar
            && !ty.is_synthesized_type_var
            && ty.is_instantiable
        {
            return Some(SemanticTokenType::TypeParameter);
        }

        // Variables that hold a type form: classes get `class`, everything else gets `type`.
        if ty.is_instantiable
            && matches!(
                ty.category,
                PyrightTypeCategory::Class
                    | PyrightTypeCategory::Union
                    | PyrightTypeCategory::Function
                    | PyrightTypeCategory::Overloaded
            )
        {
            if ty.is_class() {
                return Some(self.class_token_type(
                    context,
                    ty,
                    declarations,
                    modifiers,
                    true,
                    false,
                ));
            }
            return Some(SemanticTokenType::Type);
        }

        if ty.category == PyrightTypeCategory::Any && ty.is_special_form {
            return Some(SemanticTokenType::Type);
        }

        if matches!(
            ty.category,
            PyrightTypeCategory::Function | PyrightTypeCategory::Overloaded
        ) {
            return Some(self.function_token_type(
                context,
                declarations.first(),
                declarations,
                Some(ty),
                modifiers,
            ));
        }

        if ty.category == PyrightTypeCategory::Union && ty.all_functions {
            return Some(SemanticTokenType::Function);
        }

        if !is_parameter && ty.category == PyrightTypeCategory::Module {
            return Some(SemanticTokenType::Namespace);
        }

        // pyright's `_getNeverTokenType`: a type alias to `Never` or `NoReturn` is a type.
        if ty.category == PyrightTypeCategory::Never
            && !declarations.is_empty()
            && declarations.iter().all(|declaration| {
                declaration.definition.is_some_and(|definition| {
                    pyright_is_type_alias_declaration(self.db(), definition)
                })
            })
        {
            return Some(SemanticTokenType::Type);
        }

        if is_class_member {
            return Some(SemanticTokenType::Property);
        }
        if is_parameter || is_keyword_argument {
            return Some(SemanticTokenType::Parameter);
        }
        Some(SemanticTokenType::Variable)
    }

    /// pyright's `_applyClassMemberAccessModifiers`: adds `classMember` (and `static` and
    /// `readonly` where they apply) for names that are class or instance members. Returns whether
    /// the name is a class member.
    fn apply_class_member_access(
        &self,
        context: NameContext<'db>,
        declarations: &[PyrightDeclaration<'db>],
        modifiers: &mut SemanticTokenModifier,
        already_readonly: bool,
    ) -> bool {
        let info = if let Some(declaration) = declarations
            .iter()
            .find(|declaration| declaration.in_class_body)
        {
            // Declared in a class body.
            Some((
                declarations
                    .iter()
                    .all(|declaration| declaration.is_property_without_setter),
                declaration.enclosing_class_has_member,
            ))
        } else if let Some(receiver) = context.receiver
            && let Some(receiver) = pyright_receiver(self.model, receiver)
        {
            // An attribute access `x.name` on a class or instance.
            if declarations.is_empty() {
                Some((receiver.has_magic_get && !receiver.has_magic_set, false))
            } else {
                Some((false, receiver.is_class_object))
            }
        } else {
            None
        };

        let Some((is_readonly, is_static)) = info else {
            return false;
        };
        if !already_readonly && is_readonly {
            *modifiers |= SemanticTokenModifier::READONLY;
        }
        *modifiers |= SemanticTokenModifier::CLASS_MEMBER;
        if is_static {
            *modifiers |= SemanticTokenModifier::STATIC;
        }
        true
    }

    /// pyright's `_getClassTokenType`.
    fn class_token_type(
        &self,
        context: NameContext<'db>,
        ty: PyrightType<'db>,
        declarations: &[PyrightDeclaration<'db>],
        modifiers: &mut SemanticTokenModifier,
        check_builtin: bool,
        apply_class_member_access: bool,
    ) -> SemanticTokenType {
        if check_builtin
            && declarations
                .iter()
                .any(|declaration| declaration.in_builtins_module)
        {
            *modifiers |= SemanticTokenModifier::DEFAULT_LIBRARY | SemanticTokenModifier::BUILTIN;
        }
        if context.is_keyword_argument {
            *modifiers |= SemanticTokenModifier::PARAMETER;
        }
        if apply_class_member_access {
            self.apply_class_member_access(context, declarations, modifiers, false);
        }
        if ty.is_enum_class {
            SemanticTokenType::Enum
        } else {
            SemanticTokenType::Class
        }
    }

    /// pyright's `_getFunctionTokenType`. `declaration` is the function's own declaration, if
    /// known; `function` is the name's type, if it is a function.
    fn function_token_type(
        &self,
        context: NameContext<'db>,
        declaration: Option<&PyrightDeclaration<'db>>,
        declarations: &[PyrightDeclaration<'db>],
        function: Option<PyrightType<'db>>,
        modifiers: &mut SemanticTokenModifier,
    ) -> SemanticTokenType {
        let db = self.db();

        // An instantiable function type is a `Callable` type form.
        if function.is_some_and(|function| function.is_instantiable) {
            return SemanticTokenType::Class;
        }

        if context.is_keyword_argument {
            *modifiers |= SemanticTokenModifier::PARAMETER;
        }

        let function_facts: Option<PyrightFunction<'db>> =
            function.and_then(|function| function.function);
        if function_facts.is_some_and(|facts| facts.in_builtins_module) {
            *modifiers |= SemanticTokenModifier::DEFAULT_LIBRARY | SemanticTokenModifier::BUILTIN;
        }

        if let Some(declaration) = declaration
            && declaration.kind == PyrightDeclarationKind::Function
            && let Some(definition) = declaration.definition
        {
            // Pyright reads the flag from the name's function type when it has one, and from the
            // declaration otherwise. ty loses the flag when a static method is accessed through
            // its class, so consult the declaration as well.
            let is_static = function_facts.is_some_and(|facts| facts.is_static_method)
                || pyright_function_definition_is_static(db, definition);
            if is_static {
                *modifiers |= SemanticTokenModifier::STATIC;
            }

            if declaration.in_class_body {
                *modifiers |= SemanticTokenModifier::CLASS_MEMBER;
                return match pyright_method_accessor(
                    self.model,
                    definition,
                    declarations,
                    context.is_write,
                ) {
                    PyrightAccessor::Method => SemanticTokenType::Method,
                    PyrightAccessor::Accessor {
                        is_readonly,
                        effective_type,
                    } => {
                        if is_readonly {
                            *modifiers |= SemanticTokenModifier::READONLY;
                        }
                        // Reading an attribute through a property gives the getter's return type,
                        // which pyright takes `defaultLibrary` and `builtin` from. ty has no type
                        // for the attribute if the getter isn't annotated.
                        if function_facts.is_none()
                            && context.receiver.is_some()
                            && !context.is_write
                            && let Some(effective_type) = effective_type
                            && pyright_type(self.model, effective_type, PyrightTypeContext::Value)
                                .function
                                .is_some_and(|facts| facts.in_builtins_module)
                        {
                            *modifiers |= SemanticTokenModifier::DEFAULT_LIBRARY
                                | SemanticTokenModifier::BUILTIN;
                        }
                        self.accessor_token_type(effective_type)
                    }
                };
            }
        }

        if declaration.is_none()
            && let Some(receiver) = context.receiver
        {
            let receiver_is_class =
                pyright_type(self.model, receiver, PyrightTypeContext::Value).is_class();
            if receiver_is_class {
                *modifiers |= SemanticTokenModifier::CLASS_MEMBER;
            }
            if let Some(receiver) = pyright_receiver(self.model, receiver)
                && receiver.has_magic_get
            {
                if !receiver.has_magic_set {
                    *modifiers |= SemanticTokenModifier::READONLY;
                }
                return if function_facts.is_some_and(|facts| facts.has_method_class) {
                    SemanticTokenType::Method
                } else {
                    SemanticTokenType::Function
                };
            }
            if receiver_is_class {
                return SemanticTokenType::Method;
            }
        }

        SemanticTokenType::Function
    }

    /// The token for a property or descriptor, from the type that reading (or writing) it
    /// produces.
    fn accessor_token_type(&self, effective_type: Option<Type<'db>>) -> SemanticTokenType {
        let Some(effective_type) = effective_type else {
            return SemanticTokenType::Property;
        };
        let ty = pyright_type(self.model, effective_type, PyrightTypeContext::Value);
        if matches!(
            ty.category,
            PyrightTypeCategory::Function | PyrightTypeCategory::Overloaded
        ) || (ty.category == PyrightTypeCategory::Union && ty.all_functions)
        {
            return if ty.all_method_types {
                SemanticTokenType::Method
            } else {
                SemanticTokenType::Function
            };
        }
        if ty.is_any_or_unknown && !ty.is_special_form {
            return SemanticTokenType::Property;
        }
        if ty.is_instantiable {
            let is_class = if ty.category == PyrightTypeCategory::TypeVar {
                ty.type_var_bound_is_class
            } else {
                ty.is_class()
            };
            return if is_class {
                SemanticTokenType::Class
            } else {
                SemanticTokenType::Type
            };
        }
        SemanticTokenType::Property
    }

    /// pyright's `_getParamTokenType`. `parameter` is the parameter's own definition.
    fn parameter_token_type(
        &self,
        context: NameContext<'db>,
        parameter: Option<Definition<'db>>,
        name_type: (Type<'db>, PyrightType<'db>),
        declarations: &[PyrightDeclaration<'db>],
        modifiers: &mut SemanticTokenModifier,
    ) -> SemanticTokenType {
        *modifiers |= SemanticTokenModifier::PARAMETER;
        if parameter
            .is_some_and(|parameter| pyright_parameter_is_method_receiver(self.db(), parameter))
        {
            return if name_type.1.is_instantiable {
                SemanticTokenType::ClsParameter
            } else {
                SemanticTokenType::SelfParameter
            };
        }
        self.variable_token_type(context, name_type, declarations, modifiers, true)
            .unwrap_or(SemanticTokenType::Parameter)
    }

    fn type_context_for(&self, ctx: ExprContext) -> PyrightTypeContext {
        if self.flags.contains(VisitFlags::TYPE_FORM) {
            PyrightTypeContext::TypeExpression
        } else if self.flags.contains(VisitFlags::TYPE_ARGUMENT) {
            PyrightTypeContext::TypeArgument
        } else if ctx.is_store() {
            PyrightTypeContext::StoreTarget
        } else {
            PyrightTypeContext::Value
        }
    }

    fn is_write(&self, ctx: ExprContext) -> bool {
        self.flags.contains(VisitFlags::WRITE) || ctx.is_store() || ctx.is_del()
    }

    fn visit_name_expr(&mut self, name: &ast::ExprName) {
        if name.id.is_empty() {
            return;
        }
        if self.string_lambda_parameters.contains(&name.id) {
            self.add_classified(
                name.range(),
                Some((
                    SemanticTokenType::Parameter,
                    SemanticTokenModifier::PARAMETER,
                )),
            );
            return;
        }
        if self.flags.contains(VisitFlags::SKIPPED_BRANCH) {
            if self.flags.contains(VisitFlags::BASE) {
                self.add_skipped_base(name.range(), NameContext::new(), ExprRef::Name(name));
            }
            return;
        }
        if self.flags.contains(VisitFlags::UNTYPED) {
            let declarations = self.name_declarations(name.id.as_str(), name.into());
            // Pyright looks builtins up without code flow, so they keep their type.
            let name_type = if self.flags.contains(VisitFlags::BASE)
                || (!declarations.is_empty()
                    && declarations
                        .iter()
                        .all(|declaration| declaration.in_builtins_module))
            {
                self.name_type(
                    self.effective_name_type(name, true),
                    PyrightTypeContext::Value,
                )
            } else {
                None
            };
            let classification = self.classify_name(NameContext::new(), &declarations, name_type);
            self.add_classified(name.range(), classification);
            return;
        }
        let context = NameContext {
            is_write: self.is_write(name.ctx),
            ..NameContext::new()
        };
        let type_context = self.type_context_for(name.ctx);
        let classification = match self.reachability(name.range()) {
            Reachability::Unbound => {
                let name_type = self.name_type(self.effective_name_type(name, false), type_context);
                name_type.and_then(|name_type| {
                    self.classify_name_with_type(NameContext::new(), &[], name_type)
                })
            }
            Reachability::Unevaluated => {
                let declarations = self.name_declarations(name.id.as_str(), name.into());
                let name_type = self.name_type(self.effective_name_type(name, true), type_context);
                self.classify_name(context, &declarations, name_type)
            }
            Reachability::Reachable => {
                let declarations = self.name_declarations(name.id.as_str(), name.into());
                // Pyright declares `__spec__`, `__loader__` and `__builtins__` as `Any`, and
                // `__debug__` is a keyword constant for it.
                if matches!(
                    name.id.as_str(),
                    "__spec__" | "__loader__" | "__builtins__" | "__debug__"
                ) && declarations.iter().all(|declaration| {
                    declaration
                        .definition
                        .is_none_or(|definition| definition.file(self.db()) != self.model.file())
                }) {
                    return;
                }
                let name_type =
                    self.name_type(self.reachable_name_type(name, &declarations), type_context);
                self.classify_name(context, &declarations, name_type)
                    .or_else(|| self.unresolved_module_import(name, name_type))
            }
        };
        self.add_classified(name.range(), classification);
    }

    /// The type of a name in reachable code.
    fn reachable_name_type(
        &self,
        name: &ast::ExprName,
        declarations: &[PyrightDeclaration<'db>],
    ) -> Option<Type<'db>> {
        let db = self.db();
        // In type expressions ty records the type of the annotated value (an instance of `int`
        // for `int`), while pyright evaluates the name itself, so use the value bound to the name.
        // Names that ty records no type for (in match patterns, for example) also fall back to the
        // bound value.
        let bound_type = || {
            declarations
                .first()
                .and_then(|declaration| declaration.definition)
                .and_then(|definition| pyright_definition_type(db, definition))
        };
        if self
            .flags
            .intersects(VisitFlags::TYPE_FORM | VisitFlags::TYPE_ARGUMENT)
        {
            match (bound_type(), name.inferred_type(self.model)) {
                // A bare `Callable` in a type expression means `Callable[..., Unknown]`, an
                // instantiable function type for pyright; only the `Callable` of `Callable[...]`
                // is the special form itself. ty records that callable type on the bare name.
                (
                    Some(Type::SpecialForm(SpecialFormType::TypingCallable)),
                    Some(callable @ Type::Callable(_)),
                ) if self.flags.contains(VisitFlags::TYPE_FORM) => Some(callable),
                (bound, inferred) => bound.or(inferred),
            }
            .map(|ty| {
                if self.flags.contains(VisitFlags::TYPE_FORM)
                    && !self.is_valid_in_type_form(declarations, Some(ty))
                {
                    Type::unknown()
                } else {
                    ty
                }
            })
        } else {
            let declared = self.declared_variable_type(declarations);
            match name.inferred_type(self.model) {
                // ty narrows a name that an enclosing `hasattr()` check rules out to `Never`
                // (`if not hasattr(self, "x")` where the class assigns `self.x`), while pyright's
                // `hasattr()` doesn't narrow.
                Some(Type::Never) if self.hasattr_names.contains(&name.id) => bound_type(),
                // Pyright doesn't narrow a variable declared as `Any` on assignment, and keeps
                // the declared type of other variables when the assigned value is `Any`.
                _ if name.ctx.is_store() && declared.is_some_and(pyright_is_explicit_any) => {
                    declared
                }
                Some(inferred) if pyright_is_explicit_any(inferred) && declared.is_some() => {
                    declared
                }
                inferred => inferred.or_else(bound_type),
            }
        }
    }

    /// Pyright types the name bound by an unresolved `import module` as a module.
    fn unresolved_module_import(
        &self,
        name: &ast::ExprName,
        name_type: NameType<'db>,
    ) -> Classification {
        if name_type.is_some_and(|(_, ty)| ty.category != PyrightTypeCategory::Unknown) {
            return None;
        }
        let definitions = definitions_for_name(
            self.model,
            name.id.as_str(),
            name.into(),
            ImportAliasResolution::PreserveAliases,
        );
        let definition = definitions.first()?.definition()?;
        matches!(definition.kind(self.db()), DefinitionKind::Import(_))
            .then_some((SemanticTokenType::Namespace, SemanticTokenModifier::empty()))
    }

    /// The declarations of the symbol that `definition` binds (see [`pyright_symbol_declarations`]).
    fn symbol_declarations(&self, definition: Definition<'db>) -> Rc<[PyrightDeclaration<'db>]> {
        let db = self.db();
        let key = (definition.scope(db), definition.place(db));
        if let Some(declarations) = self.caches.borrow().symbol_declarations.get(&key) {
            return declarations.clone();
        }
        let declarations: Rc<[PyrightDeclaration<'db>]> =
            pyright_symbol_declarations(self.model, definition).into();
        self.caches
            .borrow_mut()
            .symbol_declarations
            .insert(key, declarations.clone());
        declarations
    }

    /// The effective type of a symbol with `declarations`, without code flow: the declared type of
    /// the last declaration that has one, or the union of the inferred types of all of them.
    fn declarations_effective_type(
        &self,
        declarations: &[PyrightDeclaration<'db>],
    ) -> Option<Type<'db>> {
        let db = self.db();
        if let Some(definition) = declarations
            .iter()
            .filter(|declaration| declaration.has_declared_type)
            .filter_map(|declaration| declaration.definition)
            .next_back()
        {
            return pyright_definition_type(db, definition);
        }
        let elements: Vec<Type<'db>> = declarations
            .iter()
            .filter_map(|declaration| declaration.definition)
            .filter_map(|definition| self.inferred_definition_type(definition))
            .collect();
        (!elements.is_empty()).then(|| pyright_union(self.model, elements))
    }

    /// The declared type of an annotated variable or parameter among `declarations`.
    fn declared_variable_type(
        &self,
        declarations: &[PyrightDeclaration<'db>],
    ) -> Option<Type<'db>> {
        let db = self.db();
        declarations
            .iter()
            .filter(|declaration| {
                declaration.has_declared_type
                    && matches!(
                        declaration.kind,
                        PyrightDeclarationKind::Variable | PyrightDeclarationKind::Parameter
                    )
            })
            .filter_map(|declaration| declaration.definition)
            .next_back()
            .and_then(|definition| pyright_declared_type(db, definition))
    }

    /// Whether pyright accepts a name with `declarations` and type `ty` in a type expression.
    /// Variables and parameters are only valid there if they are type aliases, type variables, or
    /// classes created by calling `NewType`, `NamedTuple` and the like; pyright evaluates others
    /// to an unknown type.
    fn is_valid_in_type_form(
        &self,
        declarations: &[PyrightDeclaration<'db>],
        ty: Option<Type<'db>>,
    ) -> bool {
        let db = self.db();
        let is_variable = |declaration: &PyrightDeclaration<'db>| {
            matches!(
                declaration.kind,
                PyrightDeclarationKind::Variable | PyrightDeclarationKind::Parameter
            ) && declaration
                .definition
                .is_some_and(|definition| !pyright_is_in_typing_stub(db, definition))
        };
        // ty resolves an import through every branch of the imported module, while pyright only
        // sees the branches that its binder doesn't skip (`if TYPE_CHECKING:` ... `else:` ...). A
        // class or special form among the declarations comes from such a branch.
        if declarations.is_empty() || !declarations.iter().all(is_variable) {
            return true;
        }
        if declarations.iter().all(|declaration| {
            declaration
                .definition
                .is_some_and(|definition| self.is_type_alias_definition(definition))
        }) {
            return true;
        }
        ty.is_some_and(pyright_is_type_form_variable_type)
    }

    /// The execution scope of a scope, which is how pyright scopes code flow: class bodies and
    /// comprehensions belong to the function or module that contains them.
    fn execution_scope(&self, scope: FileScopeId) -> FileScopeId {
        semantic_index(self.db(), self.model.program_file())
            .ancestor_scopes(scope)
            .find(|(_, scope)| {
                matches!(
                    scope.kind(),
                    ScopeKind::Module | ScopeKind::Function | ScopeKind::Lambda
                )
            })
            .map_or(scope, |(id, _)| id)
    }

    /// Pyright's effective type of the symbol that `name` refers to, for code where ty has no
    /// type for the name (`getEffectiveTypeOfSymbolForUsage` without code-flow analysis).
    ///
    /// If the symbol has declarations with a declared type (functions, classes and annotated
    /// variables), it's the last one's declared type. In code that pyright's binder doesn't bind
    /// (`is_bound` is false), declarations in the same execution scope can't reach the name, so
    /// they only count if there's a single one. Otherwise the type is the union of the types of
    /// the symbol's assignments, except those of the same execution scope that come after the
    /// name.
    fn effective_name_type(&self, name: &ast::ExprName, is_bound: bool) -> Option<Type<'db>> {
        let db = self.db();
        let file = self.model.file();
        // ty's lookup follows code flow, which doesn't reach this name.
        let first = pyright_symbol_definition(self.model, name).or_else(|| {
            definitions_for_name(
                self.model,
                name.id.as_str(),
                name.into(),
                ImportAliasResolution::PreserveAliases,
            )
            .iter()
            .find_map(ResolvedDefinition::definition)
        })?;
        let use_scope = self
            .model
            .scope(name.into())
            .map(|scope| self.execution_scope(scope));
        let is_same_scope = |definition: Definition<'db>| {
            definition.file(db) == file
                && Some(self.execution_scope(definition.file_scope(db))) == use_scope
        };
        let parsed = parsed_module(db, first.python_file(db)).load(db);
        let definitions: Vec<Definition<'db>> = self
            .symbol_declarations(first)
            .iter()
            .filter_map(|declaration| declaration.definition)
            // Pyright doesn't bind the declarations in code that its binder skips.
            .filter(|definition| {
                definition.file(db) != file
                    || self.reachability(definition.focus_range(db, &parsed).range())
                        != Reachability::Unbound
            })
            .collect();

        let typed: Vec<Definition<'db>> = definitions
            .iter()
            .copied()
            .filter(|definition| pyright_has_declared_type(db, *definition))
            .collect();
        if !typed.is_empty() {
            let last = if is_bound || typed.len() == 1 {
                typed.last()
            } else {
                typed
                    .iter()
                    .rfind(|definition| !is_same_scope(**definition))
            };
            return last.and_then(|definition| pyright_definition_type(db, *definition));
        }

        let counted: Vec<Definition<'db>> = definitions
            .iter()
            .copied()
            .filter(|definition| {
                let is_import = matches!(
                    definition.kind(db),
                    DefinitionKind::Import(_)
                        | DefinitionKind::ImportFrom(_)
                        | DefinitionKind::ImportFromSubmodule(_)
                        | DefinitionKind::StarImport(_)
                );
                is_import
                    || !is_same_scope(*definition)
                    || definition.focus_range(db, &parsed).start() < name.start()
            })
            .collect();
        // Names in unreachable code are often used many times after the same definitions.
        let key = (first, is_bound, counted.len());
        if let Some(ty) = self.caches.borrow().effective_types.get(&key) {
            return *ty;
        }
        let elements: Vec<Type<'db>> = counted
            .into_iter()
            .filter_map(|definition| self.inferred_definition_type(definition))
            .map(|ty| pyright_widen_literal(self.model, ty))
            .collect();
        let ty = (!elements.is_empty()).then(|| pyright_union(self.model, elements));
        self.caches.borrow_mut().effective_types.insert(key, ty);
        ty
    }

    /// The type pyright infers for a definition without a declared type. An unannotated
    /// parameter's type comes from its default value, except that containers and functions give
    /// an unknown type.
    fn inferred_definition_type(&self, definition: Definition<'db>) -> Option<Type<'db>> {
        let db = self.db();
        if let DefinitionKind::Parameter(ParameterDefinitionNodeKind::Parameter(parameter)) =
            definition.kind(db)
            && definition.file(db) == self.model.file()
            && !pyright_parameter_is_method_receiver(db, definition)
        {
            let parsed = parsed_module(db, definition.python_file(db)).load(db);
            return Some(match parameter.node(&parsed).default.as_deref() {
                None
                | Some(
                    Expr::Lambda(_)
                    | Expr::Tuple(_)
                    | Expr::List(_)
                    | Expr::Set(_)
                    | Expr::Dict(_)
                    | Expr::ListComp(_)
                    | Expr::SetComp(_)
                    | Expr::DictComp(_),
                ) => Type::unknown(),
                Some(default) => match default.inferred_type(self.model) {
                    Some(Type::FunctionLiteral(_)) | None => Type::unknown(),
                    Some(ty) => ty,
                },
            });
        }
        pyright_definition_type(db, definition)
    }

    /// The type of an expression in code where ty has no types: pyright's type without code-flow
    /// narrowing.
    fn unevaluated_expression_type(&self, expr: &Expr) -> Option<Type<'db>> {
        if matches!(expr, Expr::Attribute(_) | Expr::Call(_)) {
            if let Some(ty) = self.caches.borrow().unevaluated_types.get(&expr.range()) {
                return *ty;
            }
            let ty = self.uncached_unevaluated_expression_type(expr);
            self.caches
                .borrow_mut()
                .unevaluated_types
                .insert(expr.range(), ty);
            return ty;
        }
        self.uncached_unevaluated_expression_type(expr)
    }

    fn uncached_unevaluated_expression_type(&self, expr: &Expr) -> Option<Type<'db>> {
        match expr {
            Expr::Name(name) => {
                let is_bound = self.reachability(name.range()) != Reachability::Unbound;
                self.effective_name_type(name, is_bound)
            }
            Expr::Attribute(attribute) => {
                let receiver = self.unevaluated_expression_type(&attribute.value)?;
                pyright_member_type(self.model, receiver, attribute.attr.as_str())
            }
            Expr::Call(call) => {
                let callee = self.unevaluated_expression_type(&call.func)?;
                pyright_call_return_type(self.model, callee)
            }
            _ => None,
        }
    }

    /// Adds the token for the receiver of an attribute, or the callee of a call, in a branch of
    /// a conditional expression that pyright doesn't evaluate. Pyright still evaluates these, to
    /// `Never` for names that the execution scope binds, and to their type otherwise.
    fn add_skipped_base(&mut self, range: TextRange, context: NameContext<'db>, expr: ExprRef<'_>) {
        let classification = self
            .name_type(self.skipped_base_type(expr), PyrightTypeContext::Value)
            .and_then(|name_type| self.classify_name_with_type(context, &[], name_type));
        if let Some((token_type, modifiers)) = classification {
            self.add_token(range, token_type, modifiers);
        }
    }

    fn skipped_base_type(&self, expr: ExprRef<'_>) -> Option<Type<'db>> {
        match expr {
            ExprRef::Name(name) => {
                let db = self.db();
                let index = semantic_index(db, self.model.program_file());
                let use_scope = self.model.scope(name.into())?;
                let execution_scope = self.execution_scope(use_scope);
                let is_bound_here = index
                    .ancestor_scopes(use_scope)
                    .take_while(|(id, _)| *id != execution_scope)
                    .map(|(id, _)| id)
                    .chain([execution_scope])
                    .any(|scope| {
                        let table = index.place_table(scope);
                        table
                            .symbol_id(name.id.as_str())
                            .is_some_and(|symbol| table.symbol(symbol).is_bound())
                    });
                if is_bound_here {
                    Some(Type::Never)
                } else {
                    self.effective_name_type(name, true)
                }
            }
            ExprRef::Attribute(attribute) => {
                match self.skipped_base_type((&*attribute.value).into())? {
                    Type::Never => Some(Type::Never),
                    receiver => pyright_member_type(self.model, receiver, attribute.attr.as_str()),
                }
            }
            ExprRef::Call(call) => match self.skipped_base_type((&*call.func).into())? {
                Type::Never => Some(Type::Never),
                callee => pyright_call_return_type(self.model, callee),
            },
            _ => None,
        }
    }

    fn visit_attribute_expr(&mut self, attribute: &ast::ExprAttribute, has_type: bool) {
        let is_write = self.is_write(attribute.ctx);
        // The receiver of an attribute is read, not written, and pyright evaluates it even in an
        // expression that it doesn't evaluate otherwise.
        self.visit_expr_with_flags(&attribute.value, (self.flags - VisitFlags::WRITE).base());

        if attribute.attr.is_empty() {
            return;
        }
        if self.flags.contains(VisitFlags::SKIPPED_BRANCH) {
            if self.flags.contains(VisitFlags::BASE) {
                let receiver = self.skipped_base_type((&*attribute.value).into());
                let context = NameContext {
                    receiver,
                    ..NameContext::new()
                };
                self.add_skipped_base(
                    attribute.attr.range(),
                    context,
                    ExprRef::Attribute(attribute),
                );
            }
            return;
        }
        if self.flags.contains(VisitFlags::UNTYPED) {
            // The member has declarations if pyright evaluates the receiver, but a type only if
            // it evaluates the attribute itself.
            let receiver = if self.flags.contains(VisitFlags::OPAQUE) {
                None
            } else {
                self.unevaluated_expression_type(&attribute.value)
            };
            let declarations = receiver
                .filter(|receiver| {
                    !pyright_type(self.model, *receiver, PyrightTypeContext::Value)
                        .is_any_or_unknown
                })
                .map(|receiver| self.attribute_declarations(receiver, attribute.attr.as_str()))
                .unwrap_or_default();
            let name_type = if self.flags.contains(VisitFlags::BASE) {
                self.name_type(
                    receiver.and_then(|receiver| {
                        pyright_member_type(self.model, receiver, attribute.attr.as_str())
                    }),
                    PyrightTypeContext::Value,
                )
            } else {
                None
            };
            let context = NameContext {
                receiver,
                is_write,
                ..NameContext::new()
            };
            let classification = self.classify_name(context, &declarations, name_type);
            self.add_classified(attribute.attr.range(), classification);
            return;
        }

        // `P.args` and `P.kwargs` of a parameter specification in an annotation.
        if self.flags.contains(VisitFlags::TYPE_FORM)
            && matches!(attribute.attr.as_str(), "args" | "kwargs")
            && let Expr::Name(receiver) = attribute.value.as_ref()
            && let Some(definition) = self
                .name_declarations(receiver.id.as_str(), receiver.into())
                .first()
                .and_then(|declaration| declaration.definition)
            && let Some(receiver_type) = pyright_definition_type(self.db(), definition)
            && pyright_type(
                self.model,
                receiver_type,
                PyrightTypeContext::TypeExpression,
            )
            .category
                == PyrightTypeCategory::TypeVar
        {
            self.add_token(
                attribute.attr.range(),
                SemanticTokenType::TypeParameter,
                SemanticTokenModifier::empty(),
            );
            return;
        }

        let db = self.db();
        let name = attribute.attr.as_str();
        let reachability = self.reachability(attribute.range());
        let receiver = match reachability {
            Reachability::Reachable => self.receiver_type(&attribute.value),
            Reachability::Unevaluated | Reachability::Unbound => {
                self.unevaluated_expression_type(&attribute.value)
            }
        };

        // Pyright only looks up member declarations on classes, instances, modules and unions of
        // those; attributes of functions and of `Any` have none. Code that pyright's binder
        // skips has no declarations at all.
        let receiver_type =
            receiver.map(|receiver| pyright_type(self.model, receiver, PyrightTypeContext::Value));
        let has_member_declarations = reachability != Reachability::Unbound
            && receiver_type.is_some_and(|receiver_type| {
                !matches!(
                    receiver_type.category,
                    PyrightTypeCategory::Function
                        | PyrightTypeCategory::Overloaded
                        | PyrightTypeCategory::Any
                        | PyrightTypeCategory::Unknown
                )
            });
        let declarations = match receiver {
            Some(receiver) if has_member_declarations => {
                self.attribute_declarations(receiver, name)
            }
            _ => Vec::new(),
        };
        let is_declared = declarations
            .iter()
            .any(|declaration| declaration.has_declared_type);

        let mut ty = if !has_type {
            None
        } else if reachability == Reachability::Reachable {
            self.reachable_attribute_type(attribute, receiver, &declarations, is_declared)
        } else {
            receiver.and_then(|receiver| pyright_member_type(self.model, receiver, name))
        };
        // Pyright infers the type of a `__slots__` entry from its string when nothing assigns it.
        if has_type
            && ty.is_none_or(|ty| ty.is_unknown())
            && declarations.iter().any(PyrightDeclaration::is_slot)
        {
            ty = Some(pyright_slot_type(self.model));
        }
        let mut name_type = self.name_type(ty, self.type_context_for(attribute.ctx));

        // Every member of an `Any` or unknown receiver has the receiver's type in pyright, while
        // ty knows a few (`Unknown.__class__` is `type[Unknown]`). Both narrow the member to the
        // type assigned to it, though.
        if has_type
            && let Some(receiver) = receiver
            && let Some(receiver_type) = receiver_type
            && receiver_type.is_any_or_unknown
            && !receiver_type.is_special_form
            && ty.is_none_or(|ty| {
                matches!(ty, Type::Dynamic(_)) || pyright_is_dynamic_class_object(ty)
            })
        {
            name_type = Some((receiver, receiver_type));
        }
        // Inside the class, pyright types attributes assigned from the parameters of a
        // pseudo-generic class with a synthesized type variable.
        let through_self = receiver_type.is_some_and(|receiver| receiver.is_synthesized_type_var);
        // Pyright's `is not None` narrowing turns the synthesized type variable into an unknown
        // type (`if self.fs is not None: self.fs...`), while truthiness narrowing keeps it.
        if has_type
            && through_self
            && name_type.is_none_or(|(_, ty)| ty.is_any_or_unknown)
            && !ty.is_some_and(|ty| pyright_is_narrowed_not_none(db, ty))
            && pyright_is_pseudo_generic_attribute(self.model, &declarations)
        {
            name_type = Some((
                ty.unwrap_or_else(Type::unknown),
                PyrightType::synthesized_type_var_instance(),
            ));
        }
        let context = NameContext {
            receiver,
            is_write,
            ..NameContext::new()
        };
        let classification = if reachability == Reachability::Unbound {
            name_type.and_then(|name_type| self.classify_name_with_type(context, &[], name_type))
        } else {
            self.classify_name(context, &declarations, name_type)
        };
        self.add_classified(attribute.attr.range(), classification);
    }

    /// The type of the receiver of an attribute in reachable code.
    ///
    /// Pyright infers the return type of functions without a return annotation, while ty's type
    /// for calling them is unknown. That makes a difference for the members of the result, such
    /// as `complete` in `self.transaction.complete()` where `transaction` is a property whose
    /// getter isn't annotated.
    fn receiver_type(&self, expr: &Expr) -> Option<Type<'db>> {
        if let Expr::Name(name) = expr
            && self.hasattr_names.contains(&name.id)
        {
            let declarations = self.name_declarations(name.id.as_str(), name.into());
            return self.reachable_name_type(name, &declarations);
        }
        let ty = expr.inferred_type(self.model);
        if ty.is_some_and(|ty| !ty.is_unknown()) {
            return ty;
        }
        let inferred = match expr {
            Expr::Attribute(attribute) if attribute.ctx.is_load() => {
                let receiver = self.receiver_type(&attribute.value)?;
                self.member_read_type(receiver, attribute.attr.as_str())
            }
            Expr::Call(call) => self
                .receiver_type(&call.func)
                .and_then(|callee| pyright_inferred_call_type(self.model, callee)),
            _ => None,
        };
        inferred.or(ty)
    }

    /// The type of reading `name` through `receiver`, with the inferred return type of a property
    /// getter that has no return annotation.
    fn member_read_type(&self, receiver: Type<'db>, name: &str) -> Option<Type<'db>> {
        let declarations = self.attribute_declarations(receiver, name);
        if let Some(declaration) = declarations.first()
            && declaration.kind == PyrightDeclarationKind::Function
            && let Some(definition) = declaration.definition
            && let PyrightAccessor::Accessor {
                effective_type: Some(effective_type),
                ..
            } = pyright_method_accessor(self.model, definition, &declarations, false)
        {
            return Some(effective_type);
        }
        pyright_member_type(self.model, receiver, name).filter(|ty| !ty.is_unknown())
    }

    /// The type of `x.name` in reachable code.
    fn reachable_attribute_type(
        &self,
        attribute: &ast::ExprAttribute,
        receiver: Option<Type<'db>>,
        declarations: &[PyrightDeclaration<'db>],
        is_declared: bool,
    ) -> Option<Type<'db>> {
        let db = self.db();
        let name = attribute.attr.as_str();
        let from_declaration = || {
            declarations
                .first()
                .and_then(|declaration| declaration.definition)
                .and_then(|definition| pyright_definition_type(db, definition))
        };
        let Some(receiver) = receiver else {
            return attribute
                .inferred_type(self.model)
                .or_else(from_declaration);
        };
        // In a type expression, ty records the type of the annotated value (an instance of `C`
        // for `m.C`), while pyright evaluates the member itself.
        if self
            .flags
            .intersects(VisitFlags::TYPE_FORM | VisitFlags::TYPE_ARGUMENT)
            && let Some(member) = pyright_member_type(self.model, receiver, name)
        {
            // Module variables are only valid in a type expression if they are type aliases
            // (see `is_valid_in_type_form`).
            if self.flags.contains(VisitFlags::TYPE_FORM)
                && matches!(receiver, Type::ModuleLiteral(_))
                && !self.is_valid_in_type_form(declarations, Some(member))
            {
                return Some(Type::unknown());
            }
            return Some(member);
        }
        // ty types every member of `type[Unknown]` as unknown, while pyright finds it on `type`.
        if pyright_is_dynamic_class_object(receiver) && !declarations.is_empty() {
            return from_declaration();
        }
        // ty's `super()` doesn't find the instance attributes that the next class in the MRO
        // assigns, while pyright's does.
        if matches!(receiver, Type::BoundSuper(_))
            && attribute
                .inferred_type(self.model)
                .is_none_or(|ty| ty.is_unknown())
        {
            return from_declaration();
        }
        let mut ty = attribute
            .inferred_type(self.model)
            .or_else(from_declaration);
        // ty widens a function stored in a class attribute (`concat = "".join`) to a callable
        // type without its definition, while pyright keeps the function.
        if matches!(ty, Some(Type::Callable(_)))
            && declarations
                .first()
                .is_some_and(PyrightDeclaration::is_variable)
            && let Some(bound @ (Type::FunctionLiteral(_) | Type::BoundMethod(_))) =
                from_declaration()
        {
            ty = Some(bound);
        }
        // ty has no type for the members of a receiver that only pyright's return type inference
        // knows (see `receiver_type`).
        if attribute.ctx.is_load()
            && ty.is_none_or(|ty| ty.is_unknown())
            && attribute
                .value
                .inferred_type(self.model)
                .is_none_or(|ty| ty.is_unknown())
            && let Some(member) = self.member_read_type(receiver, name)
        {
            ty = Some(member);
        }
        match attribute.ctx {
            // A member that only a `hasattr()` check provides is unknown for pyright.
            ExprContext::Load
                if let Some(stripped) = pyright_hasattr_receiver(self.model, receiver) =>
            {
                Some(pyright_member_type(self.model, stripped, name).unwrap_or_else(Type::unknown))
            }
            // ty narrows an unknown receiver to `Unknown & F` (after `isinstance(x, F)`), and the
            // attribute's type to `Unknown`; pyright narrows the receiver to `F`. Pyright also
            // keeps the declared type of an attribute that ty narrowed to an assigned `Any`.
            ExprContext::Load => {
                if is_declared
                    && ty.is_some_and(pyright_is_explicit_any)
                    && let Some(declared) = pyright_member_type(self.model, receiver, name)
                {
                    return Some(declared);
                }
                pyright_narrowed_receiver_member_type(self.model, receiver, name).or(ty)
            }
            // Pyright narrows the declared type of an assignment target to the assigned type,
            // but keeps the declared type where the value is `Any` or unknown
            // (`t.name = namespace["name"]`), or where the declared type is `Any`. ty types the
            // target as unknown where the assignment isn't allowed.
            ExprContext::Store => {
                if is_declared
                    && let Some(declared) = pyright_member_type(self.model, receiver, name)
                    && (pyright_is_explicit_any(declared)
                        || ty.is_none_or(|ty| {
                            let ty = pyright_type(self.model, ty, PyrightTypeContext::Value);
                            ty.is_any_or_unknown && !ty.is_special_form
                        }))
                {
                    Some(declared)
                } else {
                    ty
                }
            }
            // Pyright types a deleted attribute with its declared type, and as unknown if it has
            // none.
            ExprContext::Del => {
                if is_declared {
                    pyright_member_type(self.model, receiver, name).or(ty)
                } else {
                    Some(Type::unknown())
                }
            }
            ExprContext::Invalid => ty,
        }
    }

    fn visit_keyword_argument(
        &mut self,
        keyword: &ast::Keyword,
        call: &ast::ExprCall,
        argument: Option<PyrightKeywordArgument<'db>>,
    ) {
        // An invalid keyword (`f(x.y=1)`) has an empty name.
        if let Some(name) = &keyword.arg
            && !name.is_empty()
            && let Some(argument) = argument
        {
            let mut declarations = self.declarations(&argument.definitions);
            // The fields of functional named tuples have synthesized declarations in pyright.
            if declarations.is_empty()
                && let Some(callee) = call.func.inferred_type(self.model)
                && let Some(field) =
                    pyright_functional_named_tuple_field(self.model, callee, name.as_str())
            {
                declarations.push(field);
            }
            let name_type = self.name_type(argument.ty, PyrightTypeContext::Value);
            let context = NameContext {
                is_keyword_argument: true,
                ..NameContext::new()
            };
            let classification = self.classify_name(context, &declarations, name_type);
            self.add_classified(name.range(), classification);
        }
        self.visit_expr(&keyword.value);
    }

    /// Classifies a name that is bound by a definition without an expression node, such as an
    /// exception handler's `as` name or a match-pattern capture.
    fn visit_bound_identifier(&mut self, identifier: &ast::Identifier, ty: Option<Type<'db>>) {
        // Pyright has neither a declaration nor a type for names bound in code that its binder
        // skips.
        if self.reachability(identifier.range()) == Reachability::Unbound {
            return;
        }
        let declarations =
            self.name_declarations(identifier.as_str(), AnyNodeRef::Identifier(identifier));
        let name_type = self.name_type(ty, PyrightTypeContext::StoreTarget);
        let context = NameContext {
            is_write: true,
            ..NameContext::new()
        };
        let classification = self.classify_name(context, &declarations, name_type);
        self.add_classified(identifier.range(), classification);
    }

    fn visit_capture(&mut self, name: &ast::Identifier) {
        let db = self.db();
        let ty = semantic_index(db, self.model.program_file())
            .try_definitions(AnyNodeRef::Identifier(name))
            .and_then(|definitions| definitions.first())
            .and_then(|definition| pyright_definition_type(db, *definition));
        self.visit_bound_identifier(name, ty);
    }

    fn visit_decorator_expression(&mut self, decorator: &ast::Decorator) {
        // The `@` itself.
        self.add_token(
            TextRange::at(decorator.start(), TextSize::from(1)),
            SemanticTokenType::Decorator,
            SemanticTokenModifier::empty(),
        );
        match &decorator.expression {
            // Simple decorators like `@staticmethod` get the decorator token; the parts of more
            // complex decorators like `@app.route("/")` are classified like any other expression.
            Expr::Name(name) => self.add_token(
                name.range(),
                SemanticTokenType::Decorator,
                SemanticTokenModifier::empty(),
            ),
            expression => self.visit_value(expression),
        }
    }

    fn visit_function_name(&mut self, function: &ast::StmtFunctionDef) {
        let mut modifiers = SemanticTokenModifier::DECLARATION;
        if function.is_async {
            modifiers |= SemanticTokenModifier::ASYNC;
        }
        // Pyright has no declaration for a function that its binder skips.
        if self.reachability(function.name.range()) == Reachability::Unbound {
            self.add_token(
                function.name.range(),
                SemanticTokenType::Function,
                modifiers,
            );
            return;
        }
        let definition = function.definition(self.model);
        let declarations = self.symbol_declarations(definition);
        let own_declaration = declarations
            .iter()
            .find(|declaration| declaration.definition == Some(definition))
            .copied()
            .unwrap_or_else(|| pyright_declaration(self.model, definition));
        let token_type = self.function_token_type(
            NameContext::new(),
            Some(&own_declaration),
            &declarations,
            None,
            &mut modifiers,
        );
        self.add_token(function.name.range(), token_type, modifiers);
    }

    fn visit_class_name(&mut self, class: &ast::StmtClassDef) {
        let db = self.db();
        let definition = class.definition(self.model);
        let mut modifiers = SemanticTokenModifier::DECLARATION;
        // Pyright classifies the class itself, before its decorators are applied.
        let ty = pyright_type(
            self.model,
            pyright_undecorated_type(db, definition),
            PyrightTypeContext::StoreTarget,
        );
        let token_type = if !ty.is_class() {
            SemanticTokenType::Class
        } else if self.reachability(class.name.range()) == Reachability::Unbound {
            // Pyright has no declaration for a class that its binder skips.
            self.class_token_type(NameContext::new(), ty, &[], &mut modifiers, false, false)
        } else {
            let declarations = self.symbol_declarations(definition);
            self.class_token_type(
                NameContext::new(),
                ty,
                &declarations,
                &mut modifiers,
                false,
                true,
            )
        };
        self.add_token(class.name.range(), token_type, modifiers);
    }

    /// pyright's `visitParameter`.
    fn visit_parameter_name(&mut self, parameter: &ast::Parameter) {
        // Pyright has no type for the parameters of a lambda in a branch it doesn't evaluate.
        if self.flags.contains(VisitFlags::SKIPPED_BRANCH) {
            self.add_token(
                parameter.name.range(),
                SemanticTokenType::Parameter,
                SemanticTokenModifier::DECLARATION,
            );
            return;
        }
        let db = self.db();
        let Some(definition) =
            semantic_index(db, self.model.program_file()).try_definition(parameter)
        else {
            self.add_token(
                parameter.name.range(),
                SemanticTokenType::Parameter,
                SemanticTokenModifier::DECLARATION | SemanticTokenModifier::PARAMETER,
            );
            return;
        };
        let declarations = self.name_declarations(
            parameter.name.as_str(),
            AnyNodeRef::Identifier(&parameter.name),
        );
        let mut modifiers = SemanticTokenModifier::DECLARATION;
        let token_type = match self.name_type(
            pyright_definition_type(db, definition),
            PyrightTypeContext::StoreTarget,
        ) {
            Some(name_type) => self.parameter_token_type(
                NameContext::new(),
                Some(definition),
                name_type,
                &declarations,
                &mut modifiers,
            ),
            None => SemanticTokenType::Parameter,
        };
        self.add_token(parameter.name.range(), token_type, modifiers);
    }

    /// A lambda inside a string annotation. Pyright binds its parameters, but has no type for them.
    fn visit_string_annotation_lambda(&mut self, lambda: &ast::ExprLambda) {
        let enclosing_parameters = self.string_lambda_parameters.len();
        if let Some(parameters) = &lambda.parameters {
            for parameter in parameters.iter_source_order() {
                if let Some(default) = parameter.default() {
                    self.visit_value(default);
                }
                self.add_classified(
                    parameter.as_parameter().name.range(),
                    Some((
                        SemanticTokenType::Parameter,
                        SemanticTokenModifier::DECLARATION | SemanticTokenModifier::PARAMETER,
                    )),
                );
            }
            self.string_lambda_parameters.extend(
                parameters
                    .iter_source_order()
                    .map(|parameter| parameter.as_parameter().name.id.clone()),
            );
        }
        self.visit_expr(&lambda.body);
        self.string_lambda_parameters.truncate(enclosing_parameters);
    }

    fn visit_parameters(&mut self, parameters: &ast::Parameters) {
        for parameter in parameters.iter_source_order() {
            self.visit_parameter_name(parameter.as_parameter());
            if let Some(annotation) = &parameter.as_parameter().annotation {
                self.visit_annotation(annotation);
            }
            if let Some(default) = parameter.default() {
                self.visit_value(default);
            }
        }
    }

    /// Visits `expr` with `flags` in place of the current flags.
    fn visit_expr_with_flags(&mut self, expr: &Expr, flags: VisitFlags) {
        let saved = std::mem::replace(&mut self.flags, flags);
        self.visit_expr(expr);
        self.flags = saved;
    }

    /// The flags that carry over into any nested expression.
    fn inherited_flags(&self) -> VisitFlags {
        self.flags & (VisitFlags::SKIPPED_BRANCH | VisitFlags::UNTYPED | VisitFlags::OPAQUE)
    }

    fn visit_value(&mut self, expr: &Expr) {
        self.visit_expr_with_flags(expr, self.inherited_flags());
    }

    fn visit_target(&mut self, expr: &Expr) {
        self.visit_expr_with_flags(expr, self.inherited_flags() | VisitFlags::WRITE);
    }

    fn visit_import_module_name(&mut self, name: &ast::Identifier) {
        // Every part of a dotted module name is a namespace, whether or not it resolves.
        let mut offset = name.start();
        for part in name.as_str().split('.') {
            let len = TextSize::of(part);
            self.add_token(
                TextRange::at(offset, len),
                SemanticTokenType::Namespace,
                SemanticTokenModifier::empty(),
            );
            offset += len + TextSize::of('.');
        }
    }

    fn visit_type_param_name(&mut self, type_param: &TypeParam) {
        if self.reachability(type_param.range()) != Reachability::Unbound {
            self.add_token(
                type_param.name().range(),
                SemanticTokenType::TypeParameter,
                SemanticTokenModifier::empty(),
            );
        }
        match type_param {
            TypeParam::TypeVar(type_var) => {
                if let Some(bound) = &type_var.bound {
                    self.visit_value(bound);
                }
                if let Some(default) = &type_var.default {
                    self.visit_annotation(default);
                }
            }
            TypeParam::ParamSpec(param_spec) => {
                if let Some(default) = &param_spec.default {
                    self.visit_annotation(default);
                }
            }
            TypeParam::TypeVarTuple(type_var_tuple) => {
                if let Some(default) = &type_var_tuple.default {
                    self.visit_annotation(default);
                }
            }
        }
    }

    fn visit_class_keyword(&mut self, keyword: &ast::Keyword) {
        // Class-header keywords have no declarations. Pyright records the argument's type for
        // keywords that `__init_subclass__` accepts, but never for `metaclass`.
        if let Some(name) = &keyword.arg
            && !name.is_empty()
            && name.as_str() != "metaclass"
        {
            let name_type = self.name_type(
                keyword.value.inferred_type(self.model),
                PyrightTypeContext::Value,
            );
            let context = NameContext {
                is_keyword_argument: true,
                ..NameContext::new()
            };
            let classification = self.classify_name(context, &[], name_type);
            self.add_classified(name.range(), classification);
        }
        self.visit_value(&keyword.value);
    }

    /// Whether pyright treats the name assigned by `target` as a type alias, whose value it
    /// evaluates without converting special forms. The value must evaluate to a type.
    fn is_type_alias_target(&self, target: &Expr) -> bool {
        let Expr::Name(name) = target else {
            return false;
        };
        semantic_index(self.db(), self.model.program_file())
            .try_definition(name)
            .is_some_and(|definition| self.is_type_alias_definition(definition))
    }

    /// Whether pyright treats `definition` as a type alias (see [`Self::is_type_alias_target`]).
    fn is_type_alias_definition(&self, definition: Definition<'db>) -> bool {
        let db = self.db();
        pyright_is_type_alias_declaration(db, definition)
            && pyright_definition_type(db, definition).is_some_and(|ty| {
                let ty = pyright_type(self.model, ty, PyrightTypeContext::TypeArgument);
                ty.category == PyrightTypeCategory::Never
                    || (ty.is_instantiable && (!ty.is_any_or_unknown || ty.is_special_form))
            })
    }

    /// How pyright evaluates the two branches of a conditional expression.
    fn branch_modes(&self, if_expr: &ast::ExprIf) -> (BranchMode, BranchMode) {
        match self.reachability(if_expr.range()) {
            // Pyright doesn't evaluate either branch in code that it finds unreachable, but only
            // the binder's unreachable code has no declarations.
            Reachability::Unevaluated => return (BranchMode::Untyped, BranchMode::Untyped),
            Reachability::Unbound => return (BranchMode::Skipped, BranchMode::Skipped),
            Reachability::Reachable => {}
        }
        let strict = self.conditions.strict_truthiness(&if_expr.test);
        // The binder also marks a branch unreachable for other static conditions, which only
        // matters for branches that pyright checks for reachability on their own.
        let binder = self.conditions.truthiness(&if_expr.test);
        let is_checked_on_its_own = |branch: &Expr| {
            matches!(
                branch,
                Expr::Name(_) | Expr::Attribute(_) | Expr::Subscript(_) | Expr::Lambda(_)
            )
        };
        let mode = |skipped: bool| {
            if skipped {
                BranchMode::Skipped
            } else {
                BranchMode::Evaluated
            }
        };
        (
            mode(
                strict == Some(false)
                    || (binder == Some(false) && is_checked_on_its_own(&if_expr.body)),
            ),
            mode(
                strict == Some(true)
                    || (binder == Some(true) && is_checked_on_its_own(&if_expr.orelse)),
            ),
        )
    }
}

impl SourceOrderVisitor<'_> for SemanticTokenVisitor<'_> {
    fn enter_node(&mut self, node: AnyNodeRef<'_>) -> TraversalSignal {
        // If we have a range filter and this node doesn't intersect, skip it
        // and all its children as an optimization
        if self.is_outside_range_filter(node.range()) {
            TraversalSignal::Skip
        } else {
            TraversalSignal::Traverse
        }
    }

    fn visit_stmt(&mut self, stmt: &Stmt) {
        if self.is_outside_range_filter(stmt.range()) {
            return;
        }
        // Statements never inherit flags from an enclosing expression.
        let saved = std::mem::take(&mut self.flags);
        self.visit_stmt_without_flags(stmt);
        self.flags = saved;
    }

    /// Visit an annotation or other expression that should be interpreted as a type form.
    fn visit_annotation(&mut self, expr: &'_ Expr) {
        self.visit_expr_with_flags(
            expr,
            self.inherited_flags() | VisitFlags::TYPE_FORM | VisitFlags::PARSE_STRINGS,
        );
    }

    fn visit_expr(&mut self, expr: &Expr) {
        if self.is_outside_range_filter(expr.range()) {
            return;
        }
        // Only names, attributes and calls are evaluated on their own in an expression that
        // pyright doesn't evaluate otherwise.
        if self.flags.contains(VisitFlags::BASE)
            && !matches!(expr, Expr::Name(_) | Expr::Attribute(_) | Expr::Call(_))
        {
            self.visit_expr_with_flags(expr, self.flags - VisitFlags::BASE);
            return;
        }
        match expr {
            Expr::Name(name) => self.visit_name_expr(name),
            Expr::Attribute(attribute) => self.visit_attribute_expr(attribute, true),
            Expr::If(if_expr) => {
                let (body_mode, else_mode) = self.branch_modes(if_expr);
                let flags = self.flags;
                let branch_flags = |mode: BranchMode| match mode {
                    BranchMode::Evaluated => flags,
                    BranchMode::Skipped => flags | VisitFlags::SKIPPED_BRANCH,
                    BranchMode::Untyped if flags.contains(VisitFlags::SKIPPED_BRANCH) => flags,
                    BranchMode::Untyped => flags | VisitFlags::UNTYPED,
                };
                self.visit_expr_with_flags(&if_expr.body, branch_flags(body_mode));
                self.visit_expr_with_flags(&if_expr.test, flags.base());
                self.visit_expr_with_flags(&if_expr.orelse, branch_flags(else_mode));
            }
            Expr::Lambda(lambda) if self.model.is_in_string_annotation() => {
                self.visit_string_annotation_lambda(lambda);
            }
            Expr::Lambda(lambda) => {
                if let Some(parameters) = &lambda.parameters {
                    self.visit_parameters(parameters);
                }
                // A lambda's body is bound even in a skipped branch, but it has no types.
                let body_flags = if self.flags.contains(VisitFlags::SKIPPED_BRANCH) {
                    (self.flags - VisitFlags::SKIPPED_BRANCH) | VisitFlags::UNTYPED
                } else {
                    self.flags
                };
                self.visit_expr_with_flags(&lambda.body, body_flags);
            }
            Expr::Named(named) => {
                self.visit_target(&named.target);
                self.visit_expr_with_flags(&named.value, self.flags - VisitFlags::WRITE);
            }
            Expr::StringLiteral(string_expr) => {
                // Pyright parses the contents of string annotations and classifies the names
                // inside them, but not the contents of other strings.
                if self.flags.contains(VisitFlags::PARSE_STRINGS)
                    && let Some((sub_ast, sub_model)) =
                        self.model.enter_string_annotation(string_expr)
                {
                    let mut sub_visitor = SemanticTokenVisitor::with_reachability(
                        &sub_model,
                        self.range_filter,
                        self.unbound.clone(),
                        self.conditions.clone(),
                    );
                    sub_visitor.flags =
                        self.inherited_flags() | VisitFlags::TYPE_FORM | VisitFlags::PARSE_STRINGS;
                    sub_visitor.visit_expr(sub_ast.expr());
                    self.tokens.extend(sub_visitor.tokens);
                }
            }
            Expr::Subscript(subscript) if self.flags.contains(VisitFlags::TYPE_FORM) => {
                self.visit_expr(&subscript.value);
                // Pyright doesn't parse strings inside `Literal[...]` and `Annotated[...]`.
                let flags = self.flags - VisitFlags::PARSE_STRINGS;
                match subscript.value.inferred_type(self.model) {
                    Some(Type::SpecialForm(SpecialFormType::Annotated)) => {
                        // The first argument of `Annotated` is a type expression; the metadata
                        // that follows are value expressions.
                        if let Expr::Tuple(tuple) = subscript.slice.as_ref()
                            && let Some((annotation, metadata)) = tuple.elts.split_first()
                        {
                            self.visit_expr_with_flags(annotation, flags);
                            for element in metadata {
                                if let Expr::StringLiteral(string) = element {
                                    self.add_classified(
                                        string.range(),
                                        Some((
                                            SemanticTokenType::String,
                                            SemanticTokenModifier::empty(),
                                        )),
                                    );
                                } else {
                                    self.visit_value(element);
                                }
                            }
                        } else {
                            self.visit_expr_with_flags(&subscript.slice, flags);
                        }
                    }
                    Some(Type::SpecialForm(SpecialFormType::Literal)) => {
                        self.visit_expr_with_flags(&subscript.slice, flags);
                    }
                    _ => self.visit_expr(&subscript.slice),
                }
            }
            Expr::Subscript(subscript) => {
                self.visit_expr(&subscript.value);
                // The subscript of a special form or a generic class holds type arguments; pyright
                // doesn't parse strings among them.
                let base = (self.reachability(subscript.range()) == Reachability::Reachable)
                    .then(|| subscript.value.inferred_type(self.model))
                    .flatten();
                let inherited = self.flags
                    & (VisitFlags::SKIPPED_BRANCH
                        | VisitFlags::UNTYPED
                        | VisitFlags::OPAQUE
                        | VisitFlags::WRITE);
                let flags = match base {
                    Some(Type::SpecialForm(_) | Type::KnownInstance(_)) => {
                        inherited | VisitFlags::TYPE_FORM
                    }
                    Some(base) if pyright_is_generic_class_subscript(self.model, base) => {
                        inherited | VisitFlags::TYPE_ARGUMENT
                    }
                    _ => inherited,
                };
                self.visit_expr_with_flags(&subscript.slice, flags);
            }
            Expr::Call(call) => {
                // In an expression that pyright doesn't evaluate, it still evaluates the callee,
                // and the whole call if the call is itself evaluated on its own (as a receiver,
                // for example). Otherwise nothing in the arguments is evaluated.
                self.visit_expr_with_flags(&call.func, self.flags.base());
                let is_evaluated = !self.flags.is_unevaluated()
                    || (self.flags.contains(VisitFlags::BASE)
                        && !self.flags.contains(VisitFlags::OPAQUE));
                let inherited = if is_evaluated {
                    self.flags & VisitFlags::WRITE
                } else {
                    (self.flags
                        & (VisitFlags::SKIPPED_BRANCH | VisitFlags::UNTYPED | VisitFlags::WRITE))
                        | VisitFlags::OPAQUE
                };
                let mut keyword_arguments =
                    if !is_evaluated || call.arguments.keywords.is_empty() {
                        Vec::new()
                    } else {
                        pyright_keyword_arguments(self.model, call)
                    }
                    .into_iter();
                // Arguments such as the first argument of `cast` are type expressions, but pyright
                // doesn't parse string arguments as annotations.
                let argument_forms = call_argument_forms(self.model, call);
                for (argument, form) in call.arguments.iter_source_order().zip(argument_forms) {
                    let flags = if form == CallArgumentForm::Type {
                        inherited | VisitFlags::TYPE_FORM
                    } else {
                        inherited
                    };
                    let saved = std::mem::replace(&mut self.flags, flags);
                    match argument {
                        ast::ArgOrKeyword::Arg(argument) => self.visit_expr(argument),
                        ast::ArgOrKeyword::Keyword(keyword) => {
                            let resolved = keyword_arguments.next();
                            self.visit_keyword_argument(keyword, call, resolved);
                        }
                    }
                    self.flags = saved;
                }
            }
            _ => walk_expr(self, expr),
        }
    }

    fn visit_comprehension(&mut self, comprehension: &ast::Comprehension) {
        self.visit_expr_with_flags(&comprehension.iter, self.flags - VisitFlags::WRITE);
        self.visit_target(&comprehension.target);
        for condition in &comprehension.ifs {
            self.visit_expr(condition);
        }
    }

    fn visit_except_handler(&mut self, except_handler: &ast::ExceptHandler) {
        let ast::ExceptHandler::ExceptHandler(handler) = except_handler;
        if let Some(type_expr) = &handler.type_ {
            self.visit_value(type_expr);
        }
        if let Some(name) = &handler.name {
            self.visit_bound_identifier(name, handler.inferred_type(self.model));
        }
        self.visit_body(&handler.body);
    }

    fn visit_pattern(&mut self, pattern: &ast::Pattern) {
        match pattern {
            ast::Pattern::MatchAs(pattern_as) => {
                if let Some(nested_pattern) = &pattern_as.pattern {
                    self.visit_pattern(nested_pattern);
                }
                if let Some(name) = &pattern_as.name {
                    self.visit_capture(name);
                }
            }
            ast::Pattern::MatchMapping(pattern_mapping) => {
                // `**rest` can appear before or after the key-value pairs
                // (the parser can produce either AST, but emits an
                // invalid-syntax error for the former).
                let rest_before_keys = pattern_mapping.rest.as_ref().filter(|rest| {
                    pattern_mapping
                        .keys
                        .first()
                        .is_some_and(|key| rest.start() < key.start())
                });
                if let Some(rest_name) = rest_before_keys {
                    self.visit_capture(rest_name);
                }
                for (key, nested_pattern) in
                    pattern_mapping.keys.iter().zip(&pattern_mapping.patterns)
                {
                    self.visit_value(key);
                    self.visit_pattern(nested_pattern);
                }
                if let Some(rest_name) = &pattern_mapping.rest
                    && rest_before_keys.is_none()
                {
                    self.visit_capture(rest_name);
                }
            }
            ast::Pattern::MatchStar(pattern_star) => {
                if let Some(rest_name) = &pattern_star.name {
                    self.visit_capture(rest_name);
                }
            }
            ast::Pattern::MatchClass(pattern_class) => {
                self.visit_value(&pattern_class.cls);
                for nested_pattern in &pattern_class.arguments.patterns {
                    self.visit_pattern(nested_pattern);
                }
                for keyword in &pattern_class.arguments.keywords {
                    // Pyright gives keyword names in class patterns the default variable token.
                    if self.reachability(keyword.range()) != Reachability::Unbound {
                        self.add_token(
                            keyword.attr.range(),
                            SemanticTokenType::Variable,
                            SemanticTokenModifier::empty(),
                        );
                    }
                    self.visit_pattern(&keyword.pattern);
                }
            }
            _ => walk_pattern(self, pattern),
        }
    }
}

/// Collects the names that `condition` passes to `hasattr()`, through `not`, `and` and `or`.
fn collect_hasattr_names(condition: &Expr, names: &mut Vec<Name>) {
    match condition {
        Expr::Call(call) if matches!(call.func.as_ref(), Expr::Name(func) if func.id.as_str() == "hasattr") => {
            if let Some(Expr::Name(name)) = call.arguments.args.first() {
                names.push(name.id.clone());
            }
        }
        Expr::UnaryOp(unary) if unary.op == ast::UnaryOp::Not => {
            collect_hasattr_names(&unary.operand, names);
        }
        Expr::BoolOp(bool_op) => {
            for value in &bool_op.values {
                collect_hasattr_names(value, names);
            }
        }
        _ => {}
    }
}

impl SemanticTokenVisitor<'_> {
    fn visit_stmt_without_flags(&mut self, stmt: &Stmt) {
        match stmt {
            Stmt::FunctionDef(function) => {
                self.visit_function_name(function);
                for decorator in &function.decorator_list {
                    self.visit_decorator_expression(decorator);
                }
                if let Some(type_params) = &function.type_params {
                    for type_param in &type_params.type_params {
                        self.visit_type_param_name(type_param);
                    }
                }
                self.visit_parameters(&function.parameters);
                if let Some(returns) = &function.returns {
                    self.visit_annotation(returns);
                }
                self.visit_body(&function.body);
            }
            Stmt::ClassDef(class) => {
                self.visit_class_name(class);
                for decorator in &class.decorator_list {
                    self.visit_decorator_expression(decorator);
                }
                if let Some(type_params) = &class.type_params {
                    for type_param in &type_params.type_params {
                        self.visit_type_param_name(type_param);
                    }
                }
                if let Some(arguments) = &class.arguments {
                    for argument in arguments.iter_source_order() {
                        match argument {
                            ast::ArgOrKeyword::Arg(base) => self.visit_value(base),
                            ast::ArgOrKeyword::Keyword(keyword) => {
                                self.visit_class_keyword(keyword);
                            }
                        }
                    }
                }
                self.visit_body(&class.body);
            }
            Stmt::TypeAlias(type_alias) => {
                // The `type` soft keyword.
                self.add_token(
                    TextRange::at(type_alias.start(), TextSize::from(4)),
                    SemanticTokenType::Keyword,
                    SemanticTokenModifier::empty(),
                );
                if let Expr::Name(name) = type_alias.name.as_ref() {
                    let declarations = self.name_declarations(name.id.as_str(), name.into());
                    let name_type = self.name_type(
                        name.inferred_type(self.model),
                        PyrightTypeContext::TypeExpression,
                    );
                    let classification =
                        self.classify_name(NameContext::new(), &declarations, name_type);
                    self.add_classified(name.range(), classification);
                }
                if let Some(type_params) = &type_alias.type_params {
                    for type_param in &type_params.type_params {
                        self.visit_type_param_name(type_param);
                    }
                }
                self.visit_annotation(&type_alias.value);
            }
            Stmt::Import(import) => {
                // Pyright has no declarations for the names that imports in code that its binder
                // skips bind.
                let is_bound = self.reachability(import.range()) != Reachability::Unbound;
                for alias in &import.names {
                    self.visit_import_module_name(&alias.name);
                    // The alias of a module import is a namespace, even if the module doesn't
                    // resolve.
                    if is_bound && let Some(asname) = &alias.asname {
                        self.add_token(
                            asname.range(),
                            SemanticTokenType::Namespace,
                            SemanticTokenModifier::empty(),
                        );
                    }
                }
            }
            Stmt::ImportFrom(import) => {
                if let Some(module) = &import.module {
                    self.visit_import_module_name(module);
                }
                // Pyright's binder records the names imported from `typing` even in code that it
                // skips otherwise.
                let is_bound = self.reachability(import.range()) != Reachability::Unbound
                    || import.module.as_ref().is_some_and(|module| {
                        matches!(module.as_str(), "typing" | "typing_extensions")
                    });
                for alias in &import.names {
                    if !is_bound || alias.name.as_str() == "*" {
                        continue;
                    }
                    let declarations = self.declarations(&definitions_for_imported_symbol(
                        self.model,
                        import,
                        alias.name.as_str(),
                        ImportAliasResolution::ResolveAliases,
                    ));
                    let name_type = self.name_type(
                        alias.inferred_type(self.model),
                        PyrightTypeContext::ImportedName,
                    );
                    for identifier in std::iter::once(&alias.name).chain(alias.asname.as_ref()) {
                        let context = NameContext {
                            is_write: true,
                            ..NameContext::new()
                        };
                        let classification = self.classify_name(context, &declarations, name_type);
                        self.add_classified(identifier.range(), classification);
                    }
                }
            }
            Stmt::Global(ast::StmtGlobal { names, .. })
            | Stmt::Nonlocal(ast::StmtNonlocal { names, .. }) => {
                // Pyright gives these names the effective type of the symbol they refer to.
                for identifier in names {
                    let declarations = self
                        .name_declarations(identifier.as_str(), AnyNodeRef::Identifier(identifier));
                    let name_type = self.name_type(
                        self.declarations_effective_type(&declarations),
                        PyrightTypeContext::Value,
                    );
                    let classification =
                        self.classify_name(NameContext::new(), &declarations, name_type);
                    self.add_classified(identifier.range(), classification);
                }
            }
            Stmt::Assign(assignment) => {
                // Pyright evaluates the value of an implicit type alias (`JSON = Any`) without
                // converting special forms.
                let alias_flags = match assignment.targets.as_slice() {
                    [target] if self.is_type_alias_target(target) => VisitFlags::TYPE_ARGUMENT,
                    _ => VisitFlags::empty(),
                };
                for target in &assignment.targets {
                    self.visit_expr_with_flags(target, VisitFlags::WRITE | alias_flags);
                }
                self.visit_expr_with_flags(&assignment.value, alias_flags);
            }
            Stmt::AnnAssign(assignment) => {
                match (&assignment.value, assignment.target.as_ref()) {
                    // Pyright has no type for an attribute that is only declared (`self.x: int`),
                    // and doesn't count it as written.
                    (None, Expr::Attribute(attribute)) => {
                        self.visit_attribute_expr(attribute, false);
                    }
                    (None, target) => self.visit_value(target),
                    (Some(_), target) => {
                        let alias_flags = if self.is_type_alias_target(target) {
                            VisitFlags::TYPE_ARGUMENT
                        } else {
                            VisitFlags::empty()
                        };
                        self.visit_expr_with_flags(target, VisitFlags::WRITE | alias_flags);
                    }
                }
                self.visit_annotation(&assignment.annotation);
                if let Some(value) = &assignment.value {
                    // PEP 613 alias values are type forms even though they appear as annotated
                    // assignments rather than dedicated `type` statements.
                    if self.model.is_type_alias_annotation(&assignment.annotation) {
                        self.visit_annotation(value);
                    } else {
                        self.visit_value(value);
                    }
                }
            }
            Stmt::AugAssign(assignment) => {
                self.visit_target(&assignment.target);
                self.visit_value(&assignment.value);
            }
            Stmt::Delete(delete) => {
                for target in &delete.targets {
                    self.visit_target(target);
                }
            }
            Stmt::For(for_stmt) => {
                self.visit_target(&for_stmt.target);
                self.visit_value(&for_stmt.iter);
                self.visit_body(&for_stmt.body);
                self.visit_body(&for_stmt.orelse);
            }
            Stmt::With(with_stmt) => {
                for item in &with_stmt.items {
                    self.visit_target(&item.context_expr);
                    if let Some(expr) = &item.optional_vars {
                        self.visit_target(expr);
                    }
                }
                self.visit_body(&with_stmt.body);
            }
            Stmt::If(ast::StmtIf { test, .. }) | Stmt::While(ast::StmtWhile { test, .. }) => {
                let enclosing = self.hasattr_names.len();
                collect_hasattr_names(test, &mut self.hasattr_names);
                walk_stmt(self, stmt);
                self.hasattr_names.truncate(enclosing);
            }
            _ => walk_stmt(self, stmt),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use insta::assert_snapshot;
    use ruff_db::{
        files::{File, system_path_to_file},
        system::{DbWithWritableSystem, SystemPath, SystemPathBuf},
    };
    use ruff_text_size::TextLen;
    use ty_project::ProjectMetadata;

    #[test]
    fn semantic_tokens_basic() {
        let test = SemanticTokenTest::new("def foo(): pass");

        let tokens = test.highlight_file();

        assert_snapshot!(test.to_snapshot(&tokens), @r#""foo" @ 4..7: Function [declaration]"#);
    }

    #[test]
    fn semantic_tokens_class() {
        let test = SemanticTokenTest::new("class MyClass: pass");

        let tokens = test.highlight_file();

        assert_snapshot!(test.to_snapshot(&tokens), @r#""MyClass" @ 6..13: Class [declaration]"#);
    }

    #[test]
    fn semantic_tokens_class_args() {
        // This used to cause a panic because of an incorrect
        // insertion-order when visiting arguments inside
        // class definitions.
        let test = SemanticTokenTest::new("class Foo(m=x, m)");

        let tokens = test.highlight_file();

        assert_snapshot!(test.to_snapshot(&tokens), @r#""Foo" @ 6..9: Class [declaration]"#);
    }

    #[test]
    fn semantic_tokens_annotated_metadata() {
        let test = SemanticTokenTest::new(
            "
from typing import Annotated

class Metadata:
    field = 1

def f(x: Annotated[int, Metadata.field]): ...
",
        );

        let tokens = test.highlight_file();

        assert_snapshot!(test.to_snapshot(&tokens), @r#"
        "typing" @ 6..12: Namespace
        "Annotated" @ 20..29: Class
        "Metadata" @ 37..45: Class [declaration]
        "field" @ 51..56: Property [static, classMember]
        "f" @ 66..67: Function [declaration]
        "x" @ 68..69: Parameter [declaration, parameter]
        "Annotated" @ 71..80: Class
        "int" @ 81..84: Class [defaultLibrary, builtin]
        "Metadata" @ 86..94: Class
        "field" @ 95..100: Property [static, classMember]
        "#);
    }

    #[test]
    fn semantic_tokens_match_class_pattern_keyword_before_positional() {
        // Regression test for https://github.com/astral-sh/ty/issues/2417
        // This used to cause a panic because keyword patterns and positional
        // patterns in a match class were not visited in source order.
        let test = SemanticTokenTest::new(
            "
import ast
def f(x: ast.AST):
    match x:
        case ast.Attribute(value=ast.Name(id), attr):
            pass
",
        );

        let tokens = test.highlight_file();

        assert_snapshot!(test.to_snapshot(&tokens), @r#"
        "ast" @ 8..11: Namespace
        "f" @ 16..17: Function [declaration]
        "x" @ 18..19: Parameter [declaration, parameter]
        "ast" @ 21..24: Namespace
        "AST" @ 25..28: Class
        "x" @ 41..42: Parameter [parameter]
        "ast" @ 57..60: Namespace
        "Attribute" @ 61..70: Class
        "value" @ 71..76: Variable
        "ast" @ 77..80: Namespace
        "Name" @ 81..85: Class
        "id" @ 86..88: Variable
        "attr" @ 91..95: Variable
        "#);
    }

    #[test]
    fn semantic_tokens_match_mapping_pattern_rest_before_keys() {
        let test = SemanticTokenTest::new(
            "
def f(x):
    match x:
        case {**rest, 'key': value}:
            pass
",
        );

        let tokens = test.highlight_file();

        assert_snapshot!(test.to_snapshot(&tokens), @r#"
        "f" @ 5..6: Function [declaration]
        "x" @ 7..8: Parameter [declaration, parameter]
        "x" @ 21..22: Parameter [parameter]
        "rest" @ 40..44: Variable
        "#);
    }

    #[test]
    fn semantic_tokens_variables() {
        let test = SemanticTokenTest::new(
            "
x = 42
y = 'hello'
",
        );

        let tokens = test.highlight_file();

        assert_snapshot!(test.to_snapshot(&tokens), @r#"
        "x" @ 1..2: Variable
        "y" @ 8..9: Variable
        "#);
    }

    #[test]
    fn semantic_tokens_legacy_typevar() {
        let test = SemanticTokenTest::new(
            r#"
from typing import Generic, TypeVar

KT = TypeVar("KT")

class Box(Generic[KT]):
    value: KT
"#,
        );

        let tokens = test.highlight_file();

        assert_snapshot!(test.to_snapshot(&tokens), @r#"
        "typing" @ 6..12: Namespace
        "Generic" @ 20..27: Class
        "TypeVar" @ 29..36: Class
        "KT" @ 38..40: TypeParameter [readonly]
        "TypeVar" @ 43..50: Class
        "Box" @ 64..67: Class [declaration]
        "Generic" @ 68..75: Class
        "KT" @ 76..78: TypeParameter [readonly]
        "value" @ 86..91: Property [static, classMember]
        "KT" @ 93..95: TypeParameter [readonly]
        "#);
    }

    #[test]
    fn semantic_tokens_legacy_paramspec() {
        let test = SemanticTokenTest::new(
            r#"
from typing import Callable, ParamSpec

P = ParamSpec("P")

def decorator(func: Callable[P, int]) -> Callable[P, str]: ...
"#,
        );

        let tokens = test.highlight_file();

        assert_snapshot!(test.to_snapshot(&tokens), @r#"
        "typing" @ 6..12: Namespace
        "Callable" @ 20..28: Class
        "ParamSpec" @ 30..39: Class
        "P" @ 41..42: TypeParameter [readonly]
        "ParamSpec" @ 45..54: Class
        "decorator" @ 65..74: Function [declaration]
        "func" @ 75..79: Function [declaration, parameter]
        "Callable" @ 81..89: Class
        "P" @ 90..91: TypeParameter [readonly]
        "int" @ 93..96: Class [defaultLibrary, builtin]
        "Callable" @ 102..110: Class
        "P" @ 111..112: TypeParameter [readonly]
        "str" @ 114..117: Class [defaultLibrary, builtin]
        "#);
    }

    #[test]
    fn semantic_tokens_walrus() {
        let test = SemanticTokenTest::new(
            "
if x := 42:
    y = 'hello'
",
        );

        let tokens = test.highlight_file();

        assert_snapshot!(test.to_snapshot(&tokens), @r#"
        "x" @ 4..5: Variable
        "y" @ 17..18: Variable
        "#);
    }

    #[test]
    fn semantic_tokens_self_parameter() {
        let test = SemanticTokenTest::new(
            "
class MyClass:
    def method(self, x):
        self.x = 10

    def method_unidiomatic_self(self2):
        print(self2.x))
",
        );

        let tokens = test.highlight_file();

        assert_snapshot!(test.to_snapshot(&tokens), @r#"
        "MyClass" @ 7..14: Class [declaration]
        "method" @ 24..30: Method [declaration, classMember]
        "self" @ 31..35: SelfParameter [declaration, parameter]
        "x" @ 37..38: Parameter [declaration, parameter]
        "self" @ 49..53: SelfParameter [parameter]
        "x" @ 54..55: Property [classMember]
        "method_unidiomatic_self" @ 70..93: Method [declaration, classMember]
        "self2" @ 94..99: SelfParameter [declaration, parameter]
        "print" @ 110..115: Function [defaultLibrary, builtin]
        "self2" @ 116..121: SelfParameter [parameter]
        "x" @ 122..123: Property [classMember]
        "#);
    }

    #[test]
    fn semantic_tokens_cls_parameter() {
        let test = SemanticTokenTest::new(
            "
class MyClass:
    @classmethod
    def method(cls, x): print(cls)
",
        );

        let tokens = test.highlight_file();

        assert_snapshot!(test.to_snapshot(&tokens), @r#"
        "MyClass" @ 7..14: Class [declaration]
        "@" @ 20..21: Decorator
        "classmethod" @ 21..32: Decorator
        "method" @ 41..47: Method [declaration, classMember]
        "cls" @ 48..51: ClsParameter [declaration, parameter]
        "x" @ 53..54: Parameter [declaration, parameter]
        "print" @ 57..62: Function [defaultLibrary, builtin]
        "cls" @ 63..66: ClsParameter [parameter]
        "#);
    }

    #[test]
    fn semantic_tokens_staticmethod_parameter() {
        let test = SemanticTokenTest::new(
            "
class MyClass:
    @staticmethod
    def method(x, y): pass
",
        );

        let tokens = test.highlight_file();

        assert_snapshot!(test.to_snapshot(&tokens), @r#"
        "MyClass" @ 7..14: Class [declaration]
        "@" @ 20..21: Decorator
        "staticmethod" @ 21..33: Decorator
        "method" @ 42..48: Method [declaration, static, classMember]
        "x" @ 49..50: Parameter [declaration, parameter]
        "y" @ 52..53: Parameter [declaration, parameter]
        "#);
    }

    #[test]
    fn semantic_tokens_aliased_staticmethod_parameter() {
        let test = SemanticTokenTest::new(
            "
sm = staticmethod

class MyClass:
    @sm
    def method(x, y): pass
",
        );

        let tokens = test.highlight_file();

        assert_snapshot!(test.to_snapshot(&tokens), @r#"
        "sm" @ 1..3: Class
        "staticmethod" @ 6..18: Class [defaultLibrary, builtin]
        "MyClass" @ 26..33: Class [declaration]
        "@" @ 39..40: Decorator
        "sm" @ 40..42: Decorator
        "method" @ 51..57: Method [declaration, static, classMember]
        "x" @ 58..59: Parameter [declaration, parameter]
        "y" @ 61..62: Parameter [declaration, parameter]
        "#);
    }

    #[test]
    fn semantic_tokens_aliased_classmethod_parameter() {
        let test = SemanticTokenTest::new(
            "
cm = classmethod

class MyClass:
    @cm
    def method(cls, x): pass
",
        );

        let tokens = test.highlight_file();

        assert_snapshot!(test.to_snapshot(&tokens), @r#"
        "cm" @ 1..3: Class
        "classmethod" @ 6..17: Class [defaultLibrary, builtin]
        "MyClass" @ 25..32: Class [declaration]
        "@" @ 38..39: Decorator
        "cm" @ 39..41: Decorator
        "method" @ 50..56: Method [declaration, classMember]
        "cls" @ 57..60: ClsParameter [declaration, parameter]
        "x" @ 62..63: Parameter [declaration, parameter]
        "#);
    }

    #[test]
    fn semantic_tokens_custom_self_cls_names() {
        let test = SemanticTokenTest::new(
            "
class MyClass:
    def method(instance, x): pass
    @classmethod
    def other(klass, y): print(klass)
    def complex_method(instance, posonly, /, regular, *args, kwonly, **kwargs): pass
",
        );

        let tokens = test.highlight_file();

        assert_snapshot!(test.to_snapshot(&tokens), @r#"
        "MyClass" @ 7..14: Class [declaration]
        "method" @ 24..30: Method [declaration, classMember]
        "instance" @ 31..39: SelfParameter [declaration, parameter]
        "x" @ 41..42: Parameter [declaration, parameter]
        "@" @ 54..55: Decorator
        "classmethod" @ 55..66: Decorator
        "other" @ 75..80: Method [declaration, classMember]
        "klass" @ 81..86: ClsParameter [declaration, parameter]
        "y" @ 88..89: Parameter [declaration, parameter]
        "print" @ 92..97: Function [defaultLibrary, builtin]
        "klass" @ 98..103: ClsParameter [parameter]
        "complex_method" @ 113..127: Method [declaration, classMember]
        "instance" @ 128..136: SelfParameter [declaration, parameter]
        "posonly" @ 138..145: Parameter [declaration, parameter]
        "regular" @ 150..157: Parameter [declaration, parameter]
        "args" @ 160..164: Parameter [declaration, parameter]
        "kwonly" @ 166..172: Parameter [declaration, parameter]
        "kwargs" @ 176..182: Parameter [declaration, parameter]
        "#);
    }

    #[test]
    fn semantic_tokens_modifiers() {
        let test = SemanticTokenTest::new(
            "
class MyClass:
    CONSTANT = 42
    async def method(self): pass
",
        );

        let tokens = test.highlight_file();

        assert_snapshot!(test.to_snapshot(&tokens), @r#"
        "MyClass" @ 7..14: Class [declaration]
        "CONSTANT" @ 20..28: Property [readonly, static, classMember]
        "method" @ 48..54: Method [declaration, async, classMember]
        "self" @ 55..59: SelfParameter [declaration, parameter]
        "#);
    }

    #[test]
    fn semantic_classification_vs_heuristic() {
        let test = SemanticTokenTest::new(
            "
import sys
class MyClass:
    pass

def my_function():
    return 42

x = MyClass()
y = my_function()
z = sys.version
",
        );

        let tokens = test.highlight_file();

        assert_snapshot!(test.to_snapshot(&tokens), @r#"
        "sys" @ 8..11: Namespace
        "MyClass" @ 18..25: Class [declaration]
        "my_function" @ 41..52: Function [declaration]
        "x" @ 71..72: Variable
        "MyClass" @ 75..82: Class
        "y" @ 85..86: Variable
        "my_function" @ 89..100: Function
        "z" @ 103..104: Variable
        "sys" @ 107..110: Namespace
        "version" @ 111..118: Variable
        "#);
    }

    #[test]
    fn builtin_types() {
        let test = SemanticTokenTest::new(
            r#"
            type U = str | int

            class Test:
                a: int
                b: bool
                c: str
                d: float
                e: list[int]
                f: list[float]
                g: int | float
                h: U
            "#,
        );

        assert_snapshot!(test.to_snapshot(&test.highlight_file()), @r#"
        "type" @ 1..5: Keyword
        "U" @ 6..7: Type
        "str" @ 10..13: Class [defaultLibrary, builtin]
        "int" @ 16..19: Class [defaultLibrary, builtin]
        "Test" @ 27..31: Class [declaration]
        "a" @ 37..38: Property [static, classMember]
        "int" @ 40..43: Class [defaultLibrary, builtin]
        "b" @ 48..49: Property [static, classMember]
        "bool" @ 51..55: Class [defaultLibrary, builtin]
        "c" @ 60..61: Property [static, classMember]
        "str" @ 63..66: Class [defaultLibrary, builtin]
        "d" @ 71..72: Property [static, classMember]
        "float" @ 74..79: Class [defaultLibrary, builtin]
        "e" @ 84..85: Property [static, classMember]
        "list" @ 87..91: Class [defaultLibrary, builtin]
        "int" @ 92..95: Class [defaultLibrary, builtin]
        "f" @ 101..102: Property [static, classMember]
        "list" @ 104..108: Class [defaultLibrary, builtin]
        "float" @ 109..114: Class [defaultLibrary, builtin]
        "g" @ 120..121: Property [static, classMember]
        "int" @ 123..126: Class [defaultLibrary, builtin]
        "float" @ 129..134: Class [defaultLibrary, builtin]
        "h" @ 139..140: Property [static, classMember]
        "U" @ 142..143: Type
        "#);
    }

    #[test]
    fn semantic_tokens_range() {
        let test = SemanticTokenTest::new(
            "
def function1():
    x = 42
    return x

def function2():
    y = \"hello\"
    z = True
    return y + z
",
        );

        let full_tokens = test.highlight_file();

        // Get the range that covers only the second function
        // Hardcoded offsets: function2 starts at position 42, source ends at position 108
        let range = TextRange::new(TextSize::from(42u32), TextSize::from(108u32));

        let range_tokens = test.highlight_range(range);

        // Range-based tokens should have fewer tokens than full scan
        // (should exclude tokens from function1)
        assert!(range_tokens.len() < full_tokens.len());

        // Test both full tokens and range tokens with snapshots
        assert_snapshot!(test.to_snapshot(&full_tokens), @r#"
        "function1" @ 5..14: Function [declaration]
        "x" @ 22..23: Variable
        "x" @ 40..41: Variable
        "function2" @ 47..56: Function [declaration]
        "y" @ 64..65: Variable
        "z" @ 80..81: Variable
        "y" @ 100..101: Variable
        "z" @ 104..105: Variable
        "#);

        assert_snapshot!(test.to_snapshot(&range_tokens), @r#"
        "function2" @ 47..56: Function [declaration]
        "y" @ 64..65: Variable
        "z" @ 80..81: Variable
        "y" @ 100..101: Variable
        "z" @ 104..105: Variable
        "#);

        // Verify that no tokens from range_tokens have ranges outside the requested range
        for token in range_tokens.iter() {
            assert!(
                range.contains_range(token.range()),
                "Token at {:?} is outside requested range {:?}",
                token.range(),
                range
            );
        }
    }

    /// When a token starts right at where the requested range ends,
    /// don't include it in the semantic tokens.
    #[test]
    fn semantic_tokens_range_excludes_boundary_tokens() {
        let test = SemanticTokenTest::new(
            "
x = 1
y = 2
z = 3
",
        );

        // Range [6..13) starts where "1" ends and ends where "z" starts.
        // Expected: only "y" @ 7..8 and "2" @ 11..12 (non-empty overlap with target range).
        // Not included: "1" @ 5..6 and "z" @ 13..14 (adjacent, but not overlapping at offsets 6 and 13).
        let range = TextRange::new(TextSize::from(6), TextSize::from(13));

        let range_tokens = test.highlight_range(range);

        assert_snapshot!(test.to_snapshot(&range_tokens), @r#""y" @ 7..8: Variable"#);
    }

    #[test]
    fn dotted_module_names() {
        let test = SemanticTokenTest::new(
            "
import os.path
import sys.version_info
from urllib.parse import urlparse
from collections.abc import Mapping
",
        );

        let tokens = test.highlight_file();

        assert_snapshot!(test.to_snapshot(&tokens), @r#"
        "os" @ 8..10: Namespace
        "path" @ 11..15: Namespace
        "sys" @ 23..26: Namespace
        "version_info" @ 27..39: Namespace
        "urllib" @ 45..51: Namespace
        "parse" @ 52..57: Namespace
        "urlparse" @ 65..73: Function
        "collections" @ 79..90: Namespace
        "abc" @ 91..94: Namespace
        "Mapping" @ 102..109: Class
        "#);
    }

    #[test]
    fn module_type_classification() {
        let test = SemanticTokenTest::new(
            "
import os
import sys
from collections import defaultdict

# os and sys should be classified as namespace/module types
x = os
y = sys
",
        );

        let tokens = test.highlight_file();

        assert_snapshot!(test.to_snapshot(&tokens), @r#"
        "os" @ 8..10: Namespace
        "sys" @ 18..21: Namespace
        "collections" @ 27..38: Namespace
        "defaultdict" @ 46..57: Class
        "x" @ 119..120: Namespace
        "os" @ 123..125: Namespace
        "y" @ 126..127: Namespace
        "sys" @ 130..133: Namespace
        "#);
    }

    #[test]
    fn import_classification() {
        let test = SemanticTokenTest::new(
            "
from os import path
from collections import defaultdict, OrderedDict, Counter
from typing import List, Dict, Optional
from mymodule import CONSTANT, my_function, MyClass
",
        );

        let tokens = test.highlight_file();

        assert_snapshot!(test.to_snapshot(&tokens), @r#"
        "os" @ 6..8: Namespace
        "path" @ 16..20: Namespace
        "collections" @ 26..37: Namespace
        "defaultdict" @ 45..56: Class
        "OrderedDict" @ 58..69: Class
        "Counter" @ 71..78: Class
        "typing" @ 84..90: Namespace
        "List" @ 98..102: Class
        "Dict" @ 104..108: Class
        "Optional" @ 110..118: Class
        "mymodule" @ 124..132: Namespace
        "#);
    }

    #[test]
    fn str_annotation() {
        let test = SemanticTokenTest::new(
            r#"
x: int = 1
y: "int" = 1
z = "int"
w1: "int | str" = "hello"
w2: "int | sr" = "hello"
w3: "int | " = "hello"
w4: "float"
w5: "float
"#,
        );

        let tokens = test.highlight_file();

        assert_snapshot!(test.to_snapshot(&tokens), @r#"
        "x" @ 1..2: Variable
        "int" @ 4..7: Class [defaultLibrary, builtin]
        "y" @ 12..13: Variable
        "int" @ 16..19: Class [defaultLibrary, builtin]
        "z" @ 25..26: Variable
        "w1" @ 35..37: Variable
        "int" @ 40..43: Class [defaultLibrary, builtin]
        "str" @ 46..49: Class [defaultLibrary, builtin]
        "w2" @ 61..63: Variable
        "int" @ 66..69: Class [defaultLibrary, builtin]
        "w3" @ 86..88: Variable
        "w4" @ 109..111: Variable
        "float" @ 114..119: Class [defaultLibrary, builtin]
        "w5" @ 121..123: Variable
        "float" @ 126..131: Class [defaultLibrary, builtin]
        "#);
    }

    #[test]
    fn str_annotation_nested() {
        let test = SemanticTokenTest::new(
            r#"
x: int
y: "int"
z: "'int'"
w: """'"int"'"""

a: list[int | str] | None
b: list["int | str"] | None
c: "list[int | str] | None"
d: "list[int | str]" | "None"
e: 'list["int | str"] | "None"'
f: """'list["int | str"]' | 'None'"""
"#,
        );

        let tokens = test.highlight_file();

        assert_snapshot!(test.to_snapshot(&tokens), @r#"
        "x" @ 1..2: Variable
        "int" @ 4..7: Class [defaultLibrary, builtin]
        "y" @ 8..9: Variable
        "int" @ 12..15: Class [defaultLibrary, builtin]
        "z" @ 17..18: Variable
        "int" @ 22..25: Class [defaultLibrary, builtin]
        "w" @ 28..29: Variable
        "a" @ 46..47: Variable
        "list" @ 49..53: Class [defaultLibrary, builtin]
        "int" @ 54..57: Class [defaultLibrary, builtin]
        "str" @ 60..63: Class [defaultLibrary, builtin]
        "b" @ 72..73: Variable
        "list" @ 75..79: Class [defaultLibrary, builtin]
        "int" @ 81..84: Class [defaultLibrary, builtin]
        "str" @ 87..90: Class [defaultLibrary, builtin]
        "c" @ 100..101: Variable
        "list" @ 104..108: Class [defaultLibrary, builtin]
        "int" @ 109..112: Class [defaultLibrary, builtin]
        "str" @ 115..118: Class [defaultLibrary, builtin]
        "d" @ 128..129: Variable
        "list" @ 132..136: Class [defaultLibrary, builtin]
        "int" @ 137..140: Class [defaultLibrary, builtin]
        "str" @ 143..146: Class [defaultLibrary, builtin]
        "e" @ 158..159: Variable
        "list" @ 162..166: Class [defaultLibrary, builtin]
        "int" @ 168..171: Class [defaultLibrary, builtin]
        "str" @ 174..177: Class [defaultLibrary, builtin]
        "f" @ 190..191: Variable
        "list" @ 197..201: Class [defaultLibrary, builtin]
        "#);
    }

    #[test]
    fn attribute_classification() {
        let test = SemanticTokenTest::new(
            "
import os
import sys
from collections import defaultdict

class MyClass:
    CONSTANT = 42

    def method(self):
        return \"hello\"

    @property
    def prop(self):
        return self.CONSTANT

obj = MyClass()

# Test various attribute accesses
x = os.path              # path should be namespace (module)
y = obj.method           # method should be method (bound method)
z = obj.CONSTANT         # CONSTANT should be variable with readonly modifier
w = obj.prop             # prop should be property
v = MyClass.method       # method should be method (function)
u = MyClass.__name__     # __name__ should resolve on the class object
t = MyClass.prop          # prop should be property on the class itself
",
        );

        let tokens = test.highlight_file();

        assert_snapshot!(test.to_snapshot(&tokens), @r#"
        "os" @ 8..10: Namespace
        "sys" @ 18..21: Namespace
        "collections" @ 27..38: Namespace
        "defaultdict" @ 46..57: Class
        "MyClass" @ 65..72: Class [declaration]
        "CONSTANT" @ 78..86: Property [readonly, static, classMember]
        "method" @ 101..107: Method [declaration, classMember]
        "self" @ 108..112: SelfParameter [declaration, parameter]
        "@" @ 143..144: Decorator
        "property" @ 144..152: Decorator
        "prop" @ 161..165: Property [declaration, readonly, classMember]
        "self" @ 166..170: SelfParameter [declaration, parameter]
        "self" @ 188..192: SelfParameter [parameter]
        "CONSTANT" @ 193..201: Property [readonly, static, classMember]
        "obj" @ 203..206: Variable
        "MyClass" @ 209..216: Class
        "x" @ 254..255: Namespace
        "os" @ 258..260: Namespace
        "path" @ 261..265: Namespace
        "y" @ 315..316: Function
        "obj" @ 319..322: Variable
        "method" @ 323..329: Method [classMember]
        "z" @ 381..382: Variable
        "obj" @ 385..388: Variable
        "CONSTANT" @ 389..397: Property [readonly, static, classMember]
        "w" @ 459..460: Variable
        "obj" @ 463..466: Variable
        "prop" @ 467..471: Property [readonly, classMember]
        "v" @ 510..511: Function
        "MyClass" @ 514..521: Class
        "method" @ 522..528: Method [classMember]
        "u" @ 572..573: Variable
        "MyClass" @ 576..583: Class
        "__name__" @ 584..592: Property [static, classMember]
        "t" @ 643..644: Variable
        "MyClass" @ 647..654: Class
        "prop" @ 655..659: Property [readonly, classMember]
        "#);
    }

    #[test]
    fn decorated_method_attribute_classification() {
        let test = SemanticTokenTest::new(
            r#"
from collections.abc import Callable
from typing import Self

def decorate[**P, R](function: Callable[P, R]) -> Callable[P, R]:
    return function

class C:
    callback: Callable[[], None] = lambda: None

    @decorate
    def instance_method(self) -> Self:
        return self

    def plain_method(self) -> Self:
        return self

    @classmethod
    @decorate
    def class_method(cls) -> None: ...

    @staticmethod
    @decorate
    def static_method() -> None: ...

c = C()
c.instance_method().plain_method().instance_method()
c.class_method()
c.static_method()
c.callback()
"#,
        );

        let tokens = test.highlight_file();

        assert_snapshot!(test.to_snapshot(&tokens), @r#"
        "collections" @ 6..17: Namespace
        "abc" @ 18..21: Namespace
        "Callable" @ 29..37: Class
        "typing" @ 43..49: Namespace
        "Self" @ 57..61: Class
        "decorate" @ 67..75: Function [declaration]
        "P" @ 78..79: TypeParameter
        "R" @ 81..82: TypeParameter
        "function" @ 84..92: Function [declaration, parameter]
        "Callable" @ 94..102: Class
        "P" @ 103..104: TypeParameter
        "R" @ 106..107: TypeParameter
        "Callable" @ 113..121: Class
        "P" @ 122..123: TypeParameter
        "R" @ 125..126: TypeParameter
        "function" @ 140..148: Function [parameter]
        "C" @ 156..157: Class [declaration]
        "callback" @ 163..171: Function [static, classMember]
        "Callable" @ 173..181: Class
        "@" @ 212..213: Decorator
        "decorate" @ 213..221: Decorator
        "instance_method" @ 230..245: Method [declaration, classMember]
        "self" @ 246..250: SelfParameter [declaration, parameter]
        "Self" @ 255..259: Type
        "self" @ 276..280: SelfParameter [parameter]
        "plain_method" @ 290..302: Method [declaration, classMember]
        "self" @ 303..307: SelfParameter [declaration, parameter]
        "Self" @ 312..316: Type
        "self" @ 333..337: SelfParameter [parameter]
        "@" @ 343..344: Decorator
        "classmethod" @ 344..355: Decorator
        "@" @ 360..361: Decorator
        "decorate" @ 361..369: Decorator
        "class_method" @ 378..390: Method [declaration, classMember]
        "cls" @ 391..394: ClsParameter [declaration, parameter]
        "@" @ 414..415: Decorator
        "staticmethod" @ 415..427: Decorator
        "@" @ 432..433: Decorator
        "decorate" @ 433..441: Decorator
        "static_method" @ 450..463: Method [declaration, static, classMember]
        "c" @ 480..481: Variable
        "C" @ 484..485: Class
        "c" @ 488..489: Variable
        "instance_method" @ 490..505: Method [classMember]
        "plain_method" @ 508..520: Method [classMember]
        "instance_method" @ 523..538: Method [classMember]
        "c" @ 541..542: Variable
        "class_method" @ 543..555: Method [classMember]
        "c" @ 558..559: Variable
        "static_method" @ 560..573: Method [static, classMember]
        "c" @ 576..577: Variable
        "callback" @ 578..586: Function [static, classMember]
        "#);
    }

    #[test]
    fn property_with_return_annotation() {
        let test = SemanticTokenTest::new(
            "
class Foo:
    @property
    def prop(self) -> int:
        return 4

foo = Foo()
w = foo.prop
x = Foo.prop
",
        );

        let tokens = test.highlight_file();

        assert_snapshot!(test.to_snapshot(&tokens), @r#"
        "Foo" @ 7..10: Class [declaration]
        "@" @ 16..17: Decorator
        "property" @ 17..25: Decorator
        "prop" @ 34..38: Property [declaration, readonly, classMember]
        "self" @ 39..43: SelfParameter [declaration, parameter]
        "int" @ 48..51: Class [defaultLibrary, builtin]
        "foo" @ 71..74: Variable
        "Foo" @ 77..80: Class
        "w" @ 83..84: Variable
        "foo" @ 87..90: Variable
        "prop" @ 91..95: Property [readonly, classMember]
        "x" @ 96..97: Variable
        "Foo" @ 100..103: Class
        "prop" @ 104..108: Property [readonly, classMember]
        "#);
    }

    #[test]
    fn property_readonly_modifier() {
        // Verify that the readonly modifier is set for getter-only properties
        // and NOT set for properties that also have a setter.
        let test = SemanticTokenTest::new(
            "
class Config:
    @property
    def read_only(self) -> str:
        return 'value'

    @property
    def read_write(self) -> int:
        return self._x

    @read_write.setter
    def read_write(self, value: int) -> None:
        self._x = value

cfg = Config()
a = cfg.read_only
b = cfg.read_write
",
        );

        let tokens = test.highlight_file();

        assert_snapshot!(test.to_snapshot(&tokens), @r#"
        "Config" @ 7..13: Class [declaration]
        "@" @ 19..20: Decorator
        "property" @ 20..28: Decorator
        "read_only" @ 37..46: Property [declaration, readonly, classMember]
        "self" @ 47..51: SelfParameter [declaration, parameter]
        "str" @ 56..59: Class [defaultLibrary, builtin]
        "@" @ 89..90: Decorator
        "property" @ 90..98: Decorator
        "read_write" @ 107..117: Property [declaration, classMember]
        "self" @ 118..122: SelfParameter [declaration, parameter]
        "int" @ 127..130: Class [defaultLibrary, builtin]
        "self" @ 147..151: SelfParameter [parameter]
        "_x" @ 152..154: Property [classMember]
        "@" @ 160..161: Decorator
        "read_write" @ 161..171: Property [classMember]
        "setter" @ 172..178: Method [classMember]
        "read_write" @ 187..197: Property [declaration, classMember]
        "self" @ 198..202: SelfParameter [declaration, parameter]
        "value" @ 204..209: Parameter [declaration, parameter]
        "int" @ 211..214: Class [defaultLibrary, builtin]
        "self" @ 233..237: SelfParameter [parameter]
        "_x" @ 238..240: Property [classMember]
        "value" @ 243..248: Parameter [parameter]
        "cfg" @ 250..253: Variable
        "Config" @ 256..262: Class
        "a" @ 265..266: Variable
        "cfg" @ 269..272: Variable
        "read_only" @ 273..282: Property [readonly, classMember]
        "b" @ 283..284: Variable
        "cfg" @ 287..290: Variable
        "read_write" @ 291..301: Property [classMember]
        "#);
    }

    #[test]
    fn property_union_with_non_property_falls_back() {
        let test = SemanticTokenTest::new(
            "
class WithProperty:
    @property
    def value(self) -> int:
        return 1

class WithAttribute:
    value = 2

def f(obj: WithProperty | WithAttribute):
    return obj.value
",
        );

        let tokens = test.highlight_file();

        assert_snapshot!(test.to_snapshot(&tokens), @r#"
        "WithProperty" @ 7..19: Class [declaration]
        "@" @ 25..26: Decorator
        "property" @ 26..34: Decorator
        "value" @ 43..48: Property [declaration, readonly, classMember]
        "self" @ 49..53: SelfParameter [declaration, parameter]
        "int" @ 58..61: Class [defaultLibrary, builtin]
        "WithAttribute" @ 87..100: Class [declaration]
        "value" @ 106..111: Property [static, classMember]
        "f" @ 121..122: Function [declaration]
        "obj" @ 123..126: Parameter [declaration, parameter]
        "WithProperty" @ 128..140: Class
        "WithAttribute" @ 143..156: Class
        "obj" @ 170..173: Parameter [parameter]
        "value" @ 174..179: Property [classMember]
        "#);
    }

    #[test]
    fn property_union_readonly_only_if_all_variants_are_readonly() {
        let test = SemanticTokenTest::new(
            "
from random import random

class ReadOnly:
    @property
    def value(self) -> int:
        return 1

class ReadWrite:
    @property
    def value(self) -> int:
        return self._value

    @value.setter
    def value(self, new_value: int) -> None:
        self._value = new_value

obj = ReadOnly() if random() else ReadWrite()
x = obj.value
",
        );

        let tokens = test.highlight_file();

        assert_snapshot!(test.to_snapshot(&tokens), @r#"
        "random" @ 6..12: Namespace
        "random" @ 20..26: Function
        "ReadOnly" @ 34..42: Class [declaration]
        "@" @ 48..49: Decorator
        "property" @ 49..57: Decorator
        "value" @ 66..71: Property [declaration, readonly, classMember]
        "self" @ 72..76: SelfParameter [declaration, parameter]
        "int" @ 81..84: Class [defaultLibrary, builtin]
        "ReadWrite" @ 110..119: Class [declaration]
        "@" @ 125..126: Decorator
        "property" @ 126..134: Decorator
        "value" @ 143..148: Property [declaration, classMember]
        "self" @ 149..153: SelfParameter [declaration, parameter]
        "int" @ 158..161: Class [defaultLibrary, builtin]
        "self" @ 178..182: SelfParameter [parameter]
        "_value" @ 183..189: Property [classMember]
        "@" @ 195..196: Decorator
        "value" @ 196..201: Property [classMember]
        "setter" @ 202..208: Method [classMember]
        "value" @ 217..222: Property [declaration, classMember]
        "self" @ 223..227: SelfParameter [declaration, parameter]
        "new_value" @ 229..238: Parameter [declaration, parameter]
        "int" @ 240..243: Class [defaultLibrary, builtin]
        "self" @ 262..266: SelfParameter [parameter]
        "_value" @ 267..273: Property [classMember]
        "new_value" @ 276..285: Parameter [parameter]
        "obj" @ 287..290: Variable
        "ReadOnly" @ 293..301: Class
        "random" @ 307..313: Function
        "ReadWrite" @ 321..330: Class
        "x" @ 333..334: Variable
        "obj" @ 337..340: Variable
        "value" @ 341..346: Property [classMember]
        "#);
    }

    #[test]
    fn attribute_fallback_classification() {
        let test = SemanticTokenTest::new(
            "
class MyClass:
    some_attr = \"value\"

obj = MyClass()
# Test attribute that might not have detailed semantic info
x = obj.some_attr        # Should fall back to variable, not property
y = obj.unknown_attr     # Should fall back to variable
",
        );

        let tokens = test.highlight_file();

        assert_snapshot!(test.to_snapshot(&tokens), @r#"
        "MyClass" @ 7..14: Class [declaration]
        "some_attr" @ 20..29: Property [static, classMember]
        "obj" @ 41..44: Variable
        "MyClass" @ 47..54: Class
        "x" @ 117..118: Variable
        "obj" @ 121..124: Variable
        "some_attr" @ 125..134: Property [static, classMember]
        "y" @ 187..188: Variable
        "obj" @ 191..194: Variable
        "#);
    }

    #[test]
    fn attribute_on_union_1() {
        let test = SemanticTokenTest::new(
            "
from random import random

class Foo:
    CONSTANT = 42

    def method(self):
        return \"hello\"

    @property
    def prop(self) -> str:
        return \"hello\"

class Bar:
    CONSTANT = 24

    def method(self, x: int = 1) -> int:
        return 42

    @property
    def prop(self) -> int:
        return self.CONSTANT


foobar = Foo() if random() else Bar()
y = foobar.method                                # method should be method (bound method)
z = foobar.CONSTANT                              # CONSTANT should be variable with readonly modifier
w = foobar.prop                                  # prop should be property
foobar_cls = Foo if random() else Bar
v = foobar_cls.method                            # method should be method (function)
x = foobar_cls.prop                              # prop should be property
",
        );

        let tokens = test.highlight_file();

        assert_snapshot!(test.to_snapshot(&tokens), @r#"
        "random" @ 6..12: Namespace
        "random" @ 20..26: Function
        "Foo" @ 34..37: Class [declaration]
        "CONSTANT" @ 43..51: Property [readonly, static, classMember]
        "method" @ 66..72: Method [declaration, classMember]
        "self" @ 73..77: SelfParameter [declaration, parameter]
        "@" @ 108..109: Decorator
        "property" @ 109..117: Decorator
        "prop" @ 126..130: Property [declaration, readonly, classMember]
        "self" @ 131..135: SelfParameter [declaration, parameter]
        "str" @ 140..143: Class [defaultLibrary, builtin]
        "Bar" @ 175..178: Class [declaration]
        "CONSTANT" @ 184..192: Property [readonly, static, classMember]
        "method" @ 207..213: Method [declaration, classMember]
        "self" @ 214..218: SelfParameter [declaration, parameter]
        "x" @ 220..221: Parameter [declaration, parameter]
        "int" @ 223..226: Class [defaultLibrary, builtin]
        "int" @ 235..238: Class [defaultLibrary, builtin]
        "@" @ 263..264: Decorator
        "property" @ 264..272: Decorator
        "prop" @ 281..285: Property [declaration, readonly, classMember]
        "self" @ 286..290: SelfParameter [declaration, parameter]
        "int" @ 295..298: Class [defaultLibrary, builtin]
        "self" @ 315..319: SelfParameter [parameter]
        "CONSTANT" @ 320..328: Property [readonly, static, classMember]
        "foobar" @ 331..337: Variable
        "Foo" @ 340..343: Class
        "random" @ 349..355: Function
        "Bar" @ 363..366: Class
        "y" @ 369..370: Function
        "foobar" @ 373..379: Variable
        "method" @ 380..386: Method [classMember]
        "z" @ 459..460: Variable
        "foobar" @ 463..469: Variable
        "CONSTANT" @ 470..478: Property [readonly, static, classMember]
        "w" @ 561..562: Variable
        "foobar" @ 565..571: Variable
        "prop" @ 572..576: Property [readonly, classMember]
        "foobar_cls" @ 636..646: Type
        "Foo" @ 649..652: Class
        "random" @ 656..662: Function
        "Bar" @ 670..673: Class
        "v" @ 674..675: Function
        "foobar_cls" @ 678..688: Type
        "method" @ 689..695: Method [classMember]
        "x" @ 760..761: Variable
        "foobar_cls" @ 764..774: Type
        "prop" @ 775..779: Property [readonly, classMember]
        "#);
    }

    #[test]
    fn attribute_on_union_2() {
        let test = SemanticTokenTest::new(
            "
from random import random

# There is also this way to create union types:
class Baz:
    if random():
        CONSTANT = 42

        def method(self) -> int:
            return 42

        @property
        def prop(self) -> int:
            return 42
    else:
        CONSTANT = \"hello\"

        def method(self) -> str:
            return \"hello\"

        @property
        def prop(self) -> str:
            return \"hello\"

baz = Baz()
s = baz.method      # method should be bound method
t = baz.CONSTANT    # CONSTANT should be variable with readonly
r = baz.prop        # prop should be property
q = Baz.prop        # prop should be property on the class as well
",
        );

        let tokens = test.highlight_file();

        assert_snapshot!(test.to_snapshot(&tokens), @r#"
        "random" @ 6..12: Namespace
        "random" @ 20..26: Function
        "Baz" @ 82..85: Class [declaration]
        "random" @ 94..100: Function
        "CONSTANT" @ 112..120: Property [readonly, static, classMember]
        "method" @ 139..145: Method [declaration, classMember]
        "self" @ 146..150: SelfParameter [declaration, parameter]
        "int" @ 155..158: Class [defaultLibrary, builtin]
        "@" @ 191..192: Decorator
        "property" @ 192..200: Decorator
        "prop" @ 213..217: Property [declaration, readonly, classMember]
        "self" @ 218..222: SelfParameter [declaration, parameter]
        "int" @ 227..230: Class [defaultLibrary, builtin]
        "CONSTANT" @ 272..280: Property [readonly, static, classMember]
        "method" @ 304..310: Method [declaration, classMember]
        "self" @ 311..315: SelfParameter [declaration, parameter]
        "str" @ 320..323: Class [defaultLibrary, builtin]
        "@" @ 361..362: Decorator
        "property" @ 362..370: Decorator
        "prop" @ 383..387: Property [declaration, readonly, classMember]
        "self" @ 388..392: SelfParameter [declaration, parameter]
        "str" @ 397..400: Class [defaultLibrary, builtin]
        "baz" @ 430..433: Variable
        "Baz" @ 436..439: Class
        "s" @ 442..443: Function
        "baz" @ 446..449: Variable
        "method" @ 450..456: Method [classMember]
        "t" @ 494..495: Variable
        "baz" @ 498..501: Variable
        "CONSTANT" @ 502..510: Property [readonly, static, classMember]
        "r" @ 558..559: Variable
        "baz" @ 562..565: Variable
        "prop" @ 566..570: Property [readonly, classMember]
        "q" @ 604..605: Variable
        "Baz" @ 608..611: Class
        "prop" @ 612..616: Property [readonly, classMember]
        "#);
    }

    #[test]
    fn attribute_on_union_3() {
        // This is a test where the unions are not actually composed of the same elements,
        // so the regular fallback logic should apply.
        let test = SemanticTokenTest::new(
            "
from random import random

class Baz:
    if random():
        CONSTANT = 42

        def method(self) -> int:
            return 42

        @property
        def prop(self) -> int:
            return 42
    else:
        def CONSTANT(self):
            return \"hello\"

        @property
        def method(self) -> str:
            return \"hello\"

        prop: str = \"hello\"

baz = Baz()
s = baz.method
t = baz.CONSTANT
r = baz.prop
q = Baz.prop
",
        );

        let tokens = test.highlight_file();

        assert_snapshot!(test.to_snapshot(&tokens), @r#"
        "random" @ 6..12: Namespace
        "random" @ 20..26: Function
        "Baz" @ 34..37: Class [declaration]
        "random" @ 46..52: Function
        "CONSTANT" @ 64..72: Property [readonly, static, classMember]
        "method" @ 91..97: Method [declaration, classMember]
        "self" @ 98..102: SelfParameter [declaration, parameter]
        "int" @ 107..110: Class [defaultLibrary, builtin]
        "@" @ 143..144: Decorator
        "property" @ 144..152: Decorator
        "prop" @ 165..169: Property [declaration, classMember]
        "self" @ 170..174: SelfParameter [declaration, parameter]
        "int" @ 179..182: Class [defaultLibrary, builtin]
        "CONSTANT" @ 228..236: Method [declaration, classMember]
        "self" @ 237..241: SelfParameter [declaration, parameter]
        "@" @ 280..281: Decorator
        "property" @ 281..289: Decorator
        "method" @ 302..308: Property [declaration, classMember]
        "self" @ 309..313: SelfParameter [declaration, parameter]
        "str" @ 318..321: Class [defaultLibrary, builtin]
        "prop" @ 359..363: Property [classMember]
        "str" @ 365..368: Class [defaultLibrary, builtin]
        "baz" @ 380..383: Variable
        "Baz" @ 386..389: Class
        "s" @ 392..393: Variable
        "baz" @ 396..399: Variable
        "method" @ 400..406: Method [classMember]
        "t" @ 407..408: Variable
        "baz" @ 411..414: Variable
        "CONSTANT" @ 415..423: Method [classMember]
        "r" @ 424..425: Variable
        "baz" @ 428..431: Variable
        "prop" @ 432..436: Property [classMember]
        "q" @ 437..438: Variable
        "Baz" @ 441..444: Class
        "prop" @ 445..449: Property [classMember]
        "#);
    }

    #[test]
    fn constant_name_detection() {
        let test = SemanticTokenTest::new(
            "
class MyClass:
    UPPER_CASE = 42
    lower_case = 24
    MixedCase = 12
    A = 1

obj = MyClass()
x = obj.UPPER_CASE    # Should have readonly modifier
y = obj.lower_case    # Should not have readonly modifier
z = obj.MixedCase     # Should not have readonly modifier
w = obj.A             # Should not have readonly modifier (length == 1)
",
        );

        let tokens = test.highlight_file();

        assert_snapshot!(test.to_snapshot(&tokens), @r#"
        "MyClass" @ 7..14: Class [declaration]
        "UPPER_CASE" @ 20..30: Property [readonly, static, classMember]
        "lower_case" @ 40..50: Property [static, classMember]
        "MixedCase" @ 60..69: Property [static, classMember]
        "A" @ 79..80: Property [readonly, static, classMember]
        "obj" @ 86..89: Variable
        "MyClass" @ 92..99: Class
        "x" @ 102..103: Variable
        "obj" @ 106..109: Variable
        "UPPER_CASE" @ 110..120: Property [readonly, static, classMember]
        "y" @ 156..157: Variable
        "obj" @ 160..163: Variable
        "lower_case" @ 164..174: Property [static, classMember]
        "z" @ 214..215: Variable
        "obj" @ 218..221: Variable
        "MixedCase" @ 222..231: Property [static, classMember]
        "w" @ 272..273: Variable
        "obj" @ 276..279: Variable
        "A" @ 280..281: Property [readonly, static, classMember]
        "#);
    }

    #[test]
    fn type_annotations() {
        let test = SemanticTokenTest::new(
            r#"
from typing import List, Optional

def function_with_annotations(param1: int, param2: str) -> Optional[List[str]]:
    pass

x: int = 42
y: Optional[str] = None
"#,
        );

        let tokens = test.highlight_file();

        assert_snapshot!(test.to_snapshot(&tokens), @r#"
        "typing" @ 6..12: Namespace
        "List" @ 20..24: Class
        "Optional" @ 26..34: Class
        "function_with_annotations" @ 40..65: Function [declaration]
        "param1" @ 66..72: Parameter [declaration, parameter]
        "int" @ 74..77: Class [defaultLibrary, builtin]
        "param2" @ 79..85: Parameter [declaration, parameter]
        "str" @ 87..90: Class [defaultLibrary, builtin]
        "Optional" @ 95..103: Class
        "List" @ 104..108: Class
        "str" @ 109..112: Class [defaultLibrary, builtin]
        "x" @ 126..127: Variable
        "int" @ 129..132: Class [defaultLibrary, builtin]
        "y" @ 138..139: Variable
        "Optional" @ 141..149: Class
        "str" @ 150..153: Class [defaultLibrary, builtin]
        "#);
    }

    #[test]
    fn type_alias_values_use_type_form_highlighting() {
        let test = SemanticTokenTest::new(
            r#"
from typing import IO, TypeAlias

def takes_file(x: IO[str]) -> None: ...

type NewStyle = IO[str]
LegacyStyle: TypeAlias = IO[str]
"#,
        );

        let tokens = test.highlight_file();
        let source = ruff_db::source::source_text(&test.db, test.file);
        let io_ranges: Vec<_> = source
            .match_indices("IO")
            .skip(1)
            .map(|(offset, _)| {
                TextRange::at(
                    TextSize::from(
                        u32::try_from(offset).expect("source offset to fit into TextSize"),
                    ),
                    "IO".text_len(),
                )
            })
            .collect();

        assert_eq!(
            io_ranges.len(),
            3,
            "expected annotation and alias RHS `IO` uses"
        );

        for io_range in io_ranges {
            let token = tokens
                .iter()
                .find(|token| token.range == io_range)
                .expect("semantic token for `IO` type-form use");
            assert_eq!(token.token_type, SemanticTokenType::Class);
        }
    }

    #[test]
    fn type_alias_classified_consistently() {
        let test = SemanticTokenTest::new(
            "
from typing import TypeAlias

type NewAlias = int
OldAlias: TypeAlias = int
def f(x: NewAlias, y: OldAlias): ...

marker = TypeAlias
",
        );

        let tokens = test.highlight_file();

        assert_snapshot!(test.to_snapshot(&tokens), @r#"
        "typing" @ 6..12: Namespace
        "TypeAlias" @ 20..29: Class
        "type" @ 31..35: Keyword
        "NewAlias" @ 36..44: Class
        "int" @ 47..50: Class [defaultLibrary, builtin]
        "OldAlias" @ 51..59: Class
        "TypeAlias" @ 61..70: Class
        "int" @ 73..76: Class [defaultLibrary, builtin]
        "f" @ 81..82: Function [declaration]
        "x" @ 83..84: Parameter [declaration, parameter]
        "NewAlias" @ 86..94: Class
        "y" @ 96..97: Parameter [declaration, parameter]
        "OldAlias" @ 99..107: Class
        "marker" @ 115..121: Class
        "TypeAlias" @ 124..133: Class
        "#);
    }

    #[test]
    fn imported_type_aliases_classified_consistently() {
        let mut test = SemanticTokenTest::new(
            "
from aliases import OldAlias
import aliases

aliases.OldAlias
",
        );
        test.db
            .write_file(
                SystemPath::new("src/aliases.py"),
                "
from typing import TypeAlias

OldAlias: TypeAlias = int | str
",
            )
            .expect("writing to the memory file system should succeed");

        let tokens = test.highlight_file();

        assert_snapshot!(test.to_snapshot(&tokens), @r#"
        "aliases" @ 6..13: Namespace
        "OldAlias" @ 21..29: Type
        "aliases" @ 37..44: Namespace
        "aliases" @ 46..53: Namespace
        "OldAlias" @ 54..62: Type
        "#);
    }

    #[test]
    fn generic_class_members_in_annotations() {
        let test = SemanticTokenTest::new(
            r#"
import os
from os import PathLike

x1: os.PathLike
x2: os.PathLike[str]

y1: PathLike
y2: PathLike[str]

z1 = os.PathLike
z2 = os.PathLike[str]
"#,
        );

        let tokens = test.highlight_file();

        assert_snapshot!(test.to_snapshot(&tokens), @r#"
        "os" @ 8..10: Namespace
        "os" @ 16..18: Namespace
        "PathLike" @ 26..34: Class
        "x1" @ 36..38: Variable
        "os" @ 40..42: Namespace
        "PathLike" @ 43..51: Class
        "x2" @ 52..54: Variable
        "os" @ 56..58: Namespace
        "PathLike" @ 59..67: Class
        "str" @ 68..71: Class [defaultLibrary, builtin]
        "y1" @ 74..76: Variable
        "PathLike" @ 78..86: Class
        "y2" @ 87..89: Variable
        "PathLike" @ 91..99: Class
        "str" @ 100..103: Class [defaultLibrary, builtin]
        "z1" @ 106..108: Class
        "os" @ 111..113: Namespace
        "PathLike" @ 114..122: Class
        "z2" @ 123..125: Class
        "os" @ 128..130: Namespace
        "PathLike" @ 131..139: Class
        "str" @ 140..143: Class [defaultLibrary, builtin]
        "#);
    }

    #[test]
    fn generic_class_members_in_cast() {
        let test = SemanticTokenTest::new(
            r#"
import os
import typing
from os import PathLike
from typing import cast

x1 = cast(os.PathLike[str], "")
x2 = cast(PathLike[str], "")
x3 = typing.cast(os.PathLike[str], "")
"#,
        );

        let tokens = test.highlight_file();

        assert_snapshot!(test.to_snapshot(&tokens), @r#"
        "os" @ 8..10: Namespace
        "typing" @ 18..24: Namespace
        "os" @ 30..32: Namespace
        "PathLike" @ 40..48: Class
        "typing" @ 54..60: Namespace
        "cast" @ 68..72: Function
        "x1" @ 74..76: Variable
        "cast" @ 79..83: Function
        "os" @ 84..86: Namespace
        "PathLike" @ 87..95: Class
        "str" @ 96..99: Class [defaultLibrary, builtin]
        "x2" @ 106..108: Variable
        "cast" @ 111..115: Function
        "PathLike" @ 116..124: Class
        "str" @ 125..128: Class [defaultLibrary, builtin]
        "x3" @ 135..137: Variable
        "typing" @ 140..146: Namespace
        "cast" @ 147..151: Function
        "os" @ 152..154: Namespace
        "PathLike" @ 155..163: Class
        "str" @ 164..167: Class [defaultLibrary, builtin]
        "#);
    }

    #[test]
    fn generic_class_members_in_assert_type() {
        let test = SemanticTokenTest::new(
            r#"
import os
import typing
from os import PathLike
from typing import assert_type

x1 = assert_type("", os.PathLike[str])
x2 = assert_type("", PathLike[str])
x3 = typing.assert_type("", os.PathLike[str])
"#,
        );

        let tokens = test.highlight_file();

        assert_snapshot!(test.to_snapshot(&tokens), @r#"
        "os" @ 8..10: Namespace
        "typing" @ 18..24: Namespace
        "os" @ 30..32: Namespace
        "PathLike" @ 40..48: Class
        "typing" @ 54..60: Namespace
        "assert_type" @ 68..79: Function
        "x1" @ 81..83: Variable
        "assert_type" @ 86..97: Function
        "os" @ 102..104: Namespace
        "PathLike" @ 105..113: Class
        "str" @ 114..117: Class [defaultLibrary, builtin]
        "x2" @ 120..122: Variable
        "assert_type" @ 125..136: Function
        "PathLike" @ 141..149: Class
        "str" @ 150..153: Class [defaultLibrary, builtin]
        "x3" @ 156..158: Variable
        "typing" @ 161..167: Namespace
        "assert_type" @ 168..179: Function
        "os" @ 184..186: Namespace
        "PathLike" @ 187..195: Class
        "str" @ 196..199: Class [defaultLibrary, builtin]
        "#);
    }

    #[test]
    fn generic_class_members_in_type_form_keyword_arguments() {
        let test = SemanticTokenTest::new(
            r#"
import os
from os import PathLike
from typing import assert_type, cast

x1 = cast(typ=os.PathLike[str], val="")
x2 = cast(val="", typ=PathLike[str])
x3 = assert_type(type=os.PathLike[str], value="")
x4 = assert_type(value="", type=PathLike[str])
"#,
        );

        let tokens = test.highlight_file();

        assert_snapshot!(test.to_snapshot(&tokens), @r#"
        "os" @ 8..10: Namespace
        "os" @ 16..18: Namespace
        "PathLike" @ 26..34: Class
        "typing" @ 40..46: Namespace
        "assert_type" @ 54..65: Function
        "cast" @ 67..71: Function
        "x1" @ 73..75: Variable
        "cast" @ 78..82: Function
        "typ" @ 83..86: Parameter [parameter]
        "os" @ 87..89: Namespace
        "PathLike" @ 90..98: Class
        "str" @ 99..102: Class [defaultLibrary, builtin]
        "val" @ 105..108: Parameter [parameter]
        "x2" @ 113..115: Variable
        "cast" @ 118..122: Function
        "val" @ 123..126: Parameter [parameter]
        "typ" @ 131..134: Parameter [parameter]
        "PathLike" @ 135..143: Class
        "str" @ 144..147: Class [defaultLibrary, builtin]
        "x3" @ 150..152: Variable
        "assert_type" @ 155..166: Function
        "type" @ 167..171: Class [parameter]
        "os" @ 172..174: Namespace
        "PathLike" @ 175..183: Class
        "str" @ 184..187: Class [defaultLibrary, builtin]
        "value" @ 190..195: Parameter [parameter]
        "x4" @ 200..202: Variable
        "assert_type" @ 205..216: Function
        "value" @ 217..222: Parameter [parameter]
        "type" @ 227..231: Class [parameter]
        "PathLike" @ 232..240: Class
        "str" @ 241..244: Class [defaultLibrary, builtin]
        "#);
    }

    #[test]
    fn semantic_tokens_ignore_failed_bindings_for_type_form_arguments() {
        let test = SemanticTokenTest::new(
            r#"
from typing import cast

flag = bool(input())
def g(x):
    return x

x = ""
f = cast if flag else g
f(int, x)
"#,
        );

        let tokens = test.highlight_file();
        assert_snapshot!(test.to_snapshot(&tokens), @r#"
        "typing" @ 6..12: Namespace
        "cast" @ 20..24: Function
        "flag" @ 26..30: Variable
        "bool" @ 33..37: Class [defaultLibrary, builtin]
        "input" @ 38..43: Function [defaultLibrary, builtin]
        "g" @ 51..52: Function [declaration]
        "x" @ 53..54: Parameter [declaration, parameter]
        "x" @ 68..69: Parameter [parameter]
        "x" @ 71..72: Variable
        "f" @ 78..79: Function
        "cast" @ 82..86: Function
        "flag" @ 90..94: Variable
        "g" @ 100..101: Function
        "f" @ 102..103: Function
        "int" @ 104..107: Class [defaultLibrary, builtin]
        "x" @ 109..110: Variable
        "#);
    }

    #[test]
    fn debug_int_classification() {
        let test = SemanticTokenTest::new(
            "
x: int = 42
",
        );

        let tokens = test.highlight_file();

        assert_snapshot!(test.to_snapshot(&tokens), @r#"
        "x" @ 1..2: Variable
        "int" @ 4..7: Class [defaultLibrary, builtin]
        "#);
    }

    #[test]
    fn debug_user_defined_type_classification() {
        let test = SemanticTokenTest::new(
            "
class MyClass:
    pass

x: MyClass = MyClass()
",
        );

        let tokens = test.highlight_file();

        assert_snapshot!(test.to_snapshot(&tokens), @r#"
        "MyClass" @ 7..14: Class [declaration]
        "x" @ 26..27: Variable
        "MyClass" @ 29..36: Class
        "MyClass" @ 39..46: Class
        "#);
    }

    #[test]
    fn type_annotation_vs_variable_classification() {
        let test = SemanticTokenTest::new(
            "
from typing import List, Optional

class MyClass:
    pass

def test_function(param: int, other: MyClass) -> Optional[List[str]]:
    # Variable assignments - should be Variable tokens
    x: int = 42
    y: MyClass = MyClass()
    z: List[str] = [\"hello\"]

    # Type annotations should be Class tokens:
    # int, MyClass, Optional, List, str
    return None
",
        );

        let tokens = test.highlight_file();

        assert_snapshot!(test.to_snapshot(&tokens), @r#"
        "typing" @ 6..12: Namespace
        "List" @ 20..24: Class
        "Optional" @ 26..34: Class
        "MyClass" @ 42..49: Class [declaration]
        "test_function" @ 65..78: Function [declaration]
        "param" @ 79..84: Parameter [declaration, parameter]
        "int" @ 86..89: Class [defaultLibrary, builtin]
        "other" @ 91..96: Parameter [declaration, parameter]
        "MyClass" @ 98..105: Class
        "Optional" @ 110..118: Class
        "List" @ 119..123: Class
        "str" @ 124..127: Class [defaultLibrary, builtin]
        "x" @ 190..191: Variable
        "int" @ 193..196: Class [defaultLibrary, builtin]
        "y" @ 206..207: Variable
        "MyClass" @ 209..216: Class
        "MyClass" @ 219..226: Class
        "z" @ 233..234: Variable
        "List" @ 236..240: Class
        "str" @ 241..244: Class [defaultLibrary, builtin]
        "#);
    }

    #[test]
    fn protocol_types_in_annotations() {
        let test = SemanticTokenTest::new(
            "
from typing import Protocol

class MyProtocol(Protocol):
    def method(self) -> int: ...

def test_function(param: MyProtocol) -> None:
    pass
",
        );

        let tokens = test.highlight_file();

        assert_snapshot!(test.to_snapshot(&tokens), @r#"
        "typing" @ 6..12: Namespace
        "Protocol" @ 20..28: Class
        "MyProtocol" @ 36..46: Class [declaration]
        "Protocol" @ 47..55: Class
        "method" @ 66..72: Method [declaration, classMember]
        "self" @ 73..77: SelfParameter [declaration, parameter]
        "int" @ 82..85: Class [defaultLibrary, builtin]
        "test_function" @ 96..109: Function [declaration]
        "param" @ 110..115: Parameter [declaration, parameter]
        "MyProtocol" @ 117..127: Class
        "#);
    }

    #[test]
    fn protocol_type_annotation_vs_value_context() {
        let test = SemanticTokenTest::new(
            "
from typing import Protocol

class MyProtocol(Protocol):
    def method(self) -> int: ...

# Value context - MyProtocol is still a class literal, so should be Class
my_protocol_var = MyProtocol

# Type annotation context - should be Class
def test_function(param: MyProtocol) -> MyProtocol:
    return param
",
        );

        let tokens = test.highlight_file();

        assert_snapshot!(test.to_snapshot(&tokens), @r#"
        "typing" @ 6..12: Namespace
        "Protocol" @ 20..28: Class
        "MyProtocol" @ 36..46: Class [declaration]
        "Protocol" @ 47..55: Class
        "method" @ 66..72: Method [declaration, classMember]
        "self" @ 73..77: SelfParameter [declaration, parameter]
        "int" @ 82..85: Class [defaultLibrary, builtin]
        "my_protocol_var" @ 166..181: Class
        "MyProtocol" @ 184..194: Class
        "test_function" @ 244..257: Function [declaration]
        "param" @ 258..263: Parameter [declaration, parameter]
        "MyProtocol" @ 265..275: Class
        "MyProtocol" @ 280..290: Class
        "param" @ 303..308: Parameter [parameter]
        "#);
    }

    #[test]
    fn type_alias_type_of() {
        let test = SemanticTokenTest::new(
            "
class Test[T]: ...

my_type_alias = Test[str]  # TODO: `my_type_alias` should be classified as a Class

def test_function(param: my_type_alias): ...
",
        );

        let tokens = test.highlight_file();

        assert_snapshot!(test.to_snapshot(&tokens), @r#"
        "Test" @ 7..11: Class [declaration]
        "T" @ 12..13: TypeParameter
        "my_type_alias" @ 21..34: Class
        "Test" @ 37..41: Class
        "str" @ 42..45: Class [defaultLibrary, builtin]
        "test_function" @ 109..122: Function [declaration]
        "param" @ 123..128: Parameter [declaration, parameter]
        "my_type_alias" @ 130..143: Class
        "#);
    }

    #[test]
    fn type_alias_to_generic_alias() {
        let test = SemanticTokenTest::new(
            "
my_type_alias = type[str]

def test_function(param: my_type_alias): ...
",
        );

        let tokens = test.highlight_file();

        assert_snapshot!(test.to_snapshot(&tokens), @r#"
        "my_type_alias" @ 1..14: Class
        "type" @ 17..21: Class [defaultLibrary, builtin]
        "str" @ 22..25: Class [defaultLibrary, builtin]
        "test_function" @ 32..45: Function [declaration]
        "param" @ 46..51: Class [declaration, parameter]
        "my_type_alias" @ 53..66: Class
        "#);
    }

    #[test]
    fn type_parameters_pep695() {
        let test = SemanticTokenTest::new(
            "
# Test Python 3.12 PEP 695 type parameter syntax

# Generic function with TypeVar
def func[T](x: T) -> T:
    return x

# Generic function with TypeVarTuple
def func_tuple[*Ts](args: tuple[*Ts]) -> tuple[*Ts]:
    return args

# Generic function with ParamSpec
def func_paramspec[**P](func: Callable[P, int]) -> Callable[P, str]:
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> str:
        return str(func(*args, **kwargs))
    return wrapper

# Generic class with multiple type parameters
class Container[T, U]:
    def __init__(self, value1: T, value2: U):
        self.value1: T = value1
        self.value2: U = value2

    def get_first(self) -> T:
        return self.value1

    def get_second(self) -> U:
        return self.value2

# Generic class with bounds and defaults
class BoundedContainer[T: int, U = str]:
    def process(self, x: T, y: U) -> tuple[T, U]:
        return (x, y)
",
        );

        let tokens = test.highlight_file();

        assert_snapshot!(test.to_snapshot(&tokens), @r#"
        "func" @ 87..91: Function [declaration]
        "T" @ 92..93: TypeParameter
        "x" @ 95..96: Parameter [declaration, parameter]
        "T" @ 98..99: TypeParameter
        "T" @ 104..105: TypeParameter
        "x" @ 118..119: Parameter [parameter]
        "func_tuple" @ 162..172: Function [declaration]
        "Ts" @ 174..176: TypeParameter
        "args" @ 178..182: Parameter [declaration, parameter]
        "tuple" @ 184..189: Class [defaultLibrary, builtin]
        "Ts" @ 191..193: TypeParameter
        "tuple" @ 199..204: Class [defaultLibrary, builtin]
        "Ts" @ 206..208: TypeParameter
        "args" @ 222..226: Parameter [parameter]
        "func_paramspec" @ 266..280: Function [declaration]
        "P" @ 283..284: TypeParameter
        "func" @ 286..290: Parameter [declaration, parameter]
        "P" @ 301..302: TypeParameter
        "int" @ 304..307: Class [defaultLibrary, builtin]
        "P" @ 322..323: TypeParameter
        "str" @ 325..328: Class [defaultLibrary, builtin]
        "wrapper" @ 339..346: Function [declaration]
        "args" @ 348..352: Parameter [declaration, parameter]
        "P" @ 354..355: TypeParameter
        "args" @ 356..360: TypeParameter
        "kwargs" @ 364..370: Parameter [declaration, parameter]
        "P" @ 372..373: TypeParameter
        "kwargs" @ 374..380: TypeParameter
        "str" @ 385..388: Class [defaultLibrary, builtin]
        "str" @ 405..408: Class [defaultLibrary, builtin]
        "func" @ 409..413: Parameter [parameter]
        "args" @ 415..419: Parameter [parameter]
        "kwargs" @ 423..429: Parameter [parameter]
        "wrapper" @ 443..450: Function
        "Container" @ 504..513: Class [declaration]
        "T" @ 514..515: TypeParameter
        "U" @ 517..518: TypeParameter
        "__init__" @ 529..537: Method [declaration, classMember]
        "self" @ 538..542: SelfParameter [declaration, parameter]
        "value1" @ 544..550: Parameter [declaration, parameter]
        "T" @ 552..553: TypeParameter
        "value2" @ 555..561: Parameter [declaration, parameter]
        "U" @ 563..564: TypeParameter
        "self" @ 575..579: SelfParameter [parameter]
        "value1" @ 580..586: Property [classMember]
        "T" @ 588..589: TypeParameter
        "value1" @ 592..598: Parameter [parameter]
        "self" @ 607..611: SelfParameter [parameter]
        "value2" @ 612..618: Property [classMember]
        "U" @ 620..621: TypeParameter
        "value2" @ 624..630: Parameter [parameter]
        "get_first" @ 640..649: Method [declaration, classMember]
        "self" @ 650..654: SelfParameter [declaration, parameter]
        "T" @ 659..660: TypeParameter
        "self" @ 677..681: SelfParameter [parameter]
        "value1" @ 682..688: Property [classMember]
        "get_second" @ 698..708: Method [declaration, classMember]
        "self" @ 709..713: SelfParameter [declaration, parameter]
        "U" @ 718..719: TypeParameter
        "self" @ 736..740: SelfParameter [parameter]
        "value2" @ 741..747: Property [classMember]
        "BoundedContainer" @ 796..812: Class [declaration]
        "T" @ 813..814: TypeParameter
        "int" @ 816..819: Class [defaultLibrary, builtin]
        "U" @ 821..822: TypeParameter
        "str" @ 825..828: Class [defaultLibrary, builtin]
        "process" @ 839..846: Method [declaration, classMember]
        "self" @ 847..851: SelfParameter [declaration, parameter]
        "x" @ 853..854: Parameter [declaration, parameter]
        "T" @ 856..857: TypeParameter
        "y" @ 859..860: Parameter [declaration, parameter]
        "U" @ 862..863: TypeParameter
        "tuple" @ 868..873: Class [defaultLibrary, builtin]
        "T" @ 874..875: TypeParameter
        "U" @ 877..878: TypeParameter
        "x" @ 897..898: Parameter [parameter]
        "y" @ 900..901: Parameter [parameter]
        "#);
    }

    #[test]
    fn type_parameters_usage_in_function_body() {
        let test = SemanticTokenTest::new(
            "
def generic_function[T](value: T) -> T:
    # Type parameter T should be recognized here too
    result: T = value
    temp = result  # This could potentially be T as well
    return result
",
        );

        let tokens = test.highlight_file();

        assert_snapshot!(test.to_snapshot(&tokens), @r#"
        "generic_function" @ 5..21: Function [declaration]
        "T" @ 22..23: TypeParameter
        "value" @ 25..30: Parameter [declaration, parameter]
        "T" @ 32..33: TypeParameter
        "T" @ 38..39: TypeParameter
        "result" @ 98..104: Variable
        "T" @ 106..107: TypeParameter
        "value" @ 110..115: Parameter [parameter]
        "temp" @ 120..124: Variable
        "result" @ 127..133: Variable
        "result" @ 184..190: Variable
        "#);
    }

    #[test]
    fn decorator_classification() {
        let test = SemanticTokenTest::new(
            r#"
class App:
    def route(self, path):
        pass

app = App()

@staticmethod
@property
@app.route("/path")
def my_function():
    pass

@dataclass
class MyClass:
    pass
"#,
        );

        let tokens = test.highlight_file();

        assert_snapshot!(test.to_snapshot(&tokens), @r#"
        "App" @ 7..10: Class [declaration]
        "route" @ 20..25: Method [declaration, classMember]
        "self" @ 26..30: SelfParameter [declaration, parameter]
        "path" @ 32..36: Parameter [declaration, parameter]
        "app" @ 53..56: Variable
        "App" @ 59..62: Class
        "@" @ 66..67: Decorator
        "staticmethod" @ 67..79: Decorator
        "@" @ 80..81: Decorator
        "property" @ 81..89: Decorator
        "@" @ 90..91: Decorator
        "app" @ 91..94: Variable
        "route" @ 95..100: Method [classMember]
        "my_function" @ 114..125: Function [declaration]
        "@" @ 139..140: Decorator
        "dataclass" @ 140..149: Decorator
        "MyClass" @ 156..163: Class [declaration]
        "#);
    }

    #[test]
    fn constant_variations() {
        let test = SemanticTokenTest::new(
            r#"
A = 1
AB = 1
ABC = 1
A1 = 1
AB1 = 1
ABC1 = 1
A_B = 1
A1_B = 1
A_B1 = 1
A_1 = 1
"#,
        );

        let tokens = test.highlight_file();

        assert_snapshot!(test.to_snapshot(&tokens), @r#"
        "A" @ 1..2: Variable [readonly]
        "AB" @ 7..9: Variable [readonly]
        "ABC" @ 14..17: Variable [readonly]
        "A1" @ 22..24: Variable [readonly]
        "AB1" @ 29..32: Variable [readonly]
        "ABC1" @ 37..41: Variable [readonly]
        "A_B" @ 46..49: Variable [readonly]
        "A1_B" @ 54..58: Variable [readonly]
        "A_B1" @ 63..67: Variable [readonly]
        "A_1" @ 72..75: Variable [readonly]
        "#);
    }

    #[test]
    fn nonlocal_and_global_statements() {
        let test = SemanticTokenTest::new(
            r#"
x = "global_value"
y = "another_global"

def outer():
    x = "outer_value"
    z = "outer_local"

    def inner():
        nonlocal x, z  # These should be variable tokens
        global y       # This should be a variable token
        x = "modified"
        y = "modified_global"
        z = "modified_local"

        def deeper():
            nonlocal x    # Variable token
            global y, x   # Both should be variable tokens
            return x + y

        return deeper

    return inner
"#,
        );

        let tokens = test.highlight_file();

        assert_snapshot!(test.to_snapshot(&tokens), @r#"
        "x" @ 1..2: Variable
        "y" @ 20..21: Variable
        "outer" @ 46..51: Function [declaration]
        "x" @ 59..60: Variable
        "z" @ 81..82: Variable
        "inner" @ 108..113: Function [declaration]
        "x" @ 134..135: Variable
        "z" @ 137..138: Variable
        "y" @ 189..190: Variable
        "x" @ 239..240: Variable
        "y" @ 262..263: Variable
        "z" @ 292..293: Variable
        "deeper" @ 326..332: Function [declaration]
        "x" @ 357..358: Variable
        "y" @ 398..399: Variable
        "x" @ 401..402: Variable
        "x" @ 457..458: Variable
        "y" @ 461..462: Variable
        "deeper" @ 479..485: Function
        "inner" @ 498..503: Function
        "#);
    }

    #[test]
    fn nonlocal_global_edge_cases() {
        let test = SemanticTokenTest::new(
            r#"
# Single variable statements
def test():
    global x
    nonlocal y

    # Multiple variables in one statement
    global a, b, c
    nonlocal d, e, f

    return x + y + a + b + c + d + e + f
"#,
        );

        let tokens = test.highlight_file();

        assert_snapshot!(test.to_snapshot(&tokens), @r#""test" @ 34..38: Function [declaration]"#);
    }

    #[test]
    fn pattern_matching() {
        let test = SemanticTokenTest::new(
            r#"
def process_data(data):
    match data:
        case {"name": name, "age": age, **rest} as person:
            print(f"Person {name}, age {age}, extra: {rest}")
            return person
        case [first, *remaining] as sequence:
            print(f"First: {first}, remaining: {remaining}")
            return sequence
        case value as fallback:
            print(f"Fallback: {fallback}")
            return fallback
"#,
        );

        let tokens = test.highlight_file();

        assert_snapshot!(test.to_snapshot(&tokens), @r#"
        "process_data" @ 5..17: Function [declaration]
        "data" @ 18..22: Parameter [declaration, parameter]
        "data" @ 35..39: Parameter [parameter]
        "name" @ 63..67: Variable
        "age" @ 76..79: Variable
        "rest" @ 83..87: Variable
        "person" @ 92..98: Variable
        "print" @ 112..117: Function [defaultLibrary, builtin]
        "name" @ 128..132: Variable
        "age" @ 140..143: Variable
        "rest" @ 154..158: Variable
        "person" @ 181..187: Variable
        "first" @ 202..207: Variable
        "remaining" @ 210..219: Variable
        "sequence" @ 224..232: Variable
        "print" @ 246..251: Function [defaultLibrary, builtin]
        "first" @ 262..267: Variable
        "remaining" @ 282..291: Variable
        "sequence" @ 314..322: Variable
        "value" @ 336..341: Variable
        "fallback" @ 345..353: Variable
        "print" @ 367..372: Function [defaultLibrary, builtin]
        "fallback" @ 386..394: Variable
        "fallback" @ 417..425: Variable
        "#);
    }

    #[test]
    fn exception_handlers() {
        let test = SemanticTokenTest::new(
            r#"
try:
    x = 1 / 0
except ValueError as ve:
    print(ve)
except (TypeError, RuntimeError) as re:
    print(re)
except Exception as e:
    print(e)
finally:
    pass
"#,
        );

        let tokens = test.highlight_file();

        assert_snapshot!(test.to_snapshot(&tokens), @r#"
        "x" @ 10..11: Variable
        "ValueError" @ 27..37: Class [defaultLibrary, builtin]
        "ve" @ 41..43: Variable
        "print" @ 49..54: Function [defaultLibrary, builtin]
        "ve" @ 55..57: Variable
        "TypeError" @ 67..76: Class [defaultLibrary, builtin]
        "RuntimeError" @ 78..90: Class [defaultLibrary, builtin]
        "re" @ 95..97: Variable
        "print" @ 103..108: Function [defaultLibrary, builtin]
        "re" @ 109..111: Variable
        "Exception" @ 120..129: Class [defaultLibrary, builtin]
        "e" @ 133..134: Variable
        "print" @ 140..145: Function [defaultLibrary, builtin]
        "e" @ 146..147: Variable
        "#);
    }

    #[test]
    fn self_attribute_expression() {
        let test = SemanticTokenTest::new(
            r#"
from typing import Self


class C:
    def __init__(self: Self):
        self.annotated: int = 1
        self.non_annotated = 1
        self.x.test()
        self.x()
"#,
        );

        let tokens = test.highlight_file();

        assert_snapshot!(test.to_snapshot(&tokens), @r#"
        "typing" @ 6..12: Namespace
        "Self" @ 20..24: Class
        "C" @ 33..34: Class [declaration]
        "__init__" @ 44..52: Method [declaration, classMember]
        "self" @ 53..57: SelfParameter [declaration, parameter]
        "Self" @ 59..63: Type
        "self" @ 74..78: SelfParameter [parameter]
        "annotated" @ 79..88: Property [classMember]
        "int" @ 90..93: Class [defaultLibrary, builtin]
        "self" @ 106..110: SelfParameter [parameter]
        "non_annotated" @ 111..124: Property [classMember]
        "self" @ 137..141: SelfParameter [parameter]
        "self" @ 159..163: SelfParameter [parameter]
        "#);
    }

    #[test]
    fn augmented_assignment() {
        let test = SemanticTokenTest::new(
            r#"
x = 0
x += 1
"#,
        );

        let tokens = test.highlight_file();

        assert_snapshot!(test.to_snapshot(&tokens), @r#"
        "x" @ 1..2: Variable
        "x" @ 7..8: Variable
        "#);
    }

    #[test]
    fn type_alias() {
        let test = SemanticTokenTest::new("type MyList[T] = list[T]");

        let tokens = test.highlight_file();

        assert_snapshot!(test.to_snapshot(&tokens), @r#"
        "type" @ 0..4: Keyword
        "MyList" @ 5..11: Class
        "T" @ 12..13: TypeParameter
        "list" @ 17..21: Class [defaultLibrary, builtin]
        "T" @ 22..23: TypeParameter
        "#);
    }

    #[test]
    fn for_stmt() {
        let test = SemanticTokenTest::new(
            r#"
for item in []:
    print(item)
else:
    print(0)
"#,
        );

        let tokens = test.highlight_file();

        assert_snapshot!(test.to_snapshot(&tokens), @r#"
        "item" @ 5..9: Variable
        "print" @ 21..26: Function [defaultLibrary, builtin]
        "item" @ 27..31: Variable
        "print" @ 43..48: Function [defaultLibrary, builtin]
        "#);
    }

    #[test]
    fn with_stmt() {
        let test = SemanticTokenTest::new(
            r#"
with open("file.txt") as f:
    f.read()
"#,
        );

        let tokens = test.highlight_file();

        assert_snapshot!(test.to_snapshot(&tokens), @r#"
        "open" @ 6..10: Function [defaultLibrary, builtin]
        "f" @ 26..27: Variable
        "f" @ 33..34: Variable
        "read" @ 35..39: Method [classMember]
        "#);
    }

    #[test]
    fn comprehensions() {
        let test = SemanticTokenTest::new(
            r#"
list_comp = [x for x in range(10) if x % 2 == 0]
set_comp = {x for x in range(10)}
dict_comp = {k: v for k, v in zip(["a", "b"], [1, 2])}
generator = (x for x in range(10))
"#,
        );

        let tokens = test.highlight_file();
        assert_snapshot!(test.to_snapshot(&tokens), @r#"
        "list_comp" @ 1..10: Variable
        "x" @ 14..15: Variable
        "x" @ 20..21: Variable
        "range" @ 25..30: Class [defaultLibrary, builtin]
        "x" @ 38..39: Variable
        "set_comp" @ 50..58: Variable
        "x" @ 62..63: Variable
        "x" @ 68..69: Variable
        "range" @ 73..78: Class [defaultLibrary, builtin]
        "dict_comp" @ 84..93: Variable
        "k" @ 97..98: Variable
        "v" @ 100..101: Variable
        "k" @ 106..107: Variable
        "v" @ 109..110: Variable
        "zip" @ 114..117: Class [defaultLibrary, builtin]
        "generator" @ 139..148: Variable
        "x" @ 152..153: Variable
        "x" @ 158..159: Variable
        "range" @ 163..168: Class [defaultLibrary, builtin]
        "#);
    }

    /// Regression test for <https://github.com/astral-sh/ty/issues/1406>
    #[test]
    fn invalid_kwargs() {
        let test = SemanticTokenTest::new(
            r#"
def foo(self, **key, value=10):
    return
"#,
        );

        let tokens = test.highlight_file();

        assert_snapshot!(test.to_snapshot(&tokens), @r#"
        "foo" @ 5..8: Function [declaration]
        "self" @ 9..13: Parameter [declaration, parameter]
        "key" @ 17..20: Parameter [declaration, parameter]
        "value" @ 22..27: Parameter [declaration, parameter]
        "#);
    }

    #[test]
    fn import_as() {
        let test = SemanticTokenTest::new(
            r#"
            import pathlib as path
            from pathlib import Path
            "#,
        );

        let tokens = test.highlight_file();
        assert_snapshot!(test.to_snapshot(&tokens), @r#"
        "pathlib" @ 8..15: Namespace
        "path" @ 19..23: Namespace
        "pathlib" @ 29..36: Namespace
        "Path" @ 44..48: Class
        "#);
    }

    #[test]
    fn import_from_as() {
        // Test that both the imported name and its alias get highlighted
        // See: https://github.com/astral-sh/ty/issues/2547
        let test = SemanticTokenTest::new(
            r#"
from pathlib import Path as P
from collections.abc import Set as AbstractSet
"#,
        );

        let tokens = test.highlight_file();
        // Both the imported name and the alias should get the same highlighting
        assert_snapshot!(test.to_snapshot(&tokens), @r#"
        "pathlib" @ 6..13: Namespace
        "Path" @ 21..25: Class
        "P" @ 29..30: Class
        "collections" @ 36..47: Namespace
        "abc" @ 48..51: Namespace
        "Set" @ 59..62: Class
        "AbstractSet" @ 66..77: Class
        "#);
    }

    #[test]
    fn unresolved_names_do_not_receive_semantic_tokens() {
        let test = SemanticTokenTest::new(
            r#"
def f():
    missing()
"#,
        );

        let tokens = test.highlight_file();
        assert_snapshot!(test.to_snapshot(&tokens), @r#""f" @ 5..6: Function [declaration]"#);
    }

    #[test]
    fn private_builtin_helpers_do_not_receive_semantic_tokens() {
        // Private helpers excluded from implicit builtin lookup must remain unresolved for IDE
        // highlighting instead of receiving tokens from their typeshed definitions.
        let test = SemanticTokenTest::new("_T_co\n_P\n");

        let tokens = test.highlight_file();
        assert_snapshot!(test.to_snapshot(&tokens), @"");
    }

    #[test]
    fn unresolved_attributes_do_not_receive_semantic_tokens() {
        let test = SemanticTokenTest::new(
            r#"
class C: ...

def f(c: C):
    c.missing()
"#,
        );

        let tokens = test.highlight_file();
        assert_snapshot!(test.to_snapshot(&tokens), @r#"
        "C" @ 7..8: Class [declaration]
        "f" @ 19..20: Function [declaration]
        "c" @ 21..22: Parameter [declaration, parameter]
        "C" @ 24..25: Class
        "c" @ 32..33: Parameter [parameter]
        "#);
    }

    #[test]
    fn unresolved_imported_names_do_not_receive_semantic_tokens() {
        let test = SemanticTokenTest::new(
            r#"
from pathlib import Missing as Alias
"#,
        );

        let tokens = test.highlight_file();
        assert_snapshot!(test.to_snapshot(&tokens), @r#""pathlib" @ 6..13: Namespace"#);
    }

    #[test]
    fn keyword_argument_classification() {
        let test = SemanticTokenTest::new(
            r#"
from dataclasses import dataclass
from typing import Callable


def apply(values: list[int], key: Callable[[int], int], reverse: bool = False) -> None: ...


@dataclass
class Point:
    x: int
    y: int = 0


apply([1], key=abs, reverse=True)
Point(x=1, y=2)
print("a", sep=", ")
"#,
        );

        assert_snapshot!(test.to_snapshot(&test.highlight_file()), @r#"
        "dataclasses" @ 6..17: Namespace
        "dataclass" @ 25..34: Function
        "typing" @ 40..46: Namespace
        "Callable" @ 54..62: Class
        "apply" @ 69..74: Function [declaration]
        "values" @ 75..81: Parameter [declaration, parameter]
        "list" @ 83..87: Class [defaultLibrary, builtin]
        "int" @ 88..91: Class [defaultLibrary, builtin]
        "key" @ 94..97: Function [declaration, parameter]
        "Callable" @ 99..107: Class
        "int" @ 109..112: Class [defaultLibrary, builtin]
        "int" @ 115..118: Class [defaultLibrary, builtin]
        "reverse" @ 121..128: Parameter [declaration, parameter]
        "bool" @ 130..134: Class [defaultLibrary, builtin]
        "@" @ 159..160: Decorator
        "dataclass" @ 160..169: Decorator
        "Point" @ 176..181: Class [declaration]
        "x" @ 187..188: Property [static, classMember]
        "int" @ 190..193: Class [defaultLibrary, builtin]
        "y" @ 198..199: Property [static, classMember]
        "int" @ 201..204: Class [defaultLibrary, builtin]
        "apply" @ 211..216: Function
        "key" @ 222..225: Function [parameter]
        "abs" @ 226..229: Function [defaultLibrary, builtin]
        "reverse" @ 231..238: Parameter [parameter]
        "Point" @ 245..250: Class
        "x" @ 251..252: Property [static, classMember, parameter]
        "y" @ 256..257: Property [static, classMember, parameter]
        "print" @ 261..266: Function [defaultLibrary, builtin]
        "sep" @ 272..275: Parameter [parameter]
        "#);
    }

    #[test]
    fn enum_classification() {
        let test = SemanticTokenTest::new(
            r#"
from enum import Enum, IntEnum


class Color(Enum):
    RED = 1
    GREEN = 2


class Level(IntEnum):
    LOW = 1


color = Color.RED
level = Level.LOW
name = color.name
"#,
        );

        assert_snapshot!(test.to_snapshot(&test.highlight_file()), @r#"
        "enum" @ 6..10: Namespace
        "Enum" @ 18..22: Enum
        "IntEnum" @ 24..31: Enum
        "Color" @ 40..45: Enum [declaration]
        "Enum" @ 46..50: Enum
        "RED" @ 57..60: EnumMember
        "GREEN" @ 69..74: EnumMember
        "Level" @ 87..92: Enum [declaration]
        "IntEnum" @ 93..100: Enum
        "LOW" @ 107..110: EnumMember
        "color" @ 117..122: Variable
        "Color" @ 125..130: Enum
        "RED" @ 131..134: EnumMember
        "level" @ 135..140: Variable
        "Level" @ 143..148: Enum
        "LOW" @ 149..152: EnumMember
        "name" @ 153..157: Variable
        "color" @ 160..165: Variable
        "name" @ 166..170: Property [readonly, classMember]
        "#);
    }

    #[test]
    fn builtin_function_and_method_classification() {
        let test = SemanticTokenTest::new(
            r#"
values = [3, 1, 2]
values.append(len(values))
text = "a,b".split(",")
count = len(text)
mapping: dict[str, int] = {}
mapping.get("a")
"#,
        );

        assert_snapshot!(test.to_snapshot(&test.highlight_file()), @r#"
        "values" @ 1..7: Variable
        "values" @ 20..26: Variable
        "append" @ 27..33: Method [defaultLibrary, builtin, classMember]
        "len" @ 34..37: Function [defaultLibrary, builtin]
        "values" @ 38..44: Variable
        "text" @ 47..51: Variable
        "split" @ 60..65: Method [defaultLibrary, builtin, classMember]
        "count" @ 71..76: Variable
        "len" @ 79..82: Function [defaultLibrary, builtin]
        "text" @ 83..87: Variable
        "mapping" @ 89..96: Variable
        "dict" @ 98..102: Class [defaultLibrary, builtin]
        "str" @ 103..106: Class [defaultLibrary, builtin]
        "int" @ 108..111: Class [defaultLibrary, builtin]
        "mapping" @ 118..125: Variable
        "get" @ 126..129: Method [defaultLibrary, builtin, classMember]
        "#);
    }

    #[test]
    fn pseudo_generic_class_attributes() {
        let test = SemanticTokenTest::new(
            r#"
class Wrapper:
    def __init__(self, value, label="default"):
        self.value = value
        self.label = label

    def show(self):
        return self.value, self.label
"#,
        );

        assert_snapshot!(test.to_snapshot(&test.highlight_file()), @r#"
        "Wrapper" @ 7..14: Class [declaration]
        "__init__" @ 24..32: Method [declaration, classMember]
        "self" @ 33..37: SelfParameter [declaration, parameter]
        "value" @ 39..44: Parameter [declaration, parameter]
        "label" @ 46..51: Parameter [declaration, parameter]
        "self" @ 72..76: SelfParameter [parameter]
        "value" @ 77..82: Property [classMember]
        "value" @ 85..90: Parameter [parameter]
        "self" @ 99..103: SelfParameter [parameter]
        "label" @ 104..109: Property [classMember]
        "label" @ 112..117: Parameter [parameter]
        "show" @ 127..131: Method [declaration, classMember]
        "self" @ 132..136: SelfParameter [declaration, parameter]
        "self" @ 154..158: SelfParameter [parameter]
        "value" @ 159..164: Property [classMember]
        "self" @ 166..170: SelfParameter [parameter]
        "label" @ 171..176: Property [classMember]
        "#);
    }

    #[test]
    fn typed_declaration_in_base_class_wins() {
        let test = SemanticTokenTest::new(
            r#"
class Base:
    value: str

    def __init__(self) -> None:
        self.value = ""


class Derived(Base):
    def __init__(self) -> None:
        super().__init__()
        self.value = "derived"


item = Derived()
item.value
"#,
        );

        assert_snapshot!(test.to_snapshot(&test.highlight_file()), @r#"
        "Base" @ 7..11: Class [declaration]
        "value" @ 17..22: Property [static, classMember]
        "str" @ 24..27: Class [defaultLibrary, builtin]
        "__init__" @ 37..45: Method [declaration, classMember]
        "self" @ 46..50: SelfParameter [declaration, parameter]
        "self" @ 69..73: SelfParameter [parameter]
        "value" @ 74..79: Property [static, classMember]
        "Derived" @ 93..100: Class [declaration]
        "Base" @ 101..105: Class
        "__init__" @ 116..124: Method [declaration, classMember]
        "self" @ 125..129: SelfParameter [declaration, parameter]
        "super" @ 148..153: Class [defaultLibrary, builtin]
        "__init__" @ 156..164: Method [classMember]
        "self" @ 175..179: SelfParameter [parameter]
        "value" @ 180..185: Property [static, classMember]
        "item" @ 200..204: Variable
        "Derived" @ 207..214: Class
        "item" @ 217..221: Variable
        "value" @ 222..227: Property [static, classMember]
        "#);
    }

    #[test]
    fn unreachable_code_classification() {
        let test = SemanticTokenTest::new(
            r#"
import sys

CONSTANT = 1


def function(count: int) -> int:
    return count
    local = count
    print(sys, CONSTANT, len, local, count)
"#,
        );

        assert_snapshot!(test.to_snapshot(&test.highlight_file()), @r#"
        "sys" @ 8..11: Namespace
        "CONSTANT" @ 13..21: Variable [readonly]
        "function" @ 32..40: Function [declaration]
        "count" @ 41..46: Parameter [declaration, parameter]
        "int" @ 48..51: Class [defaultLibrary, builtin]
        "int" @ 56..59: Class [defaultLibrary, builtin]
        "count" @ 72..77: Parameter [parameter]
        "count" @ 90..95: Variable
        "print" @ 100..105: Function [defaultLibrary, builtin]
        "sys" @ 106..109: Namespace
        "CONSTANT" @ 111..119: Variable
        "len" @ 121..124: Function [defaultLibrary, builtin]
        "count" @ 133..138: Variable
        "#);
    }

    #[test]
    fn legend_matches_discriminants() {
        // The LSP legend and the playground's token kinds index into these lists by discriminant.
        for (index, token_type) in SemanticTokenType::all().iter().enumerate() {
            assert_eq!(*token_type as usize, index, "{token_type:?}");
        }
        let names = SemanticTokenModifier::all_names();
        assert_eq!(names.len(), SemanticTokenModifier::all().iter().count());
        for (index, modifier) in SemanticTokenModifier::all().iter().enumerate() {
            assert_eq!(modifier.bits(), 1 << index, "{}", names[index]);
        }
    }

    #[test]
    fn literals_are_not_classified() {
        let test = SemanticTokenTest::new(
            r#"
"""Module docstring."""
x = (1, 2.5, "text", b"bytes", f"{None}", True, ..., "implicit" "concat")
"#,
        );

        assert_snapshot!(test.to_snapshot(&test.highlight_file()), @r#""x" @ 25..26: Variable"#);
    }

    #[test]
    fn declaration_without_value_in_class_body() {
        let test = SemanticTokenTest::new(
            r#"
from dataclasses import dataclass
from datetime import date

@dataclass
class Event:
    date: date
    when: date

y = 2
"#,
        );

        assert_snapshot!(test.to_snapshot(&test.highlight_file()), @r#"
        "dataclasses" @ 6..17: Namespace
        "dataclass" @ 25..34: Function
        "datetime" @ 40..48: Namespace
        "date" @ 56..60: Class
        "@" @ 62..63: Decorator
        "dataclass" @ 63..72: Decorator
        "Event" @ 79..84: Class [declaration]
        "date" @ 90..94: Property [static, classMember]
        "date" @ 96..100: Variable
        "when" @ 105..109: Property [static, classMember]
        "date" @ 111..115: Variable
        "y" @ 117..118: Variable
        "#);
    }

    #[test]
    fn unreachable_use_of_declared_global() {
        let test = SemanticTokenTest::new(
            r#"
declared: int

def f():
    return
    print(declared)

class WithDeclaredOnly:
    value: int

    @property
    def value(self) -> int:
        return 1

WithDeclaredOnly().value = 3
"#,
        );

        assert_snapshot!(test.to_snapshot(&test.highlight_file()), @r#"
        "declared" @ 1..9: Variable
        "int" @ 11..14: Class [defaultLibrary, builtin]
        "f" @ 20..21: Function [declaration]
        "print" @ 40..45: Function [defaultLibrary, builtin]
        "declared" @ 46..54: Variable
        "WithDeclaredOnly" @ 63..79: Class [declaration]
        "value" @ 85..90: Property [static, classMember]
        "int" @ 92..95: Class [defaultLibrary, builtin]
        "@" @ 101..102: Decorator
        "property" @ 102..110: Decorator
        "value" @ 119..124: Property [declaration, classMember]
        "self" @ 125..129: SelfParameter [declaration, parameter]
        "int" @ 134..137: Class [defaultLibrary, builtin]
        "WithDeclaredOnly" @ 157..173: Class
        "value" @ 176..181: Property [static, classMember]
        "#);
    }

    #[test]
    fn lambda_in_string_annotation() {
        let test = SemanticTokenTest::new(
            r#"
x: "lambda a: a" = 1

def f(p: "lambda b: 0"): ...

y = 2
"#,
        );

        assert_snapshot!(test.to_snapshot(&test.highlight_file()), @r#"
        "x" @ 1..2: Variable
        "a" @ 12..13: Parameter [declaration, parameter]
        "a" @ 15..16: Parameter [parameter]
        "f" @ 27..28: Function [declaration]
        "p" @ 29..30: Parameter [declaration, parameter]
        "b" @ 40..41: Parameter [declaration, parameter]
        "y" @ 53..54: Variable
        "#);
    }

    #[test]
    fn slots_entries_are_properties() {
        let test = SemanticTokenTest::new(
            r#"
class C:
    __slots__ = ("a", "B")

    def __init__(self):
        self.a = 1

C().a
C().B
"#,
        );

        assert_snapshot!(test.to_snapshot(&test.highlight_file()), @r#"
        "C" @ 7..8: Class [declaration]
        "__slots__" @ 14..23: Property [static, classMember]
        "__init__" @ 46..54: Method [declaration, classMember]
        "self" @ 55..59: SelfParameter [declaration, parameter]
        "self" @ 70..74: SelfParameter [parameter]
        "a" @ 75..76: Property [static, classMember]
        "C" @ 82..83: Class
        "a" @ 86..87: Property [static, classMember]
        "C" @ 88..89: Class
        "B" @ 92..93: Property [readonly, static, classMember]
        "#);
    }

    #[test]
    fn with_item_member_write_access() {
        let test = SemanticTokenTest::new(
            r#"
class Lock:
    def __enter__(self): ...
    def __exit__(self, *args): ...

class Holder:
    @property
    def lock(self) -> Lock:
        return Lock()

    @property
    def inner(self) -> "Holder":
        return self

holder = Holder()
with holder.lock:
    pass
with holder.inner.lock:
    pass
"#,
        );

        assert_snapshot!(test.to_snapshot(&test.highlight_file()), @r#"
        "Lock" @ 7..11: Class [declaration]
        "__enter__" @ 21..30: Method [declaration, classMember]
        "self" @ 31..35: SelfParameter [declaration, parameter]
        "__exit__" @ 50..58: Method [declaration, classMember]
        "self" @ 59..63: SelfParameter [declaration, parameter]
        "args" @ 66..70: Parameter [declaration, parameter]
        "Holder" @ 84..90: Class [declaration]
        "@" @ 96..97: Decorator
        "property" @ 97..105: Decorator
        "lock" @ 114..118: Property [declaration, readonly, classMember]
        "self" @ 119..123: SelfParameter [declaration, parameter]
        "Lock" @ 128..132: Class
        "Lock" @ 149..153: Class
        "@" @ 161..162: Decorator
        "property" @ 162..170: Decorator
        "inner" @ 179..184: Property [declaration, readonly, classMember]
        "self" @ 185..189: SelfParameter [declaration, parameter]
        "Holder" @ 195..201: Class
        "self" @ 219..223: SelfParameter [parameter]
        "holder" @ 225..231: Variable
        "Holder" @ 234..240: Class
        "holder" @ 248..254: Variable
        "lock" @ 255..259: Property [readonly, classMember]
        "holder" @ 275..281: Variable
        "inner" @ 282..287: Property [readonly, classMember]
        "lock" @ 288..292: Property [readonly, classMember]
        "#);
    }

    #[test]
    fn conditional_expression_skipped_branch() {
        let test = SemanticTokenTest::new(
            r#"
import os
import sys
from typing import TYPE_CHECKING

class K: ...

a = K if True else os.path.join
b = os.sep if TYPE_CHECKING else K
c = K if sys.version_info >= (3, 99) else os
d = (lambda v: v + len("")) if False else None
"#,
        );

        assert_snapshot!(test.to_snapshot(&test.highlight_file()), @r#"
        "os" @ 8..10: Namespace
        "sys" @ 18..21: Namespace
        "typing" @ 27..33: Namespace
        "TYPE_CHECKING" @ 41..54: Variable [readonly]
        "K" @ 62..63: Class [declaration]
        "a" @ 70..71: Class
        "K" @ 74..75: Class
        "os" @ 89..91: Variable
        "path" @ 92..96: Variable
        "b" @ 102..103: Variable
        "os" @ 106..108: Namespace
        "sep" @ 109..112: Variable
        "TYPE_CHECKING" @ 116..129: Variable [readonly]
        "c" @ 137..138: Namespace
        "sys" @ 146..149: Namespace
        "version_info" @ 150..162: Variable
        "os" @ 179..181: Namespace
        "d" @ 182..183: Variable
        "v" @ 194..195: Parameter [declaration]
        "v" @ 197..198: Parameter
        "len" @ 201..204: Function [defaultLibrary, builtin]
        "#);
    }

    #[test]
    fn string_annotations_only_in_annotations() {
        let test = SemanticTokenTest::new(
            r#"
from typing import Annotated, Literal, cast

a: "int" = 1
b: Literal["int"] = "int"
c: Annotated[int, "int"] = 1
d = cast("int", a)
e: "list['int']" = []
"#,
        );

        assert_snapshot!(test.to_snapshot(&test.highlight_file()), @r#"
        "typing" @ 6..12: Namespace
        "Annotated" @ 20..29: Class
        "Literal" @ 31..38: Class
        "cast" @ 40..44: Function
        "a" @ 46..47: Variable
        "int" @ 50..53: Class [defaultLibrary, builtin]
        "b" @ 59..60: Variable
        "Literal" @ 62..69: Class
        "c" @ 85..86: Variable
        "Annotated" @ 88..97: Class
        "int" @ 98..101: Class [defaultLibrary, builtin]
        "\"int\"" @ 103..108: String
        "d" @ 114..115: Variable
        "cast" @ 118..122: Function
        "a" @ 130..131: Variable
        "e" @ 133..134: Variable
        "list" @ 137..141: Class [defaultLibrary, builtin]
        "int" @ 143..146: Class [defaultLibrary, builtin]
        "#);
    }

    #[test]
    fn class_body_declaration_precedes_method_assignment() {
        let test = SemanticTokenTest::new(
            r#"
class C:
    def set(self):
        self.x = "text"

    x: int = 0

C().x
"#,
        );

        assert_snapshot!(test.to_snapshot(&test.highlight_file()), @r#"
        "C" @ 7..8: Class [declaration]
        "set" @ 18..21: Method [declaration, classMember]
        "self" @ 22..26: SelfParameter [declaration, parameter]
        "self" @ 37..41: SelfParameter [parameter]
        "x" @ 42..43: Property [static, classMember]
        "x" @ 58..59: Property [static, classMember]
        "int" @ 61..64: Class [defaultLibrary, builtin]
        "C" @ 70..71: Class
        "x" @ 74..75: Property [static, classMember]
        "#);
    }

    #[test]
    fn typevar_and_partial_keywords() {
        let test = SemanticTokenTest::new(
            r#"
import functools
from typing import TypeVar

T = TypeVar("T", bound=int, covariant=True)

def f(base: int, other: str) -> None: ...

g = functools.partial(f, base=2)
"#,
        );

        assert_snapshot!(test.to_snapshot(&test.highlight_file()), @r#"
        "functools" @ 8..17: Namespace
        "typing" @ 23..29: Namespace
        "TypeVar" @ 37..44: Class
        "T" @ 46..47: TypeParameter [readonly]
        "TypeVar" @ 50..57: Class
        "int" @ 69..72: Class [defaultLibrary, builtin]
        "f" @ 95..96: Function [declaration]
        "base" @ 97..101: Parameter [declaration, parameter]
        "int" @ 103..106: Class [defaultLibrary, builtin]
        "other" @ 108..113: Parameter [declaration, parameter]
        "str" @ 115..118: Class [defaultLibrary, builtin]
        "g" @ 134..135: Variable
        "functools" @ 138..147: Namespace
        "partial" @ 148..155: Class
        "f" @ 156..157: Function
        "base" @ 159..163: Parameter
        "#);
    }

    #[test]
    fn functional_named_tuple_fields() {
        let test = SemanticTokenTest::new(
            r#"
from collections import namedtuple

Point = namedtuple("Point", ["x", "y"])
p = Point(x=1, y=2)
p.x
"#,
        );

        assert_snapshot!(test.to_snapshot(&test.highlight_file()), @r#"
        "collections" @ 6..17: Namespace
        "namedtuple" @ 25..35: Function
        "Point" @ 37..42: Class
        "namedtuple" @ 45..55: Function
        "p" @ 77..78: Variable
        "Point" @ 81..86: Class
        "x" @ 87..88: Variable
        "y" @ 92..93: Variable
        "p" @ 97..98: Variable
        "x" @ 99..100: Variable
        "#);
    }

    #[test]
    fn paramspec_args_kwargs() {
        let test = SemanticTokenTest::new(
            r#"
from typing import Callable, ParamSpec

P = ParamSpec("P")

def decorate(fn: Callable[P, int]) -> Callable[P, int]:
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> int:
        return fn(*args, **kwargs)
    return wrapper
"#,
        );

        assert_snapshot!(test.to_snapshot(&test.highlight_file()), @r#"
        "typing" @ 6..12: Namespace
        "Callable" @ 20..28: Class
        "ParamSpec" @ 30..39: Class
        "P" @ 41..42: TypeParameter [readonly]
        "ParamSpec" @ 45..54: Class
        "decorate" @ 65..73: Function [declaration]
        "fn" @ 74..76: Function [declaration, parameter]
        "Callable" @ 78..86: Class
        "P" @ 87..88: TypeParameter [readonly]
        "int" @ 90..93: Class [defaultLibrary, builtin]
        "Callable" @ 99..107: Class
        "P" @ 108..109: TypeParameter [readonly]
        "int" @ 111..114: Class [defaultLibrary, builtin]
        "wrapper" @ 125..132: Function [declaration]
        "args" @ 134..138: Parameter [declaration, parameter]
        "P" @ 140..141: TypeParameter [readonly]
        "args" @ 142..146: TypeParameter
        "kwargs" @ 150..156: Parameter [declaration, parameter]
        "P" @ 158..159: TypeParameter [readonly]
        "kwargs" @ 160..166: TypeParameter
        "int" @ 171..174: Class [defaultLibrary, builtin]
        "fn" @ 191..193: Function [parameter]
        "args" @ 195..199: Parameter [parameter]
        "kwargs" @ 203..209: Parameter [parameter]
        "wrapper" @ 222..229: Function
        "#);
    }

    #[test]
    fn staticmethod_under_cache_decorator() {
        let test = SemanticTokenTest::new(
            r#"
import functools

class C:
    @staticmethod
    @functools.cache
    def cached() -> int:
        return 1

C.cached()
"#,
        );

        assert_snapshot!(test.to_snapshot(&test.highlight_file()), @r#"
        "functools" @ 8..17: Namespace
        "C" @ 25..26: Class [declaration]
        "@" @ 32..33: Decorator
        "staticmethod" @ 33..45: Decorator
        "@" @ 50..51: Decorator
        "functools" @ 51..60: Namespace
        "cache" @ 61..66: Function
        "cached" @ 75..81: Method [declaration, static, classMember]
        "int" @ 87..90: Class [defaultLibrary, builtin]
        "C" @ 110..111: Class
        "cached" @ 112..118: Method [static, classMember]
        "#);
    }

    #[test]
    fn implicit_class_attributes() {
        let test = SemanticTokenTest::new(
            r#"
class C: ...

C.__doc__
C.__module__
C.__qualname__
C().__module__
C().__qualname__
"#,
        );

        assert_snapshot!(test.to_snapshot(&test.highlight_file()), @r#"
        "C" @ 7..8: Class [declaration]
        "C" @ 15..16: Class
        "__doc__" @ 17..24: Property [static, classMember]
        "C" @ 25..26: Class
        "__module__" @ 27..37: Property [static, classMember]
        "C" @ 38..39: Class
        "__qualname__" @ 40..52: Property [static, classMember]
        "C" @ 53..54: Class
        "__module__" @ 57..67: Property [classMember]
        "C" @ 68..69: Class
        "#);
    }

    #[test]
    fn range_request_inside_string_annotation() {
        let test = SemanticTokenTest::new(
            r#"
from typing import Optional

def f(value: "Optional[int]") -> None: ...
"#,
        );

        let source = ruff_db::source::source_text(&test.db, test.file);
        let start = source.find("int]").expect("annotation to be present");
        let range = TextRange::at(TextSize::try_from(start).unwrap(), "int".text_len());
        assert_snapshot!(test.to_snapshot(&test.highlight_range(range)), @r#""int" @ 53..56: Class [defaultLibrary, builtin]"#);
    }

    #[test]
    fn implicit_type_aliases() {
        let test = SemanticTokenTest::new(
            r#"
import typing
from typing import Any, Literal, NoReturn, Optional

JSON = Any
Incomplete = typing.Any
Bottom = NoReturn
Maybe = Optional[int]
Single = Literal["r"]
Multi = Literal["r", "w"]

class K:
    Alias = Any

def f(j: JSON) -> Bottom:
    local = Any
    raise ValueError
"#,
        );

        assert_snapshot!(test.to_snapshot(&test.highlight_file()), @r#"
        "typing" @ 8..14: Namespace
        "typing" @ 20..26: Namespace
        "Any" @ 34..37: Type
        "Literal" @ 39..46: Class
        "NoReturn" @ 48..56: Type
        "Optional" @ 58..66: Class
        "JSON" @ 68..72: Type [readonly]
        "Any" @ 75..78: Type
        "Incomplete" @ 79..89: Type
        "typing" @ 92..98: Namespace
        "Any" @ 99..102: Type
        "Bottom" @ 103..109: Type
        "NoReturn" @ 112..120: Type
        "Maybe" @ 121..126: Type
        "Optional" @ 129..137: Class
        "int" @ 138..141: Class [defaultLibrary, builtin]
        "Single" @ 143..149: Class
        "Literal" @ 152..159: Class
        "Multi" @ 165..170: Type
        "Literal" @ 173..180: Class
        "K" @ 198..199: Class [declaration]
        "Alias" @ 205..210: Type [static, classMember]
        "Any" @ 213..216: Type
        "f" @ 222..223: Function [declaration]
        "j" @ 224..225: Parameter [declaration, parameter]
        "JSON" @ 227..231: Type [readonly]
        "Bottom" @ 236..242: Type
        "local" @ 248..253: Class
        "Any" @ 256..259: Class
        "ValueError" @ 270..280: Class [defaultLibrary, builtin]
        "#);
    }

    #[test]
    fn value_subscripts() {
        let test = SemanticTokenTest::new(
            r#"
import enum
from typing import Any

class Color(enum.Enum):
    RED = 1

def lookup(name: str, cls: type) -> None:
    Color[name]
    list[Any]
    dict[str, cls]
"#,
        );

        assert_snapshot!(test.to_snapshot(&test.highlight_file()), @r#"
        "enum" @ 8..12: Namespace
        "typing" @ 18..24: Namespace
        "Any" @ 32..35: Type
        "Color" @ 43..48: Enum [declaration]
        "enum" @ 49..53: Namespace
        "Enum" @ 54..58: Enum
        "RED" @ 65..68: EnumMember
        "lookup" @ 78..84: Function [declaration]
        "name" @ 85..89: Parameter [declaration, parameter]
        "str" @ 91..94: Class [defaultLibrary, builtin]
        "cls" @ 96..99: Parameter [declaration, parameter]
        "type" @ 101..105: Class [defaultLibrary, builtin]
        "Color" @ 120..125: Enum
        "name" @ 126..130: Parameter [parameter]
        "list" @ 136..140: Class [defaultLibrary, builtin]
        "Any" @ 141..144: Type
        "dict" @ 150..154: Class [defaultLibrary, builtin]
        "str" @ 155..158: Class [defaultLibrary, builtin]
        "cls" @ 160..163: Parameter [parameter]
        "#);
    }

    #[test]
    fn code_after_no_return_call_is_bound() {
        let test = SemanticTokenTest::new(
            r#"
import sys

CONSTANT = 1

class K:
    attr = 1

def f(k: K):
    sys.exit(1)
    print(CONSTANT, k.attr)

def g(k: K):
    raise SystemExit
    print(CONSTANT, k.attr)

try:
    V = 1
except Exception as error:
    error.args
"#,
        );

        assert_snapshot!(test.to_snapshot(&test.highlight_file()), @r#"
        "sys" @ 8..11: Namespace
        "CONSTANT" @ 13..21: Variable [readonly]
        "K" @ 33..34: Class [declaration]
        "attr" @ 40..44: Property [static, classMember]
        "f" @ 54..55: Function [declaration]
        "k" @ 56..57: Parameter [declaration, parameter]
        "K" @ 59..60: Class
        "sys" @ 67..70: Namespace
        "exit" @ 71..75: Function
        "print" @ 83..88: Function [defaultLibrary, builtin]
        "CONSTANT" @ 89..97: Variable [readonly]
        "k" @ 99..100: Parameter [parameter]
        "attr" @ 101..105: Property [static, classMember]
        "g" @ 112..113: Function [declaration]
        "k" @ 114..115: Parameter [declaration, parameter]
        "K" @ 117..118: Class
        "SystemExit" @ 131..141: Class [defaultLibrary, builtin]
        "print" @ 146..151: Function [defaultLibrary, builtin]
        "CONSTANT" @ 152..160: Variable
        "k" @ 162..163: Variable
        "attr" @ 164..168: Property [classMember]
        "V" @ 180..181: Variable [readonly]
        "Exception" @ 193..202: Class [defaultLibrary, builtin]
        "error" @ 206..211: Variable
        "error" @ 217..222: Variable
        "args" @ 223..227: Property [static, classMember]
        "#);
    }

    #[test]
    fn unresolved_module_import() {
        let test = SemanticTokenTest::new(
            r#"
import not_a_module
import not_a_module as alias
from not_a_module import thing

not_a_module.x
alias.x
thing
print(__spec__, __name__)
"#,
        );

        assert_snapshot!(test.to_snapshot(&test.highlight_file()), @r#"
        "not_a_module" @ 8..20: Namespace
        "not_a_module" @ 28..40: Namespace
        "alias" @ 44..49: Namespace
        "not_a_module" @ 55..67: Namespace
        "not_a_module" @ 82..94: Namespace
        "alias" @ 97..102: Namespace
        "print" @ 111..116: Function [defaultLibrary, builtin]
        "__name__" @ 127..135: Variable
        "#);
    }

    #[test]
    fn super_and_property_getters() {
        let test = SemanticTokenTest::new(
            r#"
class Base:
    def __init__(self):
        self.base_attr = 1

class Derived(Base):
    def use(self):
        super().base_attr

    @property
    def model(self):
        return Base

    @property
    def length(self):
        return len

Derived().model
Derived().length
"#,
        );

        assert_snapshot!(test.to_snapshot(&test.highlight_file()), @r#"
        "Base" @ 7..11: Class [declaration]
        "__init__" @ 21..29: Method [declaration, classMember]
        "self" @ 30..34: SelfParameter [declaration, parameter]
        "self" @ 45..49: SelfParameter [parameter]
        "base_attr" @ 50..59: Property [classMember]
        "Derived" @ 71..78: Class [declaration]
        "Base" @ 79..83: Class
        "use" @ 94..97: Method [declaration, classMember]
        "self" @ 98..102: SelfParameter [declaration, parameter]
        "super" @ 113..118: Class [defaultLibrary, builtin]
        "base_attr" @ 121..130: Property [classMember]
        "@" @ 136..137: Decorator
        "property" @ 137..145: Decorator
        "model" @ 154..159: Class [declaration, readonly, classMember]
        "self" @ 160..164: SelfParameter [declaration, parameter]
        "Base" @ 182..186: Class
        "@" @ 192..193: Decorator
        "property" @ 193..201: Decorator
        "length" @ 210..216: Function [declaration, readonly, classMember]
        "self" @ 217..221: SelfParameter [declaration, parameter]
        "len" @ 239..242: Function [defaultLibrary, builtin]
        "Derived" @ 244..251: Class
        "model" @ 254..259: Class [readonly, classMember]
        "Derived" @ 260..267: Class
        "length" @ 270..276: Function [readonly, defaultLibrary, builtin, classMember]
        "#);
    }

    #[test]
    fn receiver_parameter_rules() {
        let test = SemanticTokenTest::new(
            r#"
def decorate(cls):
    return cls

class C:
    def keyword_only(*, x):
        return x

    def meth(self):
        self.undeclared

    def __getattr__(self, name: str) -> int:
        return 1

@decorate
class D:
    pass
"#,
        );

        assert_snapshot!(test.to_snapshot(&test.highlight_file()), @r#"
        "decorate" @ 5..13: Function [declaration]
        "cls" @ 14..17: Parameter [declaration, parameter]
        "cls" @ 31..34: Parameter [parameter]
        "C" @ 42..43: Class [declaration]
        "keyword_only" @ 53..65: Method [declaration, classMember]
        "x" @ 69..70: Parameter [declaration, parameter]
        "x" @ 88..89: Parameter [parameter]
        "meth" @ 99..103: Method [declaration, classMember]
        "self" @ 104..108: SelfParameter [declaration, parameter]
        "self" @ 119..123: SelfParameter [parameter]
        "undeclared" @ 124..134: Property [classMember]
        "__getattr__" @ 144..155: Method [declaration, classMember]
        "self" @ 156..160: SelfParameter [declaration, parameter]
        "name" @ 162..166: Parameter [declaration, parameter]
        "str" @ 168..171: Class [defaultLibrary, builtin]
        "int" @ 176..179: Class [defaultLibrary, builtin]
        "@" @ 199..200: Decorator
        "decorate" @ 200..208: Decorator
        "D" @ 215..216: Class [declaration]
        "#);
    }

    #[test]
    fn annotated_metadata_strings() {
        let test = SemanticTokenTest::new(
            r#"
from typing import Annotated as Float

from typing import Annotated

def f(x: Float[int, "batch seq"], y: Annotated[int, "meta", 3]) -> None: ...
"#,
        );

        assert_snapshot!(test.to_snapshot(&test.highlight_file()), @r#"
        "typing" @ 6..12: Namespace
        "Annotated" @ 20..29: Class
        "Float" @ 33..38: Class
        "typing" @ 45..51: Namespace
        "Annotated" @ 59..68: Class
        "f" @ 74..75: Function [declaration]
        "x" @ 76..77: Parameter [declaration, parameter]
        "Float" @ 79..84: Class
        "int" @ 85..88: Class [defaultLibrary, builtin]
        "\"batch seq\"" @ 90..101: String
        "y" @ 104..105: Parameter [declaration, parameter]
        "Annotated" @ 107..116: Class
        "int" @ 117..120: Class [defaultLibrary, builtin]
        "\"meta\"" @ 122..128: String
        "#);
    }

    struct SemanticTokenTest {
        db: ty_project::TestDb,
        file: File,
    }

    impl SemanticTokenTest {
        fn new(source: &str) -> Self {
            let mut db =
                ty_project::TestDb::new(ProjectMetadata::new("test", SystemPathBuf::from("/")));

            let path = SystemPath::new("src/main.py");
            db.write_file(path, ruff_python_trivia::textwrap::dedent(source))
                .expect("Write to memory file system to always succeed");

            let file = system_path_to_file(&db, path).expect("newly written file to existing");

            Self { db, file }
        }

        /// Get semantic tokens for the entire file
        fn highlight_file(&self) -> SemanticTokens {
            semantic_tokens(
                &self.db,
                ProgramFile::new(
                    &self.db,
                    self.file,
                    self.db.program_environment().program(&self.db),
                ),
                None,
            )
        }

        /// Get semantic tokens for a specific range in the file
        fn highlight_range(&self, range: TextRange) -> SemanticTokens {
            semantic_tokens(
                &self.db,
                ProgramFile::new(
                    &self.db,
                    self.file,
                    self.db.program_environment().program(&self.db),
                ),
                Some(range),
            )
        }

        /// Helper function to convert semantic tokens to a snapshot-friendly text format
        fn to_snapshot(&self, tokens: &SemanticTokens) -> String {
            use std::fmt::Write;
            let source = ruff_db::source::source_text(&self.db, self.file);
            let mut result = String::new();

            for token in tokens.iter() {
                let token_text = &source[token.range()];
                let modifiers_text = if token.modifiers.is_empty() {
                    String::new()
                } else {
                    let names = SemanticTokenModifier::all_names();
                    let mods: Vec<&str> = (0..names.len())
                        .filter(|bit| token.modifiers.bits() & (1 << bit) != 0)
                        .map(|bit| names[bit])
                        .collect();
                    format!(" [{}]", mods.join(", "))
                };

                writeln!(
                    result,
                    "{:?} @ {}..{}: {:?}{}",
                    token_text,
                    u32::from(token.start()),
                    u32::from(token.end()),
                    token.token_type,
                    modifiers_text
                )
                .unwrap();
            }

            result
        }
    }
}
