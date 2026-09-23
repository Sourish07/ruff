//! Pyright's binder decides part of a file's reachability from the syntax alone, and doesn't bind
//! the code that it finds unreachable: names there have no declarations. Reachability that
//! depends on types (after a call to a `NoReturn` function, for example) is decided later, and
//! that code is bound like any other.
//!
//! ty doesn't make this distinction, so this module repeats the binder's analysis.

use ruff_python_ast::name::Name;
use ruff_python_ast::{self as ast, Expr, Number, PythonVersion, Stmt};
use ruff_text_size::{Ranged, TextRange};

/// Evaluates the conditions that pyright's binder and evaluator consider statically known.
#[derive(Debug)]
pub(super) struct StaticConditions {
    python_version: PythonVersion,
    /// The names that the file imports `sys` as.
    sys_aliases: Vec<Name>,
    /// The names that the file imports `typing` and `typing_extensions` as.
    typing_aliases: Vec<Name>,
}

impl StaticConditions {
    pub(super) fn new(suite: &[Stmt], python_version: PythonVersion) -> Self {
        let mut conditions = Self {
            python_version,
            sys_aliases: Vec::new(),
            typing_aliases: Vec::new(),
        };
        conditions.collect_import_aliases(suite);
        conditions
    }

    fn collect_import_aliases(&mut self, body: &[Stmt]) {
        for statement in body {
            match statement {
                Stmt::Import(import) => {
                    for alias in &import.names {
                        let name = alias.asname.as_ref().unwrap_or(&alias.name).id.clone();
                        match alias.name.as_str() {
                            "sys" => self.sys_aliases.push(name),
                            "typing" | "typing_extensions" => self.typing_aliases.push(name),
                            _ => {}
                        }
                    }
                }
                Stmt::If(if_stmt) => {
                    self.collect_import_aliases(&if_stmt.body);
                    for clause in &if_stmt.elif_else_clauses {
                        self.collect_import_aliases(&clause.body);
                    }
                }
                Stmt::Try(try_stmt) => {
                    self.collect_import_aliases(&try_stmt.body);
                    for handler in &try_stmt.handlers {
                        let ast::ExceptHandler::ExceptHandler(handler) = handler;
                        self.collect_import_aliases(&handler.body);
                    }
                    self.collect_import_aliases(&try_stmt.orelse);
                    self.collect_import_aliases(&try_stmt.finalbody);
                }
                _ => {}
            }
        }
    }

    /// Pyright's `staticExpressions.evaluateStaticBoolLikeExpression`, which the binder uses for
    /// the conditions of `if`, `while` and `assert` statements.
    ///
    /// `sys.platform` comparisons aren't static, because basedpyright checks for every platform by
    /// default.
    pub(super) fn truthiness(&self, expr: &Expr) -> Option<bool> {
        match expr {
            Expr::BooleanLiteral(boolean) => Some(boolean.value),
            Expr::NoneLiteral(_) => Some(false),
            Expr::EllipsisLiteral(_) => Some(true),
            Expr::NumberLiteral(number) => match &number.value {
                Number::Int(int) => Some(int.as_u64() != Some(0)),
                Number::Float(float) => Some(*float != 0.0),
                Number::Complex { real, imag } => Some(*real != 0.0 || *imag != 0.0),
            },
            Expr::StringLiteral(string) => Some(!string.value.is_empty()),
            Expr::BytesLiteral(bytes) => Some(!bytes.value.is_empty()),
            Expr::List(list) => Some(!list.elts.is_empty()),
            Expr::Tuple(tuple) => Some(!tuple.elts.is_empty()),
            Expr::Set(set) => Some(!set.elts.is_empty()),
            Expr::Dict(dict) => Some(!dict.items.is_empty()),
            Expr::Named(named) => self.truthiness(&named.value),
            Expr::Name(name) if name.id.as_str() == "TYPE_CHECKING" => Some(true),
            Expr::Attribute(attribute)
                if attribute.attr.as_str() == "TYPE_CHECKING"
                    && matches!(attribute.value.as_ref(), Expr::Name(module)
                        if self.typing_aliases.contains(&module.id)) =>
            {
                Some(true)
            }
            Expr::UnaryOp(unary) if unary.op == ast::UnaryOp::Not => {
                self.truthiness(&unary.operand).map(|value| !value)
            }
            Expr::BoolOp(bool_op) => bool_operation(bool_op, |operand| self.truthiness(operand)),
            Expr::Compare(compare) => self.version_comparison(compare, &self.sys_aliases),
            _ => None,
        }
    }

    /// Pyright's stricter evaluation of the condition of a conditional expression
    /// (`evaluateStaticBoolExpression`), which only knows `True`, `False`, `TYPE_CHECKING`,
    /// `sys.version_info` comparisons, and `not`, `and` and `or` of those.
    pub(super) fn strict_truthiness(&self, expr: &Expr) -> Option<bool> {
        match expr {
            Expr::BooleanLiteral(boolean) => Some(boolean.value),
            Expr::Name(name) if name.id.as_str() == "TYPE_CHECKING" => Some(true),
            Expr::UnaryOp(unary) if unary.op == ast::UnaryOp::Not => {
                // `not` also decides for operands that are only truthy or falsy (`not 0`).
                self.truthiness(&unary.operand).map(|value| !value)
            }
            Expr::BoolOp(bool_op) => {
                let values = bool_op
                    .values
                    .iter()
                    .map(|value| self.strict_truthiness(value))
                    .collect::<Option<Vec<_>>>()?;
                Some(match bool_op.op {
                    ast::BoolOp::And => values.iter().all(|value| *value),
                    ast::BoolOp::Or => values.iter().any(|value| *value),
                })
            }
            Expr::Compare(compare) => self.version_comparison(compare, &[Name::new_static("sys")]),
            _ => None,
        }
    }

    /// Evaluates `sys.version_info <op> (major, minor, ...)` and `sys.version_info[0] <op> major`,
    /// where `sys` is one of `sys_aliases`. A tuple compares like the version it stands for,
    /// with a missing minor version of 0.
    fn version_comparison(&self, compare: &ast::ExprCompare, sys_aliases: &[Name]) -> Option<bool> {
        let ([op], [left, right]) = (&*compare.ops, &*compare.operands) else {
            return None;
        };
        let is_version_info = |expr: &Expr| {
            matches!(expr, Expr::Attribute(attribute)
                if attribute.attr.as_str() == "version_info"
                    && matches!(attribute.value.as_ref(), Expr::Name(sys)
                        if sys_aliases.contains(&sys.id)))
        };
        let int = |expr: &Expr| match expr {
            Expr::NumberLiteral(ast::ExprNumberLiteral {
                value: Number::Int(int),
                ..
            }) => int.as_u8(),
            _ => None,
        };
        let version = [self.python_version.major, self.python_version.minor];
        let ordering = match (left, right) {
            (left, Expr::Tuple(tuple)) if is_version_info(left) => {
                let expected = tuple.elts.iter().map(int).collect::<Option<Vec<_>>>()?;
                let (major, rest) = expected.split_first()?;
                let minor = rest.first().copied().unwrap_or(0);
                let ordering = version.cmp(&[*major, minor]);
                // ty only knows the major and minor version.
                if ordering.is_eq() && rest.len() > 1 {
                    return None;
                }
                ordering
            }
            (Expr::Subscript(subscript), right)
                if is_version_info(&subscript.value) && int(&subscript.slice) == Some(0) =>
            {
                version[0].cmp(&int(right)?)
            }
            _ => return None,
        };
        Some(match op {
            ast::CmpOp::Lt => ordering.is_lt(),
            ast::CmpOp::LtE => ordering.is_le(),
            ast::CmpOp::Gt => ordering.is_gt(),
            ast::CmpOp::GtE => ordering.is_ge(),
            ast::CmpOp::Eq => ordering.is_eq(),
            ast::CmpOp::NotEq => ordering.is_ne(),
            _ => return None,
        })
    }
}

/// `and` is statically false as soon as one operand is, and statically true if all are; `or` the
/// other way around.
fn bool_operation(
    bool_op: &ast::ExprBoolOp,
    truthiness: impl Fn(&Expr) -> Option<bool>,
) -> Option<bool> {
    let (decisive, other) = match bool_op.op {
        ast::BoolOp::And => (false, true),
        ast::BoolOp::Or => (true, false),
    };
    let mut all_other = true;
    for value in &bool_op.values {
        match truthiness(value) {
            Some(value) if value == decisive => return Some(decisive),
            Some(_) => {}
            None => all_other = false,
        }
    }
    all_other.then_some(other)
}

/// Returns the ranges that pyright's binder finds unreachable, sorted by start.
pub(super) fn unbound_ranges(suite: &[Stmt], conditions: &StaticConditions) -> Vec<TextRange> {
    let mut binder = StaticBinder {
        conditions,
        unbound: Vec::new(),
        loop_breaks: Vec::new(),
    };
    binder.visit_block(suite);
    binder.unbound.sort_by_key(Ranged::start);
    binder.unbound
}

struct StaticBinder<'a> {
    conditions: &'a StaticConditions,
    unbound: Vec<TextRange>,
    /// For each enclosing loop, whether a reachable `break` leaves it.
    loop_breaks: Vec<bool>,
}

impl StaticBinder<'_> {
    /// Visits a block and returns whether its end is reachable.
    fn visit_block(&mut self, body: &[Stmt]) -> bool {
        for (index, statement) in body.iter().enumerate() {
            if !self.visit_stmt(statement) {
                self.mark_unbound(&body[index + 1..]);
                return false;
            }
        }
        true
    }

    fn mark_unbound(&mut self, body: &[Stmt]) {
        if let [first, .., last] | [first @ last] = body {
            self.unbound.push(TextRange::new(first.start(), last.end()));
        }
    }

    /// Visits a loop body and returns whether a reachable `break` leaves the loop.
    fn visit_loop_body(&mut self, body: &[Stmt]) -> bool {
        self.loop_breaks.push(false);
        self.visit_block(body);
        self.loop_breaks.pop().unwrap_or(false)
    }

    /// Visits the body of a function or class, which a `break` can't leave.
    fn visit_scope_body(&mut self, body: &[Stmt]) {
        let loop_breaks = std::mem::take(&mut self.loop_breaks);
        self.visit_block(body);
        self.loop_breaks = loop_breaks;
    }

    /// Visits a statement and returns whether the code after it is reachable.
    fn visit_stmt(&mut self, statement: &Stmt) -> bool {
        match statement {
            Stmt::Return(_) | Stmt::Raise(_) | Stmt::Continue(_) => false,
            Stmt::Break(_) => {
                if let Some(breaks) = self.loop_breaks.last_mut() {
                    *breaks = true;
                }
                false
            }
            Stmt::Assert(assert) => self.conditions.truthiness(&assert.test) != Some(false),
            Stmt::If(if_stmt) => {
                let clauses =
                    std::iter::once((Some(&*if_stmt.test), &if_stmt.body, if_stmt.range())).chain(
                        if_stmt
                            .elif_else_clauses
                            .iter()
                            .map(|clause| (clause.test.as_ref(), &clause.body, clause.range())),
                    );
                let mut end_reachable = false;
                let mut taken = false;
                for (test, body, range) in clauses {
                    if taken {
                        self.unbound.push(range);
                        continue;
                    }
                    match test.map(|test| self.conditions.truthiness(test)) {
                        Some(Some(false)) => self.mark_unbound(body),
                        Some(Some(true)) | None => {
                            end_reachable |= self.visit_block(body);
                            taken = true;
                        }
                        Some(None) => end_reachable |= self.visit_block(body),
                    }
                }
                end_reachable || !taken
            }
            Stmt::While(while_stmt) => match self.conditions.truthiness(&while_stmt.test) {
                Some(false) => {
                    self.mark_unbound(&while_stmt.body);
                    self.visit_block(&while_stmt.orelse)
                }
                // The loop only ends with a `break`, which skips the `else` clause.
                Some(true) => {
                    let breaks = self.visit_loop_body(&while_stmt.body);
                    self.mark_unbound(&while_stmt.orelse);
                    breaks
                }
                None => {
                    let breaks = self.visit_loop_body(&while_stmt.body);
                    self.visit_block(&while_stmt.orelse) || breaks
                }
            },
            Stmt::For(for_stmt) => {
                let breaks = self.visit_loop_body(&for_stmt.body);
                self.visit_block(&for_stmt.orelse) || breaks
            }
            Stmt::Try(try_stmt) => {
                let body_end = self.visit_block(&try_stmt.body);
                // Every statement of the `try` body may raise, so the handlers are reachable.
                let mut handlers_end = false;
                for handler in &try_stmt.handlers {
                    let ast::ExceptHandler::ExceptHandler(handler) = handler;
                    handlers_end |= self.visit_block(&handler.body);
                }
                let else_end = if body_end {
                    self.visit_block(&try_stmt.orelse)
                } else {
                    self.mark_unbound(&try_stmt.orelse);
                    false
                };
                // Pyright's binder keeps the code after a `finally` clause reachable, even if the
                // clause itself doesn't complete.
                self.visit_block(&try_stmt.finalbody);
                else_end || handlers_end
            }
            // Whether a context manager swallows exceptions depends on its type, so the code after
            // a `with` statement is always bound.
            Stmt::With(with_stmt) => {
                self.visit_block(&with_stmt.body);
                true
            }
            Stmt::Match(match_stmt) => {
                let mut end_reachable = false;
                let mut exhaustive = false;
                for case in &match_stmt.cases {
                    if exhaustive {
                        self.unbound.push(case.range());
                        continue;
                    }
                    end_reachable |= self.visit_block(&case.body);
                    // A case without a guard that matches anything makes the cases after it
                    // unreachable.
                    exhaustive = case.guard.is_none()
                        && matches!(&case.pattern, ast::Pattern::MatchAs(pattern) if pattern.pattern.is_none());
                }
                end_reachable || !exhaustive
            }
            Stmt::FunctionDef(function) => {
                self.visit_scope_body(&function.body);
                true
            }
            Stmt::ClassDef(class) => {
                self.visit_scope_body(&class.body);
                true
            }
            _ => true,
        }
    }
}
