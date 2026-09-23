//! Pyright's binder decides part of a file's reachability from the syntax alone, and doesn't bind
//! the code that it finds unreachable: names there have no declarations. Reachability that
//! depends on types (after a call to a `NoReturn` function, for example) is decided later, and
//! that code is bound like any other.
//!
//! ty doesn't make this distinction, so this module repeats the binder's analysis.

use ruff_python_ast::{self as ast, Expr, Number, PythonVersion, Stmt};
use ruff_text_size::{Ranged, TextRange};

/// Returns the ranges that pyright's binder finds unreachable, sorted by start.
pub(super) fn unbound_ranges(suite: &[Stmt], python_version: PythonVersion) -> Vec<TextRange> {
    let mut binder = StaticBinder {
        python_version,
        unbound: Vec::new(),
    };
    binder.visit_block(suite);
    binder.unbound.sort_by_key(Ranged::start);
    binder.unbound
}

struct StaticBinder {
    python_version: PythonVersion,
    unbound: Vec<TextRange>,
}

impl StaticBinder {
    /// Visits a block and returns whether its end is reachable.
    fn visit_block(&mut self, body: &[Stmt]) -> bool {
        for (index, statement) in body.iter().enumerate() {
            if !self.visit_stmt(statement) {
                if let [first, .., last] | [first @ last] = &body[index + 1..] {
                    self.unbound.push(TextRange::new(first.start(), last.end()));
                }
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

    /// Visits a statement and returns whether the code after it is reachable.
    fn visit_stmt(&mut self, statement: &Stmt) -> bool {
        match statement {
            Stmt::Return(_) | Stmt::Raise(_) | Stmt::Continue(_) | Stmt::Break(_) => false,
            Stmt::Assert(assert) => {
                static_truthiness(&assert.test, self.python_version) != Some(false)
            }
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
                    match test.map(|test| static_truthiness(test, self.python_version)) {
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
            Stmt::While(while_stmt) => {
                match static_truthiness(&while_stmt.test, self.python_version) {
                    Some(false) => {
                        self.mark_unbound(&while_stmt.body);
                        self.visit_block(&while_stmt.orelse)
                    }
                    Some(true) => {
                        self.visit_block(&while_stmt.body);
                        if contains_break(&while_stmt.body) {
                            self.visit_block(&while_stmt.orelse);
                            true
                        } else {
                            self.mark_unbound(&while_stmt.orelse);
                            false
                        }
                    }
                    None => {
                        self.visit_block(&while_stmt.body);
                        self.visit_block(&while_stmt.orelse);
                        true
                    }
                }
            }
            Stmt::For(for_stmt) => {
                self.visit_block(&for_stmt.body);
                self.visit_block(&for_stmt.orelse);
                true
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
                let finally_end = self.visit_block(&try_stmt.finalbody);
                (else_end || handlers_end) && finally_end
            }
            // Whether a context manager swallows exceptions depends on its type, so the code after
            // a `with` statement is always bound.
            Stmt::With(with_stmt) => {
                self.visit_block(&with_stmt.body);
                true
            }
            Stmt::Match(match_stmt) => {
                for case in &match_stmt.cases {
                    self.visit_block(&case.body);
                }
                true
            }
            Stmt::FunctionDef(function) => {
                self.visit_block(&function.body);
                true
            }
            Stmt::ClassDef(class) => {
                self.visit_block(&class.body);
                true
            }
            _ => true,
        }
    }
}

/// Whether a loop body contains a `break` for that loop.
fn contains_break(body: &[Stmt]) -> bool {
    body.iter().any(|statement| match statement {
        Stmt::Break(_) => true,
        Stmt::If(if_stmt) => {
            contains_break(&if_stmt.body)
                || if_stmt
                    .elif_else_clauses
                    .iter()
                    .any(|clause| contains_break(&clause.body))
        }
        Stmt::With(with_stmt) => contains_break(&with_stmt.body),
        Stmt::Try(try_stmt) => {
            contains_break(&try_stmt.body)
                || try_stmt.handlers.iter().any(|handler| {
                    let ast::ExceptHandler::ExceptHandler(handler) = handler;
                    contains_break(&handler.body)
                })
                || contains_break(&try_stmt.orelse)
                || contains_break(&try_stmt.finalbody)
        }
        Stmt::Match(match_stmt) => match_stmt
            .cases
            .iter()
            .any(|case| contains_break(&case.body)),
        // A `break` in a nested loop, function or class doesn't leave this loop.
        _ => false,
    })
}

/// Pyright's `staticExpressions.evaluateStaticBoolLikeExpression`, which the binder uses for the
/// conditions of `if`, `while` and `assert` statements.
///
/// `sys.platform` comparisons aren't static, because basedpyright checks for every platform by
/// default.
pub(super) fn static_truthiness(expr: &Expr, python_version: PythonVersion) -> Option<bool> {
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
        Expr::List(list) if list.elts.is_empty() => Some(false),
        Expr::Name(name) if name.id.as_str() == "TYPE_CHECKING" => Some(true),
        Expr::Attribute(attribute)
            if attribute.attr.as_str() == "TYPE_CHECKING"
                && matches!(attribute.value.as_ref(), Expr::Name(module)
                    if matches!(module.id.as_str(), "typing" | "typing_extensions")) =>
        {
            Some(true)
        }
        Expr::UnaryOp(unary) if unary.op == ast::UnaryOp::Not => {
            static_truthiness(&unary.operand, python_version).map(|value| !value)
        }
        Expr::BoolOp(bool_op) => {
            let values = bool_op
                .values
                .iter()
                .map(|value| static_truthiness(value, python_version))
                .collect::<Option<Vec<_>>>()?;
            Some(match bool_op.op {
                ast::BoolOp::And => values.iter().all(|value| *value),
                ast::BoolOp::Or => values.iter().any(|value| *value),
            })
        }
        Expr::Compare(compare) => version_comparison(compare, python_version),
        _ => None,
    }
}

/// Pyright's stricter evaluation of the condition of a conditional expression
/// (`evaluateStaticBoolExpression`), which only knows `True`, `False`, `TYPE_CHECKING`,
/// `sys.version_info` comparisons, and `not`, `and` and `or` of those.
pub(super) fn strict_static_truthiness(expr: &Expr, python_version: PythonVersion) -> Option<bool> {
    match expr {
        Expr::BooleanLiteral(boolean) => Some(boolean.value),
        Expr::Name(name) if name.id.as_str() == "TYPE_CHECKING" => Some(true),
        Expr::UnaryOp(unary) if unary.op == ast::UnaryOp::Not => {
            // `not` also decides for operands that are only truthy or falsy (`not 0`).
            static_truthiness(&unary.operand, python_version).map(|value| !value)
        }
        Expr::BoolOp(bool_op) => {
            let values = bool_op
                .values
                .iter()
                .map(|value| strict_static_truthiness(value, python_version))
                .collect::<Option<Vec<_>>>()?;
            Some(match bool_op.op {
                ast::BoolOp::And => values.iter().all(|value| *value),
                ast::BoolOp::Or => values.iter().any(|value| *value),
            })
        }
        Expr::Compare(compare) => version_comparison(compare, python_version),
        _ => None,
    }
}

/// Evaluates `sys.version_info <op> (major, minor)` and `sys.version_info[0] <op> major`.
fn version_comparison(compare: &ast::ExprCompare, python_version: PythonVersion) -> Option<bool> {
    let ([op], [left, right]) = (&*compare.ops, &*compare.operands) else {
        return None;
    };
    let is_version_info = |expr: &Expr| {
        matches!(expr, Expr::Attribute(attribute)
            if attribute.attr.as_str() == "version_info"
                && matches!(attribute.value.as_ref(), Expr::Name(sys) if sys.id.as_str() == "sys"))
    };
    let int = |expr: &Expr| match expr {
        Expr::NumberLiteral(ast::ExprNumberLiteral {
            value: Number::Int(int),
            ..
        }) => int.as_u8(),
        _ => None,
    };
    let (actual, expected): (Vec<u8>, Vec<u8>) = match (left, right) {
        (left, Expr::Tuple(tuple)) if is_version_info(left) => {
            let expected = tuple.elts.iter().map(int).collect::<Option<Vec<_>>>()?;
            if expected.is_empty() || expected.len() > 2 {
                return None;
            }
            let actual = [python_version.major, python_version.minor][..expected.len()].to_vec();
            (actual, expected)
        }
        (Expr::Subscript(subscript), right)
            if is_version_info(&subscript.value) && int(&subscript.slice) == Some(0) =>
        {
            (vec![python_version.major], vec![int(right)?])
        }
        _ => return None,
    };
    let ordering = actual.cmp(&expected);
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
