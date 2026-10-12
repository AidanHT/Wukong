//! The 8-bit floats `f8e4m3` (OCP E4M3FN) and `f8e5m2` (OCP E5M2): the literal range check.
//!
//! An f8 value is computed in `f32` and rounded onto its own grid (nearest, ties to even) at every
//! value, so a literal of the type is worth its rounding. Two roundings would silently change the
//! number the programmer wrote, and both are `E0438`:
//!
//! * **past the largest finite value** — E4M3FN has no infinity and saturates to ±448, E5M2
//!   overflows to ±∞ — so `500.0f8e4m3` would be `448` and `70000.0f8e5m2` infinity;
//! * **to zero** — a nonzero literal below half the least subnormal (2⁻¹⁰ for E4M3FN, 2⁻¹⁷ for
//!   E5M2) would be `0`.
//!
//! A literal that merely loses digits (`0.1f8e4m3` is `0.1015625`) is the format's precision, not
//! an error. The check runs where a literal *has* an f8 type: a suffixed literal when it is typed,
//! and an unsuffixed one at every site it adapts to an f8 annotation, parameter, field or return
//! (`Sema::range_check_int_literal`, which every adaptation site already calls). Arithmetic that
//! overflows at run time is the format's saturation / overflow rule, not this check's.

use wukong_ast::{BinOp, Expr, ExprKind, UnOp};
use wukong_types::Scalar;

use crate::Sema;

/// The value of a float literal's text: the optional sign, a float type suffix and `_`
/// separators stripped, the rest parsed as an `f64` (the literal's exact value rounded once, as
/// every float literal is read). `None` for a malformed literal (already `E0431`).
pub(crate) fn float_literal_value(text: &str) -> Option<f64> {
    let mut s = text.trim();
    if s.starts_with('-') || s.starts_with('+') {
        s = &s[1..];
    }
    for suf in ["f8e4m3", "f8e5m2", "bf16", "f16", "f32", "f64", "f"] {
        if let Some(x) = s.strip_suffix(suf) {
            s = x;
            break;
        }
    }
    s.replace('_', "").parse::<f64>().ok()
}

/// `(mantissa bits, least normal exponent, largest finite value)` of an f8 type.
fn f8_params(sc: Scalar) -> Option<(i32, i32, f64)> {
    match sc {
        Scalar::F8E4M3 => Some((3, -6, 448.0)),
        Scalar::F8E5M2 => Some((2, -14, 57344.0)),
        _ => None,
    }
}

/// `|v|` rounded to nearest, ties to even, onto the grid of an f8 format with an unbounded
/// exponent range (so a value past the largest finite one rounds to its true neighbour).
fn round_to_f8_grid(v: f64, mbits: i32, emin: i32) -> f64 {
    let a = v.abs();
    if a == 0.0 || !a.is_finite() {
        return a;
    }
    let e = (((a.to_bits() >> 52) & 0x7ff) as i32 - 1023).max(emin);
    let ulp = 2f64.powi(e - mbits);
    (a / ulp).round_ties_even() * ulp
}

/// Why the literal value `v` does not fit the f8 type `sc`, or `None` when it does.
pub(crate) fn f8_literal_problem(v: f64, sc: Scalar) -> Option<String> {
    let (mbits, emin, max) = f8_params(sc)?;
    let r = round_to_f8_grid(v, mbits, emin);
    if r > max {
        let becomes = if sc == Scalar::F8E4M3 {
            "saturate to 448 (E4M3FN has no infinity)"
        } else {
            "become infinity"
        };
        return Some(format!(
            "rounds past the largest finite value, {max}, so it would {becomes}"
        ));
    }
    if r == 0.0 && v != 0.0 {
        let least = 2f64.powi(emin - mbits);
        return Some(format!(
            "rounds to zero (the least nonzero value is {least}); write `0.0` if zero is meant"
        ));
    }
    None
}

impl Sema<'_> {
    /// `E0438` for a suffixed f8 literal (`500.0f8e4m3`), checked where it is typed.
    pub(crate) fn check_suffixed_f8_literal(&mut self, e: &Expr, text: &str, sc: Scalar) {
        if let Some(why) = float_literal_value(text).and_then(|v| f8_literal_problem(v, sc)) {
            self.error(
                e.span,
                "E0438",
                format!(
                    "literal `{text}` is out of range for `{}`: it {why}",
                    sc.name()
                ),
            );
        }
    }

    /// `E0438` for each unsuffixed float literal of `e` adapting to the f8 type `sc`: a lone literal,
    /// one under a unary minus, or each leaf of an all-literal expression (the forms
    /// `literal_adapts` accepts), since every leaf is rounded to `sc` before the arithmetic runs.
    pub(crate) fn range_check_f8_literal(&mut self, e: &Expr, sc: Scalar) {
        match &e.kind {
            ExprKind::Float(s) => {
                let text = self.sym_str(*s).to_string();
                if crate::has_float_suffix(&text) {
                    return;
                }
                if let Some(why) =
                    float_literal_value(&text).and_then(|v| f8_literal_problem(v, sc))
                {
                    self.error(
                        e.span,
                        "E0438",
                        format!(
                            "literal `{text}` is out of range for `{}`: it {why}",
                            sc.name()
                        ),
                    );
                }
            }
            ExprKind::Unary {
                op: UnOp::Neg,
                expr,
            } => self.range_check_f8_literal(expr, sc),
            ExprKind::Binary {
                op: BinOp::Add | BinOp::Sub | BinOp::Mul | BinOp::Div | BinOp::Rem,
                lhs,
                rhs,
            } => {
                self.range_check_f8_literal(lhs, sc);
                self.range_check_f8_literal(rhs, sc);
            }
            _ => {}
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn literal_values_strip_suffixes_and_separators() {
        assert_eq!(float_literal_value("1.5f8e4m3"), Some(1.5));
        assert_eq!(float_literal_value("1_000.25f8e5m2"), Some(1000.25));
        assert_eq!(float_literal_value("2e3"), Some(2000.0));
        assert_eq!(float_literal_value("1.5z"), None);
    }

    /// The boundaries, derived from the formats: E4M3FN's largest finite value is 448 = 1.75·2⁸ and
    /// the midpoint to the absent 480 is 464, which ties to the even 448 (in range) — anything
    /// above it rounds past 448. E5M2's largest is 57344 = 1.75·2¹⁵ and the midpoint to 2¹⁶ is
    /// 61440, which ties to the even 2¹⁶ (infinity). Half the least subnormal (2⁻¹⁰, 2⁻¹⁷) ties to
    /// the even 0.
    #[test]
    fn range_boundaries() {
        let e4 = Scalar::F8E4M3;
        let e5 = Scalar::F8E5M2;
        assert!(f8_literal_problem(448.0, e4).is_none());
        assert!(f8_literal_problem(464.0, e4).is_none());
        assert!(f8_literal_problem(464.0001, e4).is_some());
        assert!(f8_literal_problem(-1e6, e4).is_some());
        assert!(f8_literal_problem(57344.0, e5).is_none());
        assert!(f8_literal_problem(61439.9, e5).is_none());
        assert!(f8_literal_problem(61440.0, e5).is_some());
        assert!(f8_literal_problem(0.0, e4).is_none());
        assert!(f8_literal_problem(2f64.powi(-9), e4).is_none());
        assert!(f8_literal_problem(2f64.powi(-10) * 1.0001, e4).is_none());
        assert!(f8_literal_problem(2f64.powi(-10), e4).is_some());
        assert!(f8_literal_problem(2f64.powi(-17), e5).is_some());
        assert!(f8_literal_problem(2f64.powi(-16), e5).is_none());
        assert!(f8_literal_problem(0.1, Scalar::F32).is_none());
    }
}
