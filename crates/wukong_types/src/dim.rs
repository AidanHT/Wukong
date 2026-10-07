//! Symbolic dimension arithmetic: the closed algebra a tensor extent may be written in — `3*D`,
//! `H*HD`, `(L - K)/S + 1`, `N % 4` — and its **canonical normal form**.
//!
//! A derived extent is held as an integer polynomial ([`DimPoly`]) over *atoms* ([`DimAtom`]): a
//! dimension variable, or a floor division / remainder that does not simplify. The polynomial is
//! kept sorted with no zero coefficients, so two spellings of one value — `N + N` and `2*N`,
//! `H*HD` and `HD*H`, `(2*N + 3)/2` and `N + 1` — are *structurally identical*, and shape equality
//! stays the plain `==` the unifier already uses. This is the approach of JAX's shape polymorphism
//! (`_DimExpr`: a sum of monomials whose factors are variables or irreducible `floordiv`/`mod`
//! atoms), restricted to what a kernel signature needs.
//!
//! **Equality is sound, not complete.** Structurally equal normal forms denote the same function of
//! their variables, so a shape the checker accepts as equal really is equal. The converse does not
//! hold — `N/2 + (N + 1)/2` is `N` for every integer `N`, and the normal form does not see it — so a
//! relation the normal form cannot prove is *rejected*, never assumed. That is the direction a shape
//! checker must err in.
//!
//! **Division is floor division** (`/`) with the matching non-negative remainder (`%`), which is
//! exactly truncation for the non-negative operands an extent has. A constant divisor `k > 0` is
//! folded by splitting the numerator into `k*Q + R` with every coefficient of `R` in `0..k`, so
//! `floor((k*Q + R)/k) = Q + floor(R/k)`: when `R` is a constant it is `< k` and the atom vanishes;
//! otherwise the residual `R/k` stays an atom. Because `R`'s coefficients are non-negative, a
//! residual atom's numerator is non-negative whenever its variables are, so evaluating it with the
//! machine's truncating division at run time agrees with the floor the normal form assumed. A
//! non-constant divisor is never simplified (`(H*HD)/H` is *not* `HD`: at `H = 0` the language's
//! division yields 0, and the rewrite would change the value).
//!
//! Folding is done in `i128` with checked arithmetic; an extent must still fit `u64` when it folds
//! to a constant ([`crate::Dim::from_poly`]).

use std::cmp::Ordering;
use std::collections::BTreeMap;

use wukong_span::{Interner, Symbol};

/// One arithmetic operator of a dimension expression.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum DimOp {
    Add,
    Sub,
    Mul,
    /// Floor division — the same as truncation for the non-negative operands an extent has.
    Div,
    /// The non-negative remainder that matches [`DimOp::Div`].
    Rem,
}

impl DimOp {
    pub fn glyph(self) -> &'static str {
        match self {
            DimOp::Add => "+",
            DimOp::Sub => "-",
            DimOp::Mul => "*",
            DimOp::Div => "/",
            DimOp::Rem => "%",
        }
    }
}

/// Why a dimension expression has no extent.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum DimError {
    /// The expression folds to a negative constant (`2 - 5`); an extent is never negative.
    Negative(i128),
    /// A `/` or `%` whose divisor folds to zero or to a negative constant.
    BadDivisor(i128),
    /// A constant or coefficient left the representable range (`i128` while folding, `u64` for a
    /// constant extent).
    Overflow,
}

impl DimError {
    /// A short human explanation, for the diagnostic that reports it.
    pub fn describe(&self) -> String {
        match self {
            DimError::Negative(c) => {
                format!("it evaluates to {c}, and an extent cannot be negative")
            }
            DimError::BadDivisor(0) => "it divides by zero".to_string(),
            DimError::BadDivisor(c) => {
                format!("it divides by {c}, and a divisor must be a positive extent")
            }
            DimError::Overflow => "its value overflows the 64-bit extent range".to_string(),
        }
    }
}

/// One indivisible symbolic factor of a [`Monomial`].
#[derive(Clone, PartialEq, Eq, Hash, Debug, PartialOrd, Ord)]
pub enum DimAtom {
    /// A dimension variable (a generic parameter).
    Var(Symbol),
    /// `floor(num / den)` that does not simplify: `den` is not a constant, or the constant does not
    /// divide the numerator's symbolic part.
    Div(Box<DimPoly>, Box<DimPoly>),
    /// `num mod den` that does not simplify (see [`DimAtom::Div`]).
    Rem(Box<DimPoly>, Box<DimPoly>),
}

impl DimAtom {
    fn mentions(&self, s: Symbol) -> bool {
        match self {
            DimAtom::Var(v) => *v == s,
            DimAtom::Div(n, d) | DimAtom::Rem(n, d) => n.mentions(s) || d.mentions(s),
        }
    }

    fn collect_vars(&self, out: &mut Vec<Symbol>) {
        match self {
            DimAtom::Var(v) => {
                if !out.contains(v) {
                    out.push(*v);
                }
            }
            DimAtom::Div(n, d) | DimAtom::Rem(n, d) => {
                n.collect_vars(out);
                d.collect_vars(out);
            }
        }
    }

    fn substitute(&self, f: &dyn Fn(Symbol) -> Option<DimPoly>) -> Result<DimPoly, DimError> {
        match self {
            DimAtom::Var(s) => Ok(f(*s).unwrap_or_else(|| DimPoly::var(*s))),
            DimAtom::Div(n, d) => n.substitute(f)?.floor_div(&d.substitute(f)?),
            DimAtom::Rem(n, d) => n.substitute(f)?.rem(&d.substitute(f)?),
        }
    }

    fn eval(&self, f: &dyn Fn(Symbol) -> Option<i128>) -> Option<i128> {
        match self {
            DimAtom::Var(s) => f(*s),
            DimAtom::Div(n, d) => {
                let (n, d) = (n.eval(f)?, d.eval(f)?);
                (d > 0).then(|| n.div_euclid(d))
            }
            DimAtom::Rem(n, d) => {
                let (n, d) = (n.eval(f)?, d.eval(f)?);
                (d > 0).then(|| n.rem_euclid(d))
            }
        }
    }

    fn display(&self, interner: &Interner) -> String {
        match self {
            DimAtom::Var(s) => interner.resolve(*s).to_string(),
            DimAtom::Div(n, d) => {
                format!("{}/{}", n.operand_str(interner), d.operand_str(interner))
            }
            DimAtom::Rem(n, d) => {
                format!("{}%{}", n.operand_str(interner), d.operand_str(interner))
            }
        }
    }
}

/// A product of atoms, each with an exponent `>= 1`, sorted ascending by atom. The empty product is
/// the constant monomial `1`.
#[derive(Clone, PartialEq, Eq, Hash, Debug, PartialOrd, Ord, Default)]
pub struct Monomial(Vec<(DimAtom, u32)>);

impl Monomial {
    /// The factors, sorted ascending, each with its exponent.
    pub fn factors(&self) -> &[(DimAtom, u32)] {
        &self.0
    }

    /// Is this the empty product (the monomial of a constant term)?
    pub fn is_one(&self) -> bool {
        self.0.is_empty()
    }

    fn of(atom: DimAtom) -> Monomial {
        Monomial(vec![(atom, 1)])
    }

    /// The product of two monomials: a merge of the two sorted factor lists, adding exponents.
    fn times(&self, o: &Monomial) -> Result<Monomial, DimError> {
        let mut out = Vec::with_capacity(self.0.len() + o.0.len());
        let (mut i, mut j) = (0, 0);
        while i < self.0.len() && j < o.0.len() {
            match self.0[i].0.cmp(&o.0[j].0) {
                Ordering::Less => {
                    out.push(self.0[i].clone());
                    i += 1;
                }
                Ordering::Greater => {
                    out.push(o.0[j].clone());
                    j += 1;
                }
                Ordering::Equal => {
                    let e = self.0[i]
                        .1
                        .checked_add(o.0[j].1)
                        .ok_or(DimError::Overflow)?;
                    out.push((self.0[i].0.clone(), e));
                    i += 1;
                    j += 1;
                }
            }
        }
        out.extend_from_slice(&self.0[i..]);
        out.extend_from_slice(&o.0[j..]);
        Ok(Monomial(out))
    }

    fn mentions(&self, s: Symbol) -> bool {
        self.0.iter().any(|(a, _)| a.mentions(s))
    }

    /// `N*HD`, `N*N`, `(N + 1)/2`. A division/remainder atom is parenthesized when it is one of
    /// several factors or carries a coefficient, so `2*(N/2)` never reads as `2*N/2`.
    fn display(&self, interner: &Interner, has_coeff: bool) -> String {
        let several = has_coeff || self.0.len() > 1 || self.0.iter().any(|(_, e)| *e > 1);
        let mut parts = Vec::new();
        for (a, e) in &self.0 {
            let s = a.display(interner);
            let s = if several && !matches!(a, DimAtom::Var(_)) {
                format!("({s})")
            } else {
                s
            };
            for _ in 0..*e {
                parts.push(s.clone());
            }
        }
        parts.join("*")
    }
}

/// An integer polynomial over [`DimAtom`]s: `Σ coeff·monomial`, sorted ascending by monomial, with
/// no zero coefficient. The zero polynomial is the empty sum. This is the canonical normal form of a
/// derived extent: see the module documentation.
#[derive(Clone, PartialEq, Eq, Hash, Debug, PartialOrd, Ord, Default)]
pub struct DimPoly(Vec<(Monomial, i128)>);

impl DimPoly {
    pub fn zero() -> DimPoly {
        DimPoly(Vec::new())
    }

    pub fn constant(c: i128) -> DimPoly {
        if c == 0 {
            DimPoly::zero()
        } else {
            DimPoly(vec![(Monomial::default(), c)])
        }
    }

    pub fn var(s: Symbol) -> DimPoly {
        DimPoly(vec![(Monomial::of(DimAtom::Var(s)), 1)])
    }

    fn atom(a: DimAtom) -> DimPoly {
        DimPoly(vec![(Monomial::of(a), 1)])
    }

    /// The terms, sorted ascending by monomial, every coefficient nonzero.
    pub fn terms(&self) -> &[(Monomial, i128)] {
        &self.0
    }

    pub fn is_zero(&self) -> bool {
        self.0.is_empty()
    }

    /// The value, when the polynomial has no symbolic term.
    pub fn as_constant(&self) -> Option<i128> {
        match self.0.as_slice() {
            [] => Some(0),
            [(m, c)] if m.is_one() => Some(*c),
            _ => None,
        }
    }

    /// The variable, when the polynomial is exactly one variable with coefficient 1.
    pub fn as_var(&self) -> Option<Symbol> {
        match self.0.as_slice() {
            [(m, 1)] => match m.0.as_slice() {
                [(DimAtom::Var(s), 1)] => Some(*s),
                _ => None,
            },
            _ => None,
        }
    }

    fn from_map(map: BTreeMap<Monomial, i128>) -> DimPoly {
        DimPoly(map.into_iter().filter(|(_, c)| *c != 0).collect())
    }

    fn accumulate(
        map: &mut BTreeMap<Monomial, i128>,
        m: Monomial,
        c: i128,
    ) -> Result<(), DimError> {
        let slot = map.entry(m).or_insert(0);
        *slot = slot.checked_add(c).ok_or(DimError::Overflow)?;
        Ok(())
    }

    pub fn add(&self, o: &DimPoly) -> Result<DimPoly, DimError> {
        let mut map = BTreeMap::new();
        for (m, c) in self.0.iter().chain(o.0.iter()) {
            Self::accumulate(&mut map, m.clone(), *c)?;
        }
        Ok(Self::from_map(map))
    }

    /// Multiply every coefficient by the constant `k`.
    pub fn scale(&self, k: i128) -> Result<DimPoly, DimError> {
        if k == 0 {
            return Ok(DimPoly::zero());
        }
        let mut out = Vec::with_capacity(self.0.len());
        for (m, c) in &self.0 {
            out.push((m.clone(), c.checked_mul(k).ok_or(DimError::Overflow)?));
        }
        Ok(DimPoly(out))
    }

    pub fn sub(&self, o: &DimPoly) -> Result<DimPoly, DimError> {
        self.add(&o.scale(-1)?)
    }

    pub fn mul(&self, o: &DimPoly) -> Result<DimPoly, DimError> {
        let mut map = BTreeMap::new();
        for (ma, ca) in &self.0 {
            for (mb, cb) in &o.0 {
                let c = ca.checked_mul(*cb).ok_or(DimError::Overflow)?;
                Self::accumulate(&mut map, ma.times(mb)?, c)?;
            }
        }
        Ok(Self::from_map(map))
    }

    /// `self == k*Q + R` with every coefficient of `R` in `0..k` (requires `k > 0`). Both halves stay
    /// sorted because the terms are visited in order.
    fn split_by(&self, k: i128) -> (DimPoly, DimPoly) {
        let mut q = Vec::new();
        let mut r = Vec::new();
        for (m, c) in &self.0 {
            let (qc, rc) = (c.div_euclid(k), c.rem_euclid(k));
            if qc != 0 {
                q.push((m.clone(), qc));
            }
            if rc != 0 {
                r.push((m.clone(), rc));
            }
        }
        (DimPoly(q), DimPoly(r))
    }

    /// Floor division `floor(self / d)`.
    pub fn floor_div(&self, d: &DimPoly) -> Result<DimPoly, DimError> {
        if let Some(k) = d.as_constant() {
            if k <= 0 {
                return Err(DimError::BadDivisor(k));
            }
            if k == 1 {
                return Ok(self.clone());
            }
            let (q, r) = self.split_by(k);
            return match r.as_constant() {
                // A constant residual is in `0..k`, so `floor(R / k)` is 0.
                Some(_) => Ok(q),
                None => q.add(&DimPoly::atom(DimAtom::Div(
                    Box::new(r),
                    Box::new(d.clone()),
                ))),
            };
        }
        if self.is_zero() {
            return Ok(DimPoly::zero());
        }
        Ok(DimPoly::atom(DimAtom::Div(
            Box::new(self.clone()),
            Box::new(d.clone()),
        )))
    }

    /// The non-negative remainder `self mod d` matching [`DimPoly::floor_div`].
    pub fn rem(&self, d: &DimPoly) -> Result<DimPoly, DimError> {
        if let Some(k) = d.as_constant() {
            if k <= 0 {
                return Err(DimError::BadDivisor(k));
            }
            let (_, r) = self.split_by(k);
            return Ok(match r.as_constant() {
                Some(c) => DimPoly::constant(c),
                None => DimPoly::atom(DimAtom::Rem(Box::new(r), Box::new(d.clone()))),
            });
        }
        if self.is_zero() {
            return Ok(DimPoly::zero());
        }
        Ok(DimPoly::atom(DimAtom::Rem(
            Box::new(self.clone()),
            Box::new(d.clone()),
        )))
    }

    /// Exact division by a nonzero constant: `Some` only when every coefficient is a multiple of
    /// `k`, so the quotient is the polynomial `self / k` for every value of the variables.
    pub fn exact_div(&self, k: i128) -> Option<DimPoly> {
        if k == 0 {
            return None;
        }
        let mut out = Vec::with_capacity(self.0.len());
        for (m, c) in &self.0 {
            if c.checked_rem(k)? != 0 {
                return None;
            }
            out.push((m.clone(), c.checked_div(k)?));
        }
        Some(DimPoly(out))
    }

    /// Substitute each variable `s` with `f(s)` (left alone when `f` declines), re-normalizing.
    pub fn substitute(&self, f: &dyn Fn(Symbol) -> Option<DimPoly>) -> Result<DimPoly, DimError> {
        let mut acc = DimPoly::zero();
        for (m, c) in &self.0 {
            let mut term = DimPoly::constant(*c);
            for (a, e) in &m.0 {
                let av = a.substitute(f)?;
                for _ in 0..*e {
                    term = term.mul(&av)?;
                }
            }
            acc = acc.add(&term)?;
        }
        Ok(acc)
    }

    /// Evaluate with the variables bound by `f`; `None` when a variable is unbound, a divisor is not
    /// positive, or the arithmetic overflows.
    pub fn eval(&self, f: &dyn Fn(Symbol) -> Option<i128>) -> Option<i128> {
        let mut acc: i128 = 0;
        for (m, c) in &self.0 {
            let mut t = *c;
            for (a, e) in &m.0 {
                let v = a.eval(f)?;
                for _ in 0..*e {
                    t = t.checked_mul(v)?;
                }
            }
            acc = acc.checked_add(t)?;
        }
        Some(acc)
    }

    /// Every variable the polynomial mentions — inside division/remainder atoms too — once each, in
    /// canonical (deterministic) order.
    pub fn vars(&self) -> Vec<Symbol> {
        let mut out = Vec::new();
        self.collect_vars(&mut out);
        out
    }

    fn collect_vars(&self, out: &mut Vec<Symbol>) {
        for (m, _) in &self.0 {
            for (a, _) in &m.0 {
                a.collect_vars(out);
            }
        }
    }

    pub fn mentions(&self, s: Symbol) -> bool {
        self.0.iter().any(|(m, _)| m.mentions(s))
    }

    /// `self == a*g + rest` with a nonzero constant `a` and a `rest` that does not mention `g` —
    /// the form a call site can solve for `g`. `None` when `g` appears in a product, under a power,
    /// or inside a division/remainder atom, or not at all.
    pub fn linear_in(&self, g: Symbol) -> Option<(i128, DimPoly)> {
        let mut a = 0i128;
        let mut rest = Vec::new();
        for (m, c) in &self.0 {
            if let [(DimAtom::Var(s), 1)] = m.0.as_slice() {
                if *s == g {
                    a = *c;
                    continue;
                }
            }
            if m.mentions(g) {
                return None;
            }
            rest.push((m.clone(), *c));
        }
        (a != 0).then_some((a, DimPoly(rest)))
    }

    /// Render for a diagnostic: the symbolic terms in canonical order, then the constant, so
    /// `N - 1` reads as written.
    pub fn display(&self, interner: &Interner) -> String {
        if self.0.is_empty() {
            return "0".to_string();
        }
        let mut ordered: Vec<&(Monomial, i128)> =
            self.0.iter().filter(|(m, _)| !m.is_one()).collect();
        ordered.extend(self.0.iter().filter(|(m, _)| m.is_one()));
        let mut out = String::new();
        for (i, (m, c)) in ordered.into_iter().enumerate() {
            let neg = *c < 0;
            let mag = c.unsigned_abs();
            if i == 0 {
                if neg {
                    out.push('-');
                }
            } else {
                out.push_str(if neg { " - " } else { " + " });
            }
            if m.is_one() {
                out.push_str(&mag.to_string());
                continue;
            }
            if mag != 1 {
                out.push_str(&mag.to_string());
                out.push('*');
            }
            out.push_str(&m.display(interner, mag != 1));
        }
        out
    }

    /// The display of a division/remainder operand: bare when it is a single constant or variable,
    /// parenthesized otherwise.
    fn operand_str(&self, interner: &Interner) -> String {
        let s = self.display(interner);
        if self.as_constant().is_some_and(|c| c >= 0) || self.as_var().is_some() {
            s
        } else {
            format!("({s})")
        }
    }
}
