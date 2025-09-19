use std::ops::{Deref, DerefMut};

use p3_field::{ExtensionField, Field};
use rand::{
    Rng,
    distr::{Distribution, StandardUniform},
};

/// A point `(x_1, ..., x_n)` in `F^n` for some field `F`.
///
/// Often, `x_i` are binary. If strictly binary, `BinaryHypercubePoint` is used.
#[derive(Default, Debug, Clone, PartialEq, Eq, Hash)]
pub struct MultilinearPoint<F>(pub Vec<F>);

impl<F> Deref for MultilinearPoint<F> {
    type Target = Vec<F>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<F> DerefMut for MultilinearPoint<F> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl<F> MultilinearPoint<F>
where
    F: Field,
{
    /// Returns the number of variables (dimension `n`).
    #[inline]
    #[must_use]
    pub fn num_variables(&self) -> usize {
        self.len()
    }

    /// Converts a univariate evaluation point into a multilinear one.
    ///
    /// Uses the bijection:
    /// ```ignore
    /// f(x_1, ..., x_n) <-> g(y) := f(y^(2^(n-1)), ..., y^4, y^2, y)
    /// ```
    /// Meaning:
    /// ```ignore
    /// x_1^i_1 * ... * x_n^i_n <-> y^i
    /// ```
    /// where `(i_1, ..., i_n)` is the **big-endian** binary decomposition of `i`.
    ///
    /// Reversing the order ensures the **big-endian** convention.
    pub fn expand_from_univariate(point: F, num_variables: usize) -> Self {
        let mut res = Vec::with_capacity(num_variables);
        let mut cur = point;

        for _ in 0..num_variables {
            res.push(cur);
            cur = cur.square(); // Compute y^(2^k) at each step
        }

        res.reverse();
        Self(res)
    }

    /// Computes `eq(c, p)`, where `p` is a general `MultilinearPoint` (not necessarily binary).
    ///
    /// The **equality polynomial** for two vectors is:
    /// ```ignore
    /// eq(s1, s2) = ∏ (s1_i * s2_i + (1 - s1_i) * (1 - s2_i))
    /// ```
    /// which evaluates to `1` if `s1 == s2`, and `0` otherwise.
    ///
    /// This uses the algebraic identity:
    /// ```ignore
    /// s1_i * s2_i + (1 - s1_i) * (1 - s2_i) = 1 + 2 * s1_i * s2_i - s1_i - s2_i
    /// ```
    /// to avoid unnecessary multiplications.
    #[must_use]
    pub fn eq_poly_outside(&self, point: &Self) -> F {
        assert_eq!(self.num_variables(), point.num_variables());

        let mut acc = F::ONE;

        for (&l, &r) in self.iter().zip(&point.0) {
            // l * r + (1 - l) * (1 - r) = 1 + 2 * l * r - l - r
            // +/- much cheaper than multiplication.
            acc *= F::ONE + l * r.double() - l - r;
        }

        acc
    }

    /// Computes `eq3(c, p)`, the **equality polynomial** for `{0,1,2}^n`.
    ///
    /// `p` is interpreted as a **big-endian** ternary number.
    ///
    /// `eq3(c, p)` is the unique polynomial of **degree ≤ 2** in each variable,
    /// such that:
    /// ```ignore
    /// eq3(c, p) = 1  if c == p
    ///           = 0  otherwise
    /// ```
    /// Uses precomputed values to reduce redundant operations.
    #[must_use]
    pub fn eq_poly3(&self, mut point: usize) -> F {
        let n_variables = self.num_variables();
        assert!(point < 3usize.pow(n_variables as u32));

        let mut acc = F::ONE;

        // Iterate in **little-endian** order and adjust using big-endian convention.
        for &val in self.iter().rev() {
            let val_minus_one = val - F::ONE;
            let val_minus_two = val - F::TWO;

            acc *= match point % 3 {
                0 => val_minus_one * val_minus_two.halve(), // (val - 1)(val - 2) / 2
                1 => -val * val_minus_two,                  // val (val - 2)(-1)
                2 => val * val_minus_one.halve(),           // val (val - 1) / 2
                _ => unreachable!(),
            };
            point /= 3;
        }

        acc
    }

    /// Embeds the point into an extension field `EF`.
    #[must_use]
    pub fn embed<EF: ExtensionField<F>>(&self) -> MultilinearPoint<EF> {
        MultilinearPoint(self.0.iter().map(|&x| EF::from(x)).collect())
    }
}

impl<F> MultilinearPoint<F>
where
    StandardUniform: Distribution<F>,
{
    pub fn rand<R: Rng>(rng: &mut R, num_variables: usize) -> Self {
        Self(
            (0..num_variables)
                .map(|_| rng.sample(StandardUniform))
                .collect(),
        )
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Evaluation<F> {
    pub point: MultilinearPoint<F>,
    pub value: F,
}

impl<F: Field> Evaluation<F> {
    pub fn new(point: MultilinearPoint<F>, value: F) -> Self {
        Self { point, value }
    }

    pub fn num_variables(&self) -> usize {
        self.point.num_variables()
    }
}
