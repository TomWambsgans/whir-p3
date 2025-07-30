use p3_field::Field;

use crate::poly::multilinear::MultilinearPoint;

/// Represents a polynomial stored in evaluation form over a ternary domain {0,1,2}^n.
///
/// This structure is uniquely determined by its evaluations over the ternary hypercube.
/// The order of storage follows big-endian lexicographic ordering with respect to the
/// evaluation points.
///
/// Given `n_variables`, the number of stored evaluations is `3^n_variables`.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct SumcheckPolynomial<F> {
    /// Number of variables in the polynomial (defines the dimension of the evaluation domain).
    n_variables: usize,
    /// Vector of function evaluations at points in `{0,1,2}^n_variables`, stored in lexicographic
    /// order.
    evaluations: Vec<F>,
}

impl<F> SumcheckPolynomial<F>
where
    F: Field,
{
    /// Creates a new sumcheck polynomial with `n_variables` variables.
    ///
    /// # Parameters:
    /// - `evaluations`: A vector of function values evaluated on `{0,1,2}^n_variables`.
    /// - `n_variables`: The number of variables (determines the evaluation domain size).
    ///
    /// The vector `evaluations` **must** have a length of `3^n_variables`.
    #[must_use]
    pub const fn new(evaluations: Vec<F>, n_variables: usize) -> Self {
        Self {
            n_variables,
            evaluations,
        }
    }

    /// Returns the vector of stored evaluations.
    ///
    /// The order follows lexicographic ordering of the ternary hypercube `{0,1,2}^n_variables`:
    ///
    /// ```ignore
    /// evaluations[i] = h(x_1, x_2, ..., x_n)  where (x_1, ..., x_n) ∈ {0,1,2}^n
    /// ```
    #[must_use]
    pub fn evaluations(&self) -> &[F] {
        &self.evaluations
    }

    /// Computes the sum of function values over the Boolean hypercube `{0,1}^n_variables`.
    ///
    /// Instead of summing over all `3^n` evaluations, this method only sums over points where all
    /// coordinates are 0 or 1.
    ///
    /// Mathematically, this computes:
    /// ```ignore
    /// sum = ∑ f(x_1, ..., x_n)  where  (x_1, ..., x_n) ∈ {0,1}^n
    /// ```
    #[must_use]
    pub fn sum_over_boolean_hypercube(&self) -> F {
        (0..(1 << self.n_variables))
            .map(|point| self.evaluations[self.binary_to_ternary_index(point)])
            .sum()
    }

    /// Converts a binary index `(0..2^n)` to its corresponding ternary index `(0..3^n)`.
    ///
    /// This maps a Boolean hypercube `{0,1}^n` to the ternary hypercube `{0,1,2}^n`.
    ///
    /// Given a binary index:
    /// ```ignore
    /// binary_index = b_{n-1} b_{n-2} ... b_0  (in bits)
    /// ```
    /// The corresponding **ternary index** is computed as:
    /// ```ignore
    /// ternary_index = b_0 * 3^0 + b_1 * 3^1 + ... + b_{n-1} * 3^{n-1}
    /// ```
    ///
    /// # Example:
    /// ```ignore
    /// binary index 0b11  (3 in decimal)  →  ternary index 4
    /// binary index 0b10  (2 in decimal)  →  ternary index 3
    /// binary index 0b01  (1 in decimal)  →  ternary index 1
    /// binary index 0b00  (0 in decimal)  →  ternary index 0
    /// ```
    fn binary_to_ternary_index(&self, mut binary_index: usize) -> usize {
        let mut ternary_index = 0;
        let mut factor = 1;

        for _ in 0..self.n_variables {
            ternary_index += (binary_index & 1) * factor;
            // Move to next bit
            binary_index >>= 1;
            // Increase ternary place value
            factor *= 3;
        }

        ternary_index
    }

    /// Evaluates the polynomial at an arbitrary point in the domain `{0,1,2}^n`.
    ///
    /// Given an interpolation point `point ∈ F^n`, this computes:
    /// ```ignore
    /// f(point) = ∑ evaluations[i] * eq_poly3(i)
    /// ```
    /// where `eq_poly3(i)` is the Lagrange basis polynomial at index `i` in `{0,1,2}^n`.
    ///
    /// This allows evaluating the polynomial at non-discrete inputs beyond `{0,1,2}^n`.
    ///
    /// # Constraints:
    /// - The input `point` must have `n_variables` dimensions.
    #[must_use]
    #[inline]
    pub fn evaluate_at_point(&self, point: &MultilinearPoint<F>) -> F {
        assert_eq!(point.num_variables(), self.n_variables);
        self.evaluations
            .iter()
            .enumerate()
            .map(|(i, &eval)| eval * point.eq_poly3(i))
            .sum()
    }
}
