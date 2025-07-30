use std::{
    collections::HashSet,
    ops::{Add, AddAssign, Mul, MulAssign},
};

use p3_field::{ExtensionField, Field};
use rand::distr::{Distribution, StandardUniform};

/// A univariate polynomial represented in coefficient form.
///
/// The coefficient of `x^i` is stored at index `i`.
///
/// Designed for verifier use: avoids parallelism by enforcing sequential Horner evaluation.
/// The verifier should be run on a cheap device.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Default)]
pub struct WhirDensePolynomial<F: Field> {
    /// The coefficient of `x^i` is stored at location `i` in `self.coeffs`.
    pub coeffs: Vec<F>,
}

impl<F: Field> WhirDensePolynomial<F> {
    /// Constructs a new polynomial from a list of coefficients.
    #[must_use]
    pub fn from_coefficients_slice(coeffs: &[F]) -> Self {
        Self::from_coefficients_vec(coeffs.to_vec())
    }

    /// Constructs a new polynomial from a list of coefficients.
    #[must_use]
    pub const fn from_coefficients_vec(coeffs: Vec<F>) -> Self {
        Self { coeffs }
    }

    /// Removes trailing zero coefficients from the polynomial's coefficient vector.
    fn truncate_leading_zeros(&mut self) {
        while self.coeffs.last().is_some_and(Field::is_zero) {
            self.coeffs.pop();
        }
    }

    /// Checks if the given polynomial is zero.
    fn is_zero(&self) -> bool {
        self.coeffs.is_empty() || self.coeffs.iter().all(Field::is_zero)
    }

    /// Evaluates `self` at the given `point` in `Self::Point`.
    pub fn evaluate<EF: ExtensionField<F>>(&self, point: EF) -> EF {
        if self.is_zero() {
            return EF::ZERO;
        } else if point.is_zero() {
            return EF::from(self.coeffs[0]);
        }
        self.horner_evaluate(point)
    }

    // Horner's method for polynomial evaluation
    fn horner_evaluate<EF: ExtensionField<F>>(&self, point: EF) -> EF {
        self.coeffs
            .iter()
            .rfold(EF::ZERO, move |result, coeff| result * point + *coeff)
    }

    /// Constructs a new polynomial from a list of coefficients
    ///  which are random elements mapped by closed interval [0, degree]
    pub fn random<R: rand::Rng>(rng: &mut R, degree: usize) -> Self
    where
        StandardUniform: Distribution<F>,
    {
        Self::from_coefficients_vec((0..=degree).map(|_| rng.random()).collect())
    }

    /// Constructs the unique interpolating polynomial `P(x)` such that:
    ///
    /// \begin{equation}
    /// P(x_i) = y_i \quad \text{for each } (x_i, y_i) \text{ in the input set}
    /// \end{equation}
    ///
    /// The result is a dense univariate polynomial of degree at most `n - 1`, where `n` is the number of input points.
    ///
    /// # Parameters
    ///
    /// - `values`: A slice of `(x_i, y_i)` pairs, where each `x_i` is a field element from `S`,
    ///   and each `y_i` is an extension field element from `F`.
    ///
    /// # Returns
    ///
    /// - `Some(P)`: The interpolating polynomial if all `x_i` are distinct.
    /// - `None`: If any duplicate `x_i` values exist (even with equal `y_i`).
    pub fn lagrange_interpolation<S>(values: &[(S, F)]) -> Option<Self>
    where
        S: Field,
        F: ExtensionField<S>,
    {
        // The number of (x, y) pairs to interpolate.
        let n = values.len();

        // Special case: no points => return the zero polynomial.
        if n == 0 {
            return Some(Self::default());
        }

        // Check for duplicate x-coordinates
        //
        // We use a HashSet to track x_i values. If any duplicates exist, we cannot interpolate.
        let mut unique_x = HashSet::with_capacity(n);
        if !values.iter().all(|(x, _)| unique_x.insert(*x)) {
            return None; // Found a duplicate x_i
        }

        // Initialize result and basis polynomials

        // The result polynomial P(x) starts at zero and is updated iteratively.
        let mut result_poly = Self::default();

        // The basis polynomial B(x) starts at 1.
        // After i steps, B(x) = (x - x_0)(x - x_1)...(x - x_{i-1})
        let mut basis_poly = Self::from_coefficients_vec(vec![F::ONE]);

        // Newton-style interpolation loop
        for (x_i, y_i) in values.iter().take(n) {
            // Promote x_i to the extension field for evaluation
            let x_i_ext = F::from(*x_i);

            // Compute current prediction: P(x_i)
            let current_y = result_poly.evaluate(x_i_ext);

            // Compute the discrepancy: how far off our polynomial is
            let delta = *y_i - current_y;

            // Compute B(x_i), the value of the basis at x_i
            let basis_eval = basis_poly.evaluate(x_i_ext);

            // The scalar coefficient c_i is the correction needed:
            // c_i = (y_i - P(x_i)) / B(x_i)
            let c_i = delta
                * basis_eval
                    .try_inverse()
                    .expect("x_i was checked to be unique");

            // Form term = c_i · B(x)

            // Multiply all coefficients of B(x) by c_i
            let mut term_coeffs = basis_poly.coeffs.clone();
            for coeff in &mut term_coeffs {
                *coeff *= c_i;
            }

            // Convert the coefficient vector into a polynomial
            let term = Self::from_coefficients_vec(term_coeffs);

            // Add the term to the result: P(x) := P(x) + c_i · B(x)
            result_poly += &term;

            // Update B(x) := B(x) · (x - x_i)

            // Monomial: (x - x_i) = -x_i + x = [-x_i, 1]
            let monomial = Self::from_coefficients_slice(&[-x_i_ext, F::ONE]);

            // Multiply current basis polynomial by the monomial
            basis_poly = &basis_poly * &monomial;
        }

        Some(result_poly)
    }
}

impl<F: Field> Add for &WhirDensePolynomial<F> {
    type Output = WhirDensePolynomial<F>;

    /// Adds two dense polynomials and returns the resulting polynomial.
    ///
    /// This function computes the sum of `self` and `other` by adding their
    /// coefficients term by term. If the polynomials have different lengths,
    /// the coefficients of the longer polynomial that do not have a corresponding
    /// term in the shorter polynomial are left unchanged in the result.
    ///
    /// # Arguments
    ///
    /// * `other` - The polynomial to add to `self`.
    ///
    /// # Returns
    ///
    /// A new `WhirDensePolynomial<F>` representing the sum of the two input polynomials.
    fn add(self, other: Self) -> WhirDensePolynomial<F> {
        let (big, small) = if self.coeffs.len() >= other.coeffs.len() {
            (self, other)
        } else {
            (other, self)
        };
        let mut sum = big.coeffs.clone();
        for (i, coeff) in small.coeffs.iter().enumerate() {
            sum[i] += *coeff;
        }
        WhirDensePolynomial::from_coefficients_vec(sum)
    }
}

impl<F: Field> AddAssign<&Self> for WhirDensePolynomial<F> {
    fn add_assign(&mut self, other: &Self) {
        *self = &*self + other;
    }
}

impl<F: Field> Mul for &WhirDensePolynomial<F> {
    type Output = WhirDensePolynomial<F>;

    /// Multiplies two dense polynomials and returns the resulting polynomial.
    ///
    /// This function computes the product of `self` and `other` using the standard
    /// schoolbook (naive) polynomial multiplication algorithm. If either polynomial
    /// is zero, the result is the zero polynomial. The resulting polynomial's
    /// coefficients are computed by summing the products of all pairs of coefficients
    /// whose degrees add up to the same value.
    ///
    /// # Arguments
    ///
    /// * `other` - The polynomial to multiply with `self`.
    ///
    /// # Returns
    ///
    /// A new `WhirDensePolynomial<F>` representing the product of the two input polynomials.
    fn mul(self, other: Self) -> WhirDensePolynomial<F> {
        if self.is_zero() || other.is_zero() {
            return WhirDensePolynomial::default();
        }
        let mut prod = vec![F::ZERO; self.coeffs.len() + other.coeffs.len() - 1];
        for i in 0..self.coeffs.len() {
            for j in 0..other.coeffs.len() {
                prod[i + j] += self.coeffs[i] * other.coeffs[j];
            }
        }
        WhirDensePolynomial::from_coefficients_vec(prod)
    }
}

impl<F: Field> MulAssign<&Self> for WhirDensePolynomial<F> {
    fn mul_assign(&mut self, other: &Self) {
        *self = &*self * other;
    }
}
