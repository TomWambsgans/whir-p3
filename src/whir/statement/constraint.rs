use p3_field::Field;

use crate::{poly::evals::EvaluationsList, whir::statement::weights::Weights};

/// Represents a single constraint in a polynomial statement.
#[derive(Clone, Debug)]
pub struct Constraint<F> {
    /// The weight function applied to the polynomial.
    ///
    /// This defines how the polynomial is combined or evaluated.
    /// It can represent either a point evaluation or a full set of weights.
    pub weights: Weights<F>,

    /// The expected result of applying the weight to the polynomial.
    ///
    /// This is the scalar value that the weighted sum must match.
    pub sum: F,

    /// If true, the verifier will not evaluate the weight directly.
    ///
    /// Instead, the prover must supply the result as a hint.
    /// This is used for deferred or externally computed evaluations.
    pub defer_evaluation: bool,
}

impl<F: Field> Constraint<F> {
    /// Verify if a polynomial (in coefficient form) satisfies the constraint.
    #[must_use]
    pub fn verify(&self, poly: &EvaluationsList<F>) -> bool {
        self.weights.evaluate_coeffs(poly) == self.sum
    }
}