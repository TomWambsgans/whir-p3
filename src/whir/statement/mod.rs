use p3_field::{ExtensionField, Field};
use tracing::instrument;

use crate::{
    poly::multilinear::MultilinearPoint,
    utils::{compute_eval_eq, uninitialized_vec},
    whir::statement::constraint::Constraint,
};

pub mod constraint;

/// Represents a system of weighted polynomial constraints over a Boolean hypercube.
///
/// A `Statement<F>` consists of multiple constraints, each enforcing a relationship of the form:
///
/// \begin{equation}
/// \sum_{x \in \{0,1\}^n} w_i(x) \cdot p(x) = s_i
/// \end{equation}
///
/// where:
/// - `w_i(x)` is a weight function, either a point evaluation (equality constraint) or a full set of weights.
/// - `p(x)` is a multilinear polynomial over $\{0,1\}^n$ in evaluation form.
/// - `s_i` is the expected sum for the $i$-th constraint.
///
/// These constraints can be combined into a single constraint using a random challenge $\gamma$:
///
/// \begin{equation}
/// W(x) = w_1(x) + \gamma w_2(x) + \gamma^2 w_3(x) + \dots + \gamma^{k-1} w_k(x)
/// \end{equation}
///
/// with a combined expected sum:
///
/// \begin{equation}
/// S = s_1 + \gamma s_2 + \gamma^2 s_3 + \dots + \gamma^{k-1} s_k
/// \end{equation}
///
/// This combined form is used in protocols like sumcheck and zerocheck to reduce many constraints to one.
#[derive(Clone, Debug)]
pub struct Statement<F> {
    /// Number of variables in the multilinear polynomial space (log₂ of evaluation length).
    num_variables: usize,

    /// List of constraints, each pairing a weight function with a target expected sum.
    ///
    /// The weight may be either a concrete evaluation point (enforcing `p(z) = s`)
    /// or a full evaluation vector of weights `w(x)` (enforcing a weighted sum).
    pub constraints: Vec<Constraint<F>>,
}

impl<F: Field> Statement<F> {
    /// Creates an empty `Statement<F>` for polynomials with `num_variables` variables.
    #[must_use]
    pub const fn new(num_variables: usize) -> Self {
        Self {
            num_variables,
            constraints: Vec::new(),
        }
    }

    /// Returns the number of variables defining the polynomial space.
    #[must_use]
    pub const fn num_variables(&self) -> usize {
        self.num_variables
    }

    /// Adds a constraint `(w(X), s)` to the system.
    ///
    /// **Precondition:**
    /// The number of variables in `w(X)` must match `self.num_variables`.
    pub fn add_constraint(&mut self, weights: MultilinearPoint<F>, sum: F) {
        assert_eq!(weights.num_variables(), self.num_variables());
        self.constraints.push(Constraint { weights, sum });
    }

    /// Inserts a constraint `(w(X), s)` at the front of the system.
    pub fn add_constraint_in_front(&mut self, weights: MultilinearPoint<F>, sum: F) {
        assert_eq!(weights.num_variables(), self.num_variables());
        self.constraints.insert(0, Constraint { weights, sum });
    }

    /// Inserts multiple constraints at the front of the system.
    ///
    /// Panics if any constraint's number of variables does not match the system.
    pub fn add_constraints_in_front(&mut self, constraints: Vec<(MultilinearPoint<F>, F)>) {
        // Store the number of variables expected by this statement.
        let n = self.num_variables();

        // Preallocate a vector for the converted constraints to avoid reallocations.
        let mut new_constraints = Vec::with_capacity(constraints.len());

        // Iterate through each (weights, sum) pair in the input.
        for (weights, sum) in constraints {
            // Ensure the number of variables in the weight matches the statement.
            assert_eq!(weights.num_variables(), n);

            // Convert the pair into a full `Constraint` with `defer_evaluation = false`.
            new_constraints.push(Constraint { weights, sum });
        }

        // Insert all new constraints at the beginning of the existing list.
        self.constraints.splice(0..0, new_constraints);
    }

    /// Combines all constraints into a single aggregated polynomial using a challenge.
    ///
    /// Given a random challenge $\gamma$, the new polynomial is:
    ///
    /// \begin{equation}
    /// W(X) = w_1(X) + \gamma w_2(X) + \gamma^2 w_3(X) + \dots + \gamma^{k-1} w_k(X)
    /// \end{equation}
    ///
    /// with the combined sum:
    ///
    /// \begin{equation}
    /// S = s_1 + \gamma s_2 + \gamma^2 s_3 + \dots + \gamma^{k-1} s_k
    /// \end{equation}
    ///
    /// **Returns:**
    /// - `EvaluationsList<F>`: The combined polynomial `W(X)`.
    /// - `F`: The combined sum `S`.
    #[instrument(skip_all)]
    pub fn combine<Base>(&self, challenge: F) -> (Vec<F>, F)
    where
        Base: Field,
        F: ExtensionField<Base>,
    {
        if self.constraints.is_empty() {
            return (F::zero_vec(1 << self.num_variables), F::ZERO);
        }
        // Alloc memory without initializing it to zero.
        // This is safe because there is at least one constraint (otherwise it would return early),
        // and the first iteration of the loop will overwrite the entire vector.
        let mut combined_evals = unsafe { uninitialized_vec::<F>(1 << self.num_variables) };
        let (combined_sum, _) = self.constraints.iter().enumerate().fold(
            (F::ZERO, F::ONE),
            |(mut acc_sum, gamma_pow), (i, constraint)| {
                if i == 0 {
                    // first iteration: combined_evals must be overwritten
                    compute_eval_eq::<Base, F, false>(
                        &constraint.weights,
                        &mut combined_evals,
                        gamma_pow,
                    );
                } else {
                    compute_eval_eq::<Base, F, true>(
                        &constraint.weights,
                        &mut combined_evals,
                        gamma_pow,
                    );
                }
                acc_sum += constraint.sum * gamma_pow;
                (acc_sum, gamma_pow * challenge)
            },
        );

        (combined_evals, combined_sum)
    }
}
