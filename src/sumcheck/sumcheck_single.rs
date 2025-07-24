use p3_challenger::{FieldChallenger, GrindingChallenger};
use p3_field::{ExtensionField, Field, TwoAdicField};
use p3_interpolation::interpolate_subgroup;
use p3_matrix::dense::RowMajorMatrix;
#[cfg(feature = "parallel")]
use rayon::prelude::*;
use tracing::instrument;

use super::sumcheck_polynomial::SumcheckPolynomial;
use crate::{
    fiat_shamir::{errors::ProofResult, prover::ProverState},
    poly::{
        coeffs::CoefficientList,
        dense::WhirDensePolynomial,
        evals::{EvaluationStorage, EvaluationsList},
        multilinear::MultilinearPoint,
    },
    sumcheck::utils::sumcheck_quadratic,
    whir::statement::Statement,
};

/// Implements the single-round sumcheck protocol for verifying a multilinear polynomial evaluation.
///
/// This struct is responsible for:
/// - Transforming a polynomial from coefficient representation into evaluation form.
/// - Constructing and evaluating weighted constraints.
/// - Computing the sumcheck polynomial, which is a quadratic polynomial in a single variable.
///
/// Given a multilinear polynomial `p(X1, ..., Xn)`, the sumcheck polynomial is computed as:
///
/// \begin{equation}
/// h(X) = \sum_b p(b, X) \cdot w(b, X)
/// \end{equation}
///
/// where:
/// - `b` ranges over evaluation points in `{0,1,2}^k` (with `k=1` in this implementation).
/// - `w(b, X)` represents generic weights applied to `p(b, X)`.
/// - The result `h(X)` is a quadratic polynomial in `X`.
///
/// The sumcheck protocol ensures that the claimed sum is correct.
#[derive(Debug, Clone)]
pub struct SumcheckSingle<F, EF> {
    /// Evaluations of the polynomial `p(X)`.
    pub(crate) evaluation_of_p: EvaluationStorage<F, EF>,
    /// Evaluations of the equality polynomial used for enforcing constraints.
    pub(crate) weights: EvaluationsList<EF>,
    /// Accumulated sum incorporating equality constraints.
    pub(crate) sum: EF,
    /// Marker for phantom type parameter `F`.
    phantom: std::marker::PhantomData<F>,
}

impl<F, EF> SumcheckSingle<F, EF>
where
    F: Field,
    EF: ExtensionField<F>,
{
    /// Constructs a new `SumcheckSingle` instance from polynomial coefficients in base field.
    ///
    /// This function:
    /// - Converts `coeffs` into evaluation form.
    /// - Initializes an empty constraint table.
    /// - Applies weighted constraints if provided.
    ///
    /// The provided `Statement` encodes constraints that contribute to the final sumcheck equation.
    pub fn from_base_coeffs(
        coeffs: CoefficientList<F>,
        statement: &Statement<EF>,
        combination_randomness: EF,
    ) -> Self {
        let (weights, sum) = statement.combine::<F>(combination_randomness);
        Self {
            evaluation_of_p: EvaluationStorage::Base(coeffs.to_evaluations()),
            weights,
            sum,
            phantom: std::marker::PhantomData,
        }
    }

    /// Constructs a new `SumcheckSingle` instance from evaluations in the base field.
    ///
    /// This function:
    /// - Uses precomputed evaluations of the polynomial `p` over the Boolean hypercube.
    /// - Applies the given constraint `Statement` using a random linear combination.
    /// - Initializes internal sumcheck state with weights and expected sum.
    ///
    /// The base field evaluations are stored without transformation.
    #[instrument(skip_all)]
    pub fn from_base_evals(
        evals: EvaluationsList<F>,
        statement: &Statement<EF>,
        combination_randomness: EF,
    ) -> Self {
        let (weights, sum) = statement.combine::<F>(combination_randomness);
        Self {
            evaluation_of_p: EvaluationStorage::Base(evals),
            weights,
            sum,
            phantom: std::marker::PhantomData,
        }
    }

    /// Constructs a new `SumcheckSingle` instance from polynomial coefficients in extension field.
    ///
    /// This function:
    /// - Converts `coeffs` into evaluation form.
    /// - Initializes an empty constraint table.
    /// - Applies weighted constraints if provided.
    ///
    /// The provided `Statement` encodes constraints that contribute to the final sumcheck equation.
    pub fn from_extension_coeffs(
        coeffs: CoefficientList<EF>,
        statement: &Statement<EF>,
        combination_randomness: EF,
    ) -> Self {
        let (weights, sum) = statement.combine::<F>(combination_randomness);
        Self {
            evaluation_of_p: EvaluationStorage::Extension(coeffs.to_evaluations::<F>()),
            weights,
            sum,
            phantom: std::marker::PhantomData,
        }
    }

    /// Constructs a new `SumcheckSingle` instance from evaluations in the extension field.
    ///
    /// This function:
    /// - Uses precomputed evaluations of the polynomial `p` over the Boolean hypercube,
    ///   where `p` is already represented over the extension field `EF`.
    /// - Applies the provided `Statement` to compute equality weights and the expected sum.
    /// - Initializes the internal state used in the sumcheck protocol.
    ///
    /// This is the entry point when the polynomial is defined directly over `EF`.
    pub fn from_extension_evals(
        evals: EvaluationsList<EF>,
        statement: &Statement<EF>,
        combination_randomness: EF,
    ) -> Self {
        let (weights, sum) = statement.combine::<F>(combination_randomness);
        Self {
            evaluation_of_p: EvaluationStorage::Extension(evals),
            weights,
            sum,
            phantom: std::marker::PhantomData,
        }
    }

    /// Returns the number of variables in the polynomial.
    pub const fn num_variables(&self) -> usize {
        self.evaluation_of_p.num_variables()
    }

    /// Adds new weighted constraints to the polynomial.
    ///
    /// This function updates the weight evaluations and sum by incorporating new constraints.
    ///
    /// Given points `z_i`, weights `ε_i`, and evaluation values `f(z_i)`, it updates:
    ///
    /// \begin{equation}
    ///     w(X) = w(X) + \sum ε_i \cdot w_{z_i}(X)
    /// \end{equation}
    ///
    /// and updates the sum as:
    ///
    /// \begin{equation}
    ///     S = S + \sum ε_i \cdot f(z_i)
    /// \end{equation}
    ///
    /// where `w_{z_i}(X)` represents the constraint encoding at point `z_i`.
    #[instrument(skip_all, fields(
        num_points = points.len(),
    ))]
    pub fn add_new_equality(
        &mut self,
        points: &[MultilinearPoint<EF>],
        evaluations: &[EF],
        combination_randomness: &[EF],
    ) {
        assert_eq!(combination_randomness.len(), points.len());
        assert_eq!(evaluations.len(), points.len());

        #[cfg(feature = "parallel")]
        {
            use tracing::info_span;

            // Parallel update of weight buffer
            info_span!("accumulate_weight_buffer").in_scope(|| {
                points
                    .iter()
                    .zip(combination_randomness.iter())
                    .for_each(|(point, &rand)| {
                        crate::utils::eval_eq::<_, _, true>(
                            &point.0,
                            self.weights.evals_mut(),
                            rand,
                        );
                    });
            });

            // Accumulate the weighted sum (cheap, done sequentially)
            self.sum += combination_randomness
                .iter()
                .zip(evaluations.iter())
                .map(|(&rand, &eval)| rand * eval)
                .sum::<EF>();
        }

        #[cfg(not(feature = "parallel"))]
        {
            // Accumulate the sum while applying all constraints simultaneously
            points
                .iter()
                .zip(combination_randomness.iter().zip(evaluations.iter()))
                .for_each(|(point, (&rand, &eval))| {
                    crate::utils::eval_eq::<F, EF, true>(&point.0, self.weights.evals_mut(), rand);
                    self.sum += rand * eval;
                });
        }
    }

    /// Computes the sumcheck polynomial `h(X)`, a quadratic polynomial resulting from the folding step.
    ///
    /// The sumcheck polynomial is computed as:
    ///
    /// \[
    /// h(X) = \sum_b p(b, X) \cdot w(b, X)
    /// \]
    ///
    /// where:
    /// - `b` ranges over evaluation points in `{0,1,2}^1` (i.e., two points per fold).
    /// - `p(b, X)` is the polynomial evaluation at `b` as a function of `X`.
    /// - `w(b, X)` is the associated weight applied at `b` as a function of `X`.
    ///
    /// **Mathematical model:**
    /// - Each chunk of two evaluations encodes a linear polynomial in `X`.
    /// - The product `p(X) * w(X)` is a quadratic polynomial.
    /// - We compute the constant and quadratic coefficients first, then infer the linear coefficient using:
    ///
    /// \[
    /// \text{sum} = 2 \cdot c_0 + c_1 + c_2
    /// \]
    ///
    /// where `sum` is the accumulated constraint sum.
    ///
    /// Returns a `SumcheckPolynomial` with evaluations at `X = 0, 1, 2`.
    #[instrument(skip_all, level = "debug")]
    pub fn compute_sumcheck_polynomial(&self) -> SumcheckPolynomial<EF> {
        assert!(self.num_variables() >= 1);

        #[cfg(feature = "parallel")]
        let (c0, c2) = match &self.evaluation_of_p {
            EvaluationStorage::Base(evals_f) => evals_f
                .evals()
                .par_chunks_exact(2)
                .zip(self.weights.evals().par_chunks_exact(2))
                .map(sumcheck_quadratic::<F, EF>)
                .reduce(
                    || (EF::ZERO, EF::ZERO),
                    |(a0, a2), (b0, b2)| (a0 + b0, a2 + b2),
                ),
            EvaluationStorage::Extension(evals_ef) => evals_ef
                .evals()
                .par_chunks_exact(2)
                .zip(self.weights.evals().par_chunks_exact(2))
                .map(sumcheck_quadratic::<EF, EF>)
                .reduce(
                    || (EF::ZERO, EF::ZERO),
                    |(a0, a2), (b0, b2)| (a0 + b0, a2 + b2),
                ),
        };

        #[cfg(not(feature = "parallel"))]
        let (c0, c2) = match &self.evaluation_of_p {
            EvaluationStorage::Base(evals_f) => evals_f
                .evals()
                .chunks_exact(2)
                .zip(self.weights.evals().chunks_exact(2))
                .map(sumcheck_quadratic::<F, EF>)
                .fold((EF::ZERO, EF::ZERO), |(a0, a2), (b0, b2)| {
                    (a0 + b0, a2 + b2)
                }),

            EvaluationStorage::Extension(evals_ef) => evals_ef
                .evals()
                .chunks_exact(2)
                .zip(self.weights.evals().chunks_exact(2))
                .map(sumcheck_quadratic::<EF, EF>)
                .fold((EF::ZERO, EF::ZERO), |(a0, a2), (b0, b2)| {
                    (a0 + b0, a2 + b2)
                }),
        };

        // Compute the middle (linear) coefficient
        //
        // The quadratic polynomial h(X) has the form:
        //     h(X) = c0 + c1 * X + c2 * X^2
        //
        // We already computed:
        // - c0: the constant coefficient (contribution at X=0)
        // - c2: the quadratic coefficient (contribution at X^2)
        //
        // To recover c1 (linear term), we use the known sum rule:
        //     sum = h(0) + h(1)
        // Expand h(0) and h(1):
        //     h(0) = c0
        //     h(1) = c0 + c1 + c2
        // Therefore:
        //     sum = c0 + (c0 + c1 + c2) = 2*c0 + c1 + c2
        //
        // Rearranging for c1 gives:
        //     c1 = sum - 2*c0 - c2
        let c1 = self.sum - c0.double() - c2;

        // Evaluate the quadratic polynomial at points 0, 1, 2
        //
        // Evaluate:
        //     h(0) = c0
        //     h(1) = c0 + c1 + c2
        //     h(2) = c0 + 2*c1 + 4*c2
        //
        // To compute h(2) efficiently, observe:
        //     h(2) = h(1) + (c1 + 2*c2)
        let eval_0 = c0;
        let eval_1 = c0 + c1 + c2;
        let eval_2 = eval_1 + c1 + c2 + c2.double();

        SumcheckPolynomial::new(vec![eval_0, eval_1, eval_2], 1)
    }

    /// Executes the sumcheck protocol for a multilinear polynomial with optional **univariate skip**.
    ///
    /// This function performs `folding_factor` rounds of the sumcheck protocol:
    ///
    /// - At each round, a univariate polynomial is sent representing a partial sum over a subset of variables.
    /// - The verifier responds with a random challenge that is used to fix one variable.
    /// - Optionally, the first `k` rounds can be skipped using the **univariate skip** optimization,
    ///   which collapses multiple Boolean variables at once over a multiplicative subgroup.
    ///
    /// The univariate skip is performed entirely in the base field and reduces expensive extension field
    /// computations, improving prover efficiency.
    ///
    /// # Arguments
    /// - `prover_state`: The state of the prover, managing Fiat-Shamir transcript and PoW grinding.
    /// - `folding_factor`: Number of variables to fold in total.
    /// - `pow_bits`: Number of PoW bits used to delay the prover (0.0 to disable).
    /// - `k_skip`: Optional number of initial variables to skip using the univariate optimization.
    /// - `dft`: A two-adic FFT backend used for low-degree extensions over cosets.
    ///
    /// # Returns
    /// A `MultilinearPoint<EF>` representing the verifier's challenges across all folded variables.
    ///
    /// # Panics
    /// - If `folding_factor > num_variables()`
    /// - If univariate skip is attempted with evaluations in the extension field.
    #[instrument(skip_all)]
    pub fn compute_sumcheck_polynomials<Challenger>(
        &mut self,
        prover_state: &mut ProverState<F, EF, Challenger>,
        folding_factor: usize,
        pow_bits: usize,
        k_skip: Option<usize>,
    ) -> ProofResult<MultilinearPoint<EF>>
    where
        F: TwoAdicField,
        EF: TwoAdicField,
        Challenger: FieldChallenger<F> + GrindingChallenger<Witness = F>,
    {
        // Will store the verifier's folding challenges for each round.
        let mut res = Vec::with_capacity(folding_factor);

        // Track number of rounds already skipped.
        let mut skip = 0;

        // Optional univariate skip
        if let Some(k) = k_skip {
            if k >= 2 && k <= folding_factor {
                // Collapse the first k variables via a univariate evaluation over a multiplicative coset.
                let (sumcheck_poly, f_mat, w_mat) = self.compute_skipping_sumcheck_polynomial(k);

                prover_state.add_extension_scalars(sumcheck_poly.evaluations());

                // Receive the verifier challenge for this entire collapsed round.
                let folding_randomness: EF = prover_state.sample();
                res.push(folding_randomness);

                // Proof-of-work challenge to delay prover.
                prover_state.pow_grinding(pow_bits);

                // Interpolate the LDE matrices at the folding randomness to get the new "folded" polynomial state.
                let new_p = interpolate_subgroup(&f_mat, folding_randomness);
                let new_w = interpolate_subgroup(&w_mat, folding_randomness);

                // Update polynomial and weights with reduced dimensionality.
                self.evaluation_of_p = EvaluationStorage::Extension(EvaluationsList::new(new_p));
                self.weights = EvaluationsList::new(new_w);

                // Compute the new target sum after folding.
                let folded_poly_eval = interpolate_subgroup(
                    &RowMajorMatrix::new_col(sumcheck_poly.evaluations().to_vec()),
                    folding_randomness,
                );
                self.sum = folded_poly_eval[0];

                // We've skipped `k` variables with one univariate round.
                skip = k;
            }
        }

        // Standard round-by-round folding
        // Proceed with one-variable-per-round folding for remaining variables.
        for _ in skip..folding_factor {
            // Compute the quadratic sumcheck polynomial for the current variable.
            let sumcheck_poly = self.compute_sumcheck_polynomial();

            let sumcheck_poly_normal = WhirDensePolynomial::lagrange_interpolation(&
                sumcheck_poly
                    .evaluations()
                    .iter()
                    .enumerate()
                    .map(|(i, &v)| (F::from_usize(i), v))
                    .collect::<Vec<_>>(),
            ).unwrap();

            prover_state.add_extension_scalars(&sumcheck_poly_normal.coeffs);

            // Sample verifier challenge.
            let folding_randomness: EF = prover_state.sample();
            res.push(folding_randomness);

            prover_state.pow_grinding(pow_bits);

            // Fold the polynomial and weight evaluations over the new challenge.
            self.compress(EF::ONE, &folding_randomness.into(), &sumcheck_poly);
        }

        // Reverse challenges to maintain order from X₀ to Xₙ.
        res.reverse();

        // Return the full vector of verifier challenges as a multilinear point.
        Ok(MultilinearPoint(res))
    }

    /// Compresses the polynomial and weight evaluations by reducing the number of variables.
    ///
    /// Given a multilinear polynomial `p(X1, ..., Xn)`, this function eliminates `X1` using the
    /// folding randomness `r`:
    /// \begin{equation}
    ///     p'(X_2, ..., X_n) = (p(1, X_2, ..., X_n) - p(0, X_2, ..., X_n)) \cdot r
    ///     + p(0, X_2, ...,X_n)
    /// \end{equation}
    ///
    /// The same transformation applies to the weights `w(X)`, and the sum is updated as:
    ///
    /// \begin{equation}
    ///     S' = \rho \cdot h(r)
    /// \end{equation}
    ///
    /// where `h(r)` is the sumcheck polynomial evaluated at `r`, and `\rho` is
    /// `combination_randomness`.
    ///
    /// # Effects
    /// - Shrinks `p(X)` and `w(X)` by half.
    /// - Updates `sum` using `sumcheck_poly`.
    #[instrument(skip_all, fields(size = self.evaluation_of_p.num_variables()))]
    pub fn compress(
        &mut self,
        combination_randomness: EF, // Scale the initial point
        folding_randomness: &MultilinearPoint<EF>,
        sumcheck_poly: &SumcheckPolynomial<EF>,
    ) {
        assert_eq!(folding_randomness.num_variables(), 1);
        assert!(self.num_variables() >= 1);

        let randomness = folding_randomness.0[0];

        // Fold between extension field elements
        let fold_extension = |slice: &[EF]| -> EF { randomness * (slice[1] - slice[0]) + slice[0] };
        // Fold between base and extension field elements
        let fold_base = |slice: &[F]| -> EF { randomness * (slice[1] - slice[0]) + slice[0] };

        #[cfg(feature = "parallel")]
        let (evaluations_of_p, evaluations_of_eq) = {
            // Threshold below which sequential computation is faster
            //
            // This was chosen based on experiments with the `compress` function.
            // It is possible that the threshold can be tuned further.
            const PARALLEL_THRESHOLD: usize = 4096;

            match &self.evaluation_of_p {
                EvaluationStorage::Base(evals_f) => {
                    if evals_f.evals().len() >= PARALLEL_THRESHOLD
                        && self.weights.evals().len() >= PARALLEL_THRESHOLD
                    {
                        rayon::join(
                            || evals_f.evals().par_chunks_exact(2).map(fold_base).collect(),
                            || {
                                self.weights
                                    .evals()
                                    .par_chunks_exact(2)
                                    .map(fold_extension)
                                    .collect()
                            },
                        )
                    } else {
                        (
                            evals_f.evals().chunks_exact(2).map(fold_base).collect(),
                            self.weights
                                .evals()
                                .chunks_exact(2)
                                .map(fold_extension)
                                .collect(),
                        )
                    }
                }
                EvaluationStorage::Extension(evals_ef) => {
                    if evals_ef.evals().len() >= PARALLEL_THRESHOLD
                        && self.weights.evals().len() >= PARALLEL_THRESHOLD
                    {
                        rayon::join(
                            || {
                                evals_ef
                                    .evals()
                                    .par_chunks_exact(2)
                                    .map(fold_extension)
                                    .collect()
                            },
                            || {
                                self.weights
                                    .evals()
                                    .par_chunks_exact(2)
                                    .map(fold_extension)
                                    .collect()
                            },
                        )
                    } else {
                        (
                            evals_ef
                                .evals()
                                .chunks_exact(2)
                                .map(fold_extension)
                                .collect(),
                            self.weights
                                .evals()
                                .chunks_exact(2)
                                .map(fold_extension)
                                .collect(),
                        )
                    }
                }
            }
        };

        #[cfg(not(feature = "parallel"))]
        let (evaluations_of_p, evaluations_of_eq) = match &self.evaluation_of_p {
            EvaluationStorage::Base(evals_f) => (
                evals_f.evals().chunks_exact(2).map(fold_base).collect(),
                self.weights
                    .evals()
                    .chunks_exact(2)
                    .map(fold_extension)
                    .collect(),
            ),
            EvaluationStorage::Extension(evals_ef) => (
                evals_ef
                    .evals()
                    .chunks_exact(2)
                    .map(fold_extension)
                    .collect(),
                self.weights
                    .evals()
                    .chunks_exact(2)
                    .map(fold_extension)
                    .collect(),
            ),
        };

        // Update internal state
        self.evaluation_of_p = EvaluationStorage::Extension(EvaluationsList::new(evaluations_of_p));
        self.weights = EvaluationsList::new(evaluations_of_eq);
        self.sum = combination_randomness * sumcheck_poly.evaluate_at_point(folding_randomness);
    }
}
