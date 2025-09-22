use p3_field::{ExtensionField, Field};
use rayon::prelude::*;

use crate::{
    PF,
    fiat_shamir::{FSChallenger, prover::ProverState},
    poly::{
        dense::DensePolynomial,
        evals::EvaluationsList,
        multilinear::{Evaluation, MultilinearPoint},
    },
    utils::compute_sparse_eval_eq,
};

const PARALLEL_THRESHOLD: usize = 4096;

fn compress_ext<F: Field, EF: ExtensionField<F>>(evals: &[F], r: EF) -> Vec<EF> {
    assert_ne!(evals.num_variables(), 0);

    // Fold between base and extension field elements
    let fold = |slice: &[F]| -> EF { r * (slice[1] - slice[0]) + slice[0] };
    if evals.len() >= PARALLEL_THRESHOLD {
        evals.par_chunks_exact(2).map(fold).collect()
    } else {
        evals.chunks_exact(2).map(fold).collect()
    }
}

fn compress<F: Field>(evals: &mut Vec<F>, r: F) {
    assert_ne!(evals.num_variables(), 0);

    if evals.len() >= PARALLEL_THRESHOLD {
        // Define the folding operation for a pair of elements.
        let fold = |slice: &[F]| -> F { r * (slice[1] - slice[0]) + slice[0] };
        // Execute the fold in parallel and collect into a new vector.
        let folded = evals.par_chunks_exact(2).map(fold).collect();
        // Replace the old evaluations with the new, folded evaluations.
        *evals = folded;
    } else {
        // For smaller inputs, we use the sequential, in-place strategy to save memory.
        let mid = evals.len() / 2;
        for i in 0..mid {
            let p0 = evals[2 * i];
            let p1 = evals[2 * i + 1];
            evals[i] = r * (p1 - p0) + p0;
        }
        evals.truncate(mid);
    }
}

fn initial_round<F: Field, EF: ExtensionField<F> + ExtensionField<PF<EF>>>(
    prover_state: &mut ProverState<PF<EF>, EF, impl FSChallenger<EF>>,
    evals: &[F],
    weights: &mut Vec<EF>,
    sum: &mut EF,
    _pow_bits: usize,
) -> (EF, Vec<EF>) {
    // Compute the quadratic sumcheck polynomial for the current variable.
    let sumcheck_poly = compute_sumcheck_polynomial(evals, weights, *sum);

    prover_state.add_extension_scalars(&sumcheck_poly.coeffs);

    // TODO: re-enable PoW grinding
    // prover_state.pow_grinding(pow_bits);

    // Sample verifier challenge.
    let r: EF = prover_state.sample();

    // Compress polynomials and update the sum.
    let evals = { rayon::join(|| compress(weights, r), || compress_ext(evals, r)).1 };

    *sum = sumcheck_poly.evaluate(r);

    (r, evals)
}

/// Executes a standard, intermediate round of the sumcheck protocol.
///
/// This function executes a standard, intermediate round of the sumcheck protocol. Unlike the initial round,
/// it operates entirely within the extension field `EF`. It computes the sumcheck polynomial from the
/// current evaluations and weights, adds it to the transcript, gets a new challenge from the verifier,
/// and then compresses both the polynomial and weight evaluations in-place.
///
/// ## Arguments
/// * `prover_state` - A mutable reference to the `ProverState`, managing the Fiat-Shamir transcript.
/// * `evals` - A mutable reference to the polynomial's evaluations in `EF`, which will be compressed.
/// * `weights` - A mutable reference to the weight evaluations in `EF`, which will also be compressed.
/// * `sum` - A mutable reference to the claimed sum, updated after folding.
/// * `pow_bits` - The number of proof-of-work bits for grinding.
///
/// ## Returns
/// The verifier's challenge `r` as an `EF` element.
fn round<F: Field, EF: ExtensionField<F> + ExtensionField<PF<EF>>>(
    prover_state: &mut ProverState<PF<EF>, EF, impl FSChallenger<EF>>,
    evals: &mut Vec<EF>,
    weights: &mut Vec<EF>,
    sum: &mut EF,
    _pow_bits: usize,
) -> EF {
    // Compute the quadratic sumcheck polynomial for the current variable.
    let sumcheck_poly = compute_sumcheck_polynomial(evals, weights, *sum);
    prover_state.add_extension_scalars(&sumcheck_poly.coeffs);

    // TODO: re-enable PoW grinding
    // prover_state.pow_grinding(pow_bits);

    // Sample verifier challenge.
    let r: EF = prover_state.sample();
    // Compress polynomials and update the sum.
    rayon::join(|| compress(evals, r), || compress(weights, r));

    *sum = sumcheck_poly.evaluate(r);

    r
}

pub(crate) fn compute_sumcheck_polynomial<F: Field, EF: ExtensionField<F>>(
    evals: &[F],
    weights: &Vec<EF>,
    sum: EF,
) -> DensePolynomial<EF> {
    assert!(evals.num_variables() >= 1);

    let (c0, c2) = evals
        .par_chunks_exact(2)
        .zip(weights.par_chunks_exact(2))
        .map(sumcheck_quadratic::<F, EF>)
        .reduce(
            || (EF::ZERO, EF::ZERO),
            |(a0, a2), (b0, b2)| (a0 + b0, a2 + b2),
        );

    let c1 = sum - c0.double() - c2;

    let eval_0 = c0;
    let eval_1 = c0 + c1 + c2;
    let eval_2 = eval_1 + c1 + c2 + c2.double();

    DensePolynomial::lagrange_interpolation(&[
        (F::ZERO, eval_0),
        (F::ONE, eval_1),
        (F::TWO, eval_2),
    ])
    .expect("Failed to interpolate sumcheck polynomial")
}

#[inline]
pub(crate) fn sumcheck_quadratic<F, EF>((p, eq): (&[F], &[EF])) -> (EF, EF)
where
    F: Field,
    EF: ExtensionField<F>,
{
    // Compute the constant coefficient:
    // p(0) * w(0)
    let constant = eq[0] * p[0];

    // Compute the quadratic coefficient:
    // (p(1) - p(0)) * (w(1) - w(0))
    let quadratic = (eq[1] - eq[0]) * (p[1] - p[0]);

    (constant, quadratic)
}

#[derive(Debug, Clone)]
pub(crate) struct SumcheckSingle<EF> {
    /// Evaluations of the polynomial `p(X)`.
    pub(crate) evals: Vec<EF>,
    /// Evaluations of the equality polynomial used for enforcing constraints.
    pub(crate) weights: Vec<EF>,
    /// Accumulated sum incorporating equality constraints.
    pub(crate) sum: EF,
}

impl<EF: Field + ExtensionField<PF<EF>>> SumcheckSingle<EF> {
    pub(crate) fn from_base_evals<F: Field>(
        evals: &[F],
        statement: &[Evaluation<EF>],
        combination_randomness: EF,
        prover_state: &mut ProverState<PF<EF>, EF, impl FSChallenger<EF>>,
        folding_factor: usize,
        pow_bits: usize,
        dot_product_statement: Option<(Vec<EF>, EF)>,
    ) -> (Self, MultilinearPoint<EF>)
    where
        EF: ExtensionField<F>,
    {
        assert_ne!(folding_factor, 0);
        let mut res = Vec::with_capacity(folding_factor);

        let (mut weights, mut sum) = combine_statement::<PF<EF>, EF>(
            statement,
            combination_randomness,
            dot_product_statement,
        );
        // In the first round base field evaluations are folded into extension field elements
        let (r, mut evals) = initial_round(prover_state, evals, &mut weights, &mut sum, pow_bits);
        res.push(r);

        // Apply rest of sumcheck rounds
        res.extend(
            (1..folding_factor).map(|_| {
                round::<F, EF>(prover_state, &mut evals, &mut weights, &mut sum, pow_bits)
            }),
        );

        res.reverse();

        let sumcheck = Self {
            evals,
            weights,
            sum,
        };

        (sumcheck, MultilinearPoint(res))
    }

    pub(crate) fn num_variables(&self) -> usize {
        self.evals.num_variables()
    }

    pub(crate) fn add_new_equality(
        &mut self,
        points: &[MultilinearPoint<EF>],
        evaluations: &[EF],
        combination_randomness: &[EF],
    ) {
        assert_eq!(combination_randomness.len(), points.len());
        assert_eq!(evaluations.len(), points.len());

        points
            .iter()
            .zip(combination_randomness.iter())
            .for_each(|(point, &rand)| {
                crate::utils::compute_eval_eq::<_, _, true>(point, &mut self.weights, rand);
            });

        self.sum += combination_randomness
            .iter()
            .zip(evaluations.iter())
            .map(|(&rand, &eval)| rand * eval)
            .sum::<EF>();
    }

    pub(crate) fn add_new_base_equality<F: Field>(
        &mut self,
        points: &[MultilinearPoint<F>],
        evaluations: &[EF],
        combination_randomness: &[EF],
    ) where
        EF: ExtensionField<F>,
    {
        assert_eq!(combination_randomness.len(), points.len());
        assert_eq!(evaluations.len(), points.len());

        // Parallel update of weight buffer

        points
            .iter()
            .zip(combination_randomness.iter())
            .for_each(|(point, &rand)| {
                crate::utils::compute_eval_eq_base::<_, _, true>(point, &mut self.weights, rand);
            });

        // Accumulate the weighted sum (cheap, done sequentially)
        self.sum += combination_randomness
            .iter()
            .zip(evaluations.iter())
            .map(|(&rand, &eval)| rand * eval)
            .sum::<EF>();
    }

    pub(crate) fn compute_sumcheck_polynomials<F: Field>(
        &mut self,
        prover_state: &mut ProverState<PF<EF>, EF, impl FSChallenger<EF>>,
        folding_factor: usize,
        pow_bits: usize,
    ) -> MultilinearPoint<EF>
    where
        EF: ExtensionField<F>,
    {
        let mut res = (0..folding_factor)
            .map(|_| {
                round::<F, EF>(
                    prover_state,
                    &mut self.evals,
                    &mut self.weights,
                    &mut self.sum,
                    pow_bits,
                )
            })
            .collect::<Vec<_>>();

        res.reverse();

        MultilinearPoint(res)
    }
}

fn combine_statement<Base, EF>(
    statement: &[Evaluation<EF>],
    challenge: EF,
    dot_product_statement: Option<(Vec<EF>, EF)>,
) -> (Vec<EF>, EF)
where
    Base: Field,
    EF: ExtensionField<Base>,
{
    let num_variables = statement[0].num_variables();
    assert!(statement.iter().all(|e| e.num_variables() == num_variables));

    let (mut combined_evals, sum_start) = if let Some((vec, val)) = dot_product_statement {
        assert_eq!(vec.len(), 1 << num_variables);
        (vec, val)
    } else {
        (EF::zero_vec(1 << num_variables), EF::ZERO)
    };
    let (combined_sum, _) = statement.iter().fold(
        (sum_start, challenge),
        |(mut acc_sum, gamma_pow), constraint| {
            compute_sparse_eval_eq::<Base, EF>(&constraint.point, &mut combined_evals, gamma_pow);
            acc_sum += constraint.value * gamma_pow;
            (acc_sum, gamma_pow * challenge)
        },
    );

    (combined_evals, combined_sum)
}
