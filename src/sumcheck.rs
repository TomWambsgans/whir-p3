use p3_challenger::{FieldChallenger, GrindingChallenger};
use p3_field::{ExtensionField, Field, TwoAdicField};
use rayon::prelude::*;
use tracing::instrument;

use crate::{
    PF,
    fiat_shamir::prover::ProverState,
    poly::{dense::WhirDensePolynomial, evals::EvaluationsList, multilinear::MultilinearPoint},
    whir::statement::Statement,
};

const PARALLEL_THRESHOLD: usize = 4096;


#[instrument(skip_all)]
pub fn compress_ext<F: Field, EF: ExtensionField<F>>(
    evals: &EvaluationsList<F>,
    r: EF,
) -> EvaluationsList<EF> {
    assert_ne!(evals.num_variables(), 0);

    // Fold between base and extension field elements
    let fold = |slice: &[F]| -> EF { r * (slice[1] - slice[0]) + slice[0] };

    let folded = if evals.evals().len() >= PARALLEL_THRESHOLD {
        evals.evals().par_chunks_exact(2).map(fold).collect()
    } else {
        evals.evals().chunks_exact(2).map(fold).collect()
    };

    EvaluationsList::new(folded)
}


pub fn compress<F: Field>(evals: &mut EvaluationsList<F>, r: F) {
    assert_ne!(evals.num_variables(), 0);

    if evals.evals().len() >= PARALLEL_THRESHOLD {
        // Define the folding operation for a pair of elements.
        let fold = |slice: &[F]| -> F { r * (slice[1] - slice[0]) + slice[0] };
        // Execute the fold in parallel and collect into a new vector.
        let folded = evals.evals().par_chunks_exact(2).map(fold).collect();
        // Replace the old evaluations with the new, folded evaluations.
        *evals = EvaluationsList::new(folded);
    } else {
        // For smaller inputs, we use the sequential, in-place strategy to save memory.
        let mid = evals.len() / 2;
        let evals_slice = evals.evals_mut();
        for i in 0..mid {
            let p0 = evals_slice[2 * i];
            let p1 = evals_slice[2 * i + 1];
            evals_slice[i] = r * (p1 - p0) + p0;
        }
        evals.truncate(mid);
    }
}

fn initial_round<Challenger, F: Field, EF: ExtensionField<F> + ExtensionField<PF<F>>>(
    prover_state: &mut ProverState<PF<F>, EF, Challenger>,
    evals: &EvaluationsList<F>,
    weights: &mut EvaluationsList<EF>,
    sum: &mut EF,
    pow_bits: usize,
) -> (EF, EvaluationsList<EF>)
where
    Challenger: FieldChallenger<PF<F>> + GrindingChallenger<Witness = PF<F>>,
{
    // Compute the quadratic sumcheck polynomial for the current variable.
    let sumcheck_poly = compute_sumcheck_polynomial(evals, weights, *sum);
    prover_state.add_extension_scalars(&sumcheck_poly.coeffs);

    // Sample verifier challenge.
    let r: EF = prover_state.sample();

    prover_state.pow_grinding(pow_bits);

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
fn round<Challenger, F: Field, EF: ExtensionField<F> + ExtensionField<PF<F>>>(
    prover_state: &mut ProverState<PF<F>, EF, Challenger>,
    evals: &mut EvaluationsList<EF>,
    weights: &mut EvaluationsList<EF>,
    sum: &mut EF,
    pow_bits: usize,
) -> EF
where
    Challenger: FieldChallenger<PF<F>> + GrindingChallenger<Witness = PF<F>>,
{
    // Compute the quadratic sumcheck polynomial for the current variable.
    let sumcheck_poly = compute_sumcheck_polynomial(evals, weights, *sum);
    prover_state.add_extension_scalars(&sumcheck_poly.coeffs);

    // Sample verifier challenge.
    let r: EF = prover_state.sample();

    prover_state.pow_grinding(pow_bits);

    // Compress polynomials and update the sum.
    rayon::join(|| compress(evals, r), || compress(weights, r));

    *sum = sumcheck_poly.evaluate(r);

    r
}

pub fn univariate_selectors<F: Field>(n: usize) -> Vec<WhirDensePolynomial<F>> {
    (0..1 << n)
        .into_par_iter()
        .map(|i| {
            let values = (0..1 << n)
                .map(|j| (F::from_u64(j), if i == j { F::ONE } else { F::ZERO }))
                .collect::<Vec<_>>();
            WhirDensePolynomial::lagrange_interpolation(&values).unwrap()
        })
        .collect()
}


#[instrument(skip_all, level = "debug")]
pub(crate) fn compute_sumcheck_polynomial<F: Field, EF: ExtensionField<F>>(
    evals: &EvaluationsList<F>,
    weights: &EvaluationsList<EF>,
    sum: EF,
) -> WhirDensePolynomial<EF> {
    assert!(evals.num_variables() >= 1);

    let (c0, c2) = evals
        .evals()
        .par_chunks_exact(2)
        .zip(weights.evals().par_chunks_exact(2))
        .map(sumcheck_quadratic::<F, EF>)
        .reduce(
            || (EF::ZERO, EF::ZERO),
            |(a0, a2), (b0, b2)| (a0 + b0, a2 + b2),
        );

   
    let c1 = sum - c0.double() - c2;

   
    let eval_0 = c0;
    let eval_1 = c0 + c1 + c2;
    let eval_2 = eval_1 + c1 + c2 + c2.double();

    WhirDensePolynomial::lagrange_interpolation(&[
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
pub struct SumcheckSingle<F, EF> {
    /// Evaluations of the polynomial `p(X)`.
    pub(crate) evals: EvaluationsList<EF>,
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
    #[instrument(skip_all)]
    pub fn from_base_evals<Challenger>(
        evals: &EvaluationsList<F>,
        statement: &Statement<EF>,
        combination_randomness: EF,

        prover_state: &mut ProverState<PF<F>, EF, Challenger>,
        folding_factor: usize,
        pow_bits: usize,
    ) -> (Self, MultilinearPoint<EF>)
    where
        F: TwoAdicField,
        EF: TwoAdicField + ExtensionField<PF<F>>,
        Challenger: FieldChallenger<PF<F>> + GrindingChallenger<Witness = PF<F>>,
    {
        assert_ne!(folding_factor, 0);
        let mut res = Vec::with_capacity(folding_factor);

        let (mut weights, mut sum) = statement.combine::<PF<F>>(combination_randomness);
        // In the first round base field evaluations are folded into extension field elements
        let (r, mut evals) = initial_round(prover_state, evals, &mut weights, &mut sum, pow_bits);
        res.push(r);

        // Apply rest of sumcheck rounds
        res.extend((1..folding_factor).map(|_| {
            round::<_, F, EF>(prover_state, &mut evals, &mut weights, &mut sum, pow_bits)
        }));

        res.reverse();

        let sumcheck = Self {
            evals,
            weights,
            sum,
            phantom: std::marker::PhantomData,
        };

        (sumcheck, MultilinearPoint(res))
    }

    pub const fn num_variables(&self) -> usize {
        self.evals.num_variables()
    }

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

        use tracing::info_span;

        info_span!("accumulate_weight_buffer").in_scope(|| {
            points
                .iter()
                .zip(combination_randomness.iter())
                .for_each(|(point, &rand)| {
                    crate::utils::eval_eq::<_, _, true>(point, self.weights.evals_mut(), rand);
                });
        });

        self.sum += combination_randomness
            .iter()
            .zip(evaluations.iter())
            .map(|(&rand, &eval)| rand * eval)
            .sum::<EF>();
    }

    #[instrument(skip_all)]
    pub fn compute_sumcheck_polynomials<Challenger>(
        &mut self,
        prover_state: &mut ProverState<PF<F>, EF, Challenger>,
        folding_factor: usize,
        pow_bits: usize,
    ) -> MultilinearPoint<EF>
    where
        F: TwoAdicField,
        EF: TwoAdicField,
        Challenger: FieldChallenger<PF<F>> + GrindingChallenger<Witness = PF<F>>,
        EF: ExtensionField<PF<F>>,
    {
        let mut res = (0..folding_factor)
            .map(|_| {
                round::<_, F, EF>(
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
