use p3_field::{ExtensionField, Field};
use rayon::prelude::*;
use tracing::instrument;

use crate::*;

fn compress<F: Field, EF: ExtensionField<F>>(evals: &[F], r: EF) -> Vec<EF> {
    let fold = |slice: &[F]| -> EF { r * (slice[1] - slice[0]) + slice[0] };
    evals[..evals.len() / 2]
        .par_iter()
        .zip(&evals[evals.len() / 2..])
        .map(|(&a, &b)| fold(&[a, b]))
        .collect()
}

fn run_sumcheck_round<F: Field, EF: ExtensionField<F> + ExtensionField<PF<EF>>>(
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

    *weights = compress(weights, r);
    let compressed_evals = compress(evals, r);

    *sum = sumcheck_poly.evaluate(r);

    (r, compressed_evals)
}

#[instrument(skip_all, fields(num_variables = evals.num_variables()))]
pub(crate) fn compute_sumcheck_polynomial<F: Field, EF: ExtensionField<F>>(
    evals: &[F],
    weights: &Vec<EF>,
    sum: EF,
) -> DensePolynomial<EF> {
    assert!(evals.len().is_power_of_two());

    let (c0, c2) = (0..evals.len() / 2)
        .into_par_iter()
        .map(|i| {
            sumcheck_quadratic::<F, EF>(
                evals[i],
                evals[i + evals.len() / 2],
                weights[i],
                weights[i + evals.len() / 2],
            )
        })
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
pub(crate) fn sumcheck_quadratic<F, EF>(p_0: F, p_1: F, eq_0: EF, eq_1: EF) -> (EF, EF)
where
    F: Field,
    EF: ExtensionField<F>,
{
    // Compute the constant coefficient:
    // p(0) * w(0)
    let constant = eq_0 * p_0;

    // Compute the quadratic coefficient:
    // (p(1) - p(0)) * (w(1) - w(0))
    let quadratic = (eq_1 - eq_0) * (p_1 - p_0);

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
    pub(crate) fn run_initial_sumcheck_rounds<F: Field>(
        evals: &[F],
        statement: &[Evaluation<EF>],
        combination_randomness: EF,
        prover_state: &mut ProverState<PF<EF>, EF, impl FSChallenger<EF>>,
        folding_factor: usize,
        pow_bits: usize,
    ) -> (Self, MultilinearPoint<EF>)
    where
        EF: ExtensionField<F>,
    {
        assert_ne!(folding_factor, 0);
        let mut res = Vec::with_capacity(folding_factor);

        let (mut weights, mut sum) =
            combine_statement::<PF<EF>, EF>(statement, combination_randomness);
        // In the first round base field evaluations are folded into extension field elements
        let (r, mut evals) =
            run_sumcheck_round(prover_state, evals, &mut weights, &mut sum, pow_bits);
        res.push(r);

        // Apply rest of sumcheck rounds
        res.extend((1..folding_factor).map(|_| {
            let (r, folded) =
                run_sumcheck_round(prover_state, &evals, &mut weights, &mut sum, pow_bits);
            evals = folded;
            r
        }));

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
                compute_eval_eq::<_, _, true>(point, &mut self.weights, rand);
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
                compute_eval_eq_base::<_, _, true>(point, &mut self.weights, rand);
            });

        // Accumulate the weighted sum (cheap, done sequentially)
        self.sum += combination_randomness
            .iter()
            .zip(evaluations.iter())
            .map(|(&rand, &eval)| rand * eval)
            .sum::<EF>();
    }

    pub(crate) fn run_sumcheck_many_rounds<F: Field>(
        &mut self,
        prover_state: &mut ProverState<PF<EF>, EF, impl FSChallenger<EF>>,
        folding_factor: usize,
        pow_bits: usize,
    ) -> MultilinearPoint<EF>
    where
        EF: ExtensionField<F>,
    {
        MultilinearPoint(
            (0..folding_factor)
                .map(|_| {
                    let (r, folded) = run_sumcheck_round(
                        prover_state,
                        &mut self.evals,
                        &mut self.weights,
                        &mut self.sum,
                        pow_bits,
                    );
                    self.evals = folded;
                    r
                })
                .collect::<Vec<_>>(),
        )
    }
}

#[instrument(skip_all, fields(num_constraints = statement.len(), n_vars = statement[0].num_variables()))]
fn combine_statement<Base, EF>(statement: &[Evaluation<EF>], challenge: EF) -> (Vec<EF>, EF)
where
    Base: Field,
    EF: ExtensionField<Base>,
{
    let num_variables = statement[0].num_variables();
    assert!(statement.iter().all(|e| e.num_variables() == num_variables));

    let mut combined_evals = EF::zero_vec(1 << num_variables);
    let (combined_sum, _) = statement.iter().fold(
        (EF::ZERO, EF::ONE),
        |(mut acc_sum, gamma_pow), constraint| {
            compute_sparse_eval_eq::<Base, EF>(&constraint.point, &mut combined_evals, gamma_pow);
            acc_sum += constraint.value * gamma_pow;
            (acc_sum, gamma_pow * challenge)
        },
    );

    (combined_evals, combined_sum)
}
