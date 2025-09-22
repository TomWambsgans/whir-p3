use p3_field::{ExtensionField, Field, TwoAdicField};
use rayon::prelude::*;

use crate::{
    PF,
    fiat_shamir::{FSChallenger, errors::ProofResult, prover::ProverState},
    poly::{
        evals::EvaluationsList,
        multilinear::{Evaluation, MultilinearPoint},
    },
    sumcheck::SumcheckSingle,
    whir::{
        committer::{RoundMerkleTree, Witness},
        config::WhirConfig,
    },
};

/// Holds all per-round prover state required during the execution of the WHIR protocol.
///
/// Each round involves:
/// - A domain extension and folding step,
/// - Merkle commitments and openings,
/// - A sumcheck polynomial generation and folding randomness sampling,
/// - Bookkeeping of constraints and evaluation points.
///
/// The `RoundState` evolves with each round and captures all intermediate data required
/// to continue proving or to verify challenges from the verifier.
#[derive(Debug)]
pub(crate) struct RoundState<F, EF, const DIGEST_ELEMS: usize>
where
    F: TwoAdicField,
    EF: ExtensionField<F> + TwoAdicField,
{
    pub(crate) domain_size: usize,

    pub(crate) next_domain_gen: F,

    /// The sumcheck prover responsible for managing constraint accumulation and sumcheck rounds.
    /// Initialized in the first round (if applicable), and reused/updated in each subsequent round.
    pub(crate) sumcheck_prover: SumcheckSingle<EF>,

    /// The sampled folding randomness for this round, used to collapse a subset of variables.
    /// Length equals the folding factor at this round.
    pub(crate) folding_randomness: MultilinearPoint<EF>,

    /// Merkle commitment prover data for the **base field** polynomial from the first round.
    /// This is used to open values at queried locations.
    pub(crate) commitment_merkle_prover_data_a: RoundMerkleTree<PF<EF>, F, DIGEST_ELEMS>,

    pub(crate) commitment_merkle_prover_data_b: Option<RoundMerkleTree<PF<EF>, EF, DIGEST_ELEMS>>,

    /// Merkle commitment prover data for the **extension field** polynomials (folded rounds).
    /// Present only after the first round.
    pub(crate) merkle_prover_data: Option<RoundMerkleTree<PF<EF>, EF, DIGEST_ELEMS>>,

    /// Flat vector of challenge values used across all rounds.
    /// Populated progressively as folding randomness is sampled.
    /// The `i`-th index corresponds to variable `X_{n - 1 - i}`.
    pub(crate) randomness_vec: Vec<EF>,

    /// The accumulated set of linear equality constraints for this round.
    /// Used in computing the weighted sum for the sumcheck polynomial.
    pub(crate) statement: Vec<Evaluation<EF>>,
}

#[allow(clippy::mismatching_type_param_order)]
impl<F, EF, const DIGEST_ELEMS: usize> RoundState<F, EF, DIGEST_ELEMS>
where
    F: TwoAdicField,
    EF: ExtensionField<F> + TwoAdicField,
    F: ExtensionField<PF<EF>>,
    EF: ExtensionField<PF<EF>>,
{
    pub(crate) fn initialize_first_round_state<MyChallenger, C>(
        prover: &WhirConfig<F, EF, MyChallenger, C, DIGEST_ELEMS>,
        prover_state: &mut ProverState<PF<EF>, EF, impl FSChallenger<EF>>,
        mut statement: Vec<Evaluation<EF>>,
        witness: Witness<F, EF, DIGEST_ELEMS>,
        polynomial: &[F],
        dot_product_statement: Option<(Vec<EF>, EF)>,
    ) -> ProofResult<Self> {
        // Convert witness ood_points into constraints
        let new_constraints = witness
            .ood_points
            .into_iter()
            .zip(witness.ood_answers)
            .map(|(point, evaluation)| {
                let weights = MultilinearPoint::expand_from_univariate(point, prover.num_variables);
                (weights, evaluation)
            })
            .collect();

        add_constraints_in_front(&mut statement, new_constraints);

        let combination_randomness_gen: EF = prover_state.sample();

        let (sumcheck_prover, folding_randomness) = SumcheckSingle::from_base_evals(
            polynomial,
            &statement,
            combination_randomness_gen,
            prover_state,
            prover.folding_factor.at_round(0),
            prover.starting_folding_pow_bits,
            dot_product_statement,
        );

        let randomness_vec = {
            let mut randomness_vec = Vec::with_capacity(prover.num_variables);
            randomness_vec.extend(folding_randomness.iter().rev().copied());
            randomness_vec.resize(prover.num_variables, EF::ZERO);
            randomness_vec
        };

        Ok(Self {
            domain_size: prover.starting_domain_size(),
            next_domain_gen: F::two_adic_generator(
                prover.starting_domain_size().ilog2() as usize - prover.folding_factor.at_round(0),
            ),
            sumcheck_prover,
            folding_randomness,
            merkle_prover_data: None,
            commitment_merkle_prover_data_a: witness.prover_data,
            commitment_merkle_prover_data_b: None,
            randomness_vec,
            statement,
        })
    }

    pub(crate) fn initialize_first_round_state_batch<MyChallenger, C>(
        prover: &WhirConfig<F, EF, MyChallenger, C, DIGEST_ELEMS>,
        prover_state: &mut ProverState<PF<EF>, EF, impl FSChallenger<EF>>,
        statement_a: Vec<Evaluation<EF>>,
        witness_a: Witness<F, EF, DIGEST_ELEMS>,
        polynomial_a: &[F],
        statement_b: Vec<Evaluation<EF>>,
        witness_b: Witness<EF, EF, DIGEST_ELEMS>,
        polynomial_b: &[EF],
        dot_product_statement_a: Option<(Vec<EF>, EF)>,
    ) -> ProofResult<Self> {
        let n_vars_a = statement_a[0].num_variables();
        let n_vars_b = statement_b[0].num_variables();

        let mut statement = Vec::new();

        for (point, evaluation) in witness_a.ood_points.into_iter().zip(witness_a.ood_answers) {
            let mut point = MultilinearPoint::expand_from_univariate(point, n_vars_a);
            point.push(EF::ONE);
            statement.push(Evaluation::new(point, evaluation));
        }
        for mut constraint in statement_a {
            constraint.point.push(EF::ONE);
            statement.push(constraint);
        }
        for (point, evaluation) in witness_b.ood_points.into_iter().zip(witness_b.ood_answers) {
            let mut point = MultilinearPoint::expand_from_univariate(point, n_vars_b);
            point.extend(vec![EF::ZERO; n_vars_a + 1 - n_vars_b]);
            statement.push(Evaluation::new(point, evaluation));
        }
        for mut constraint in statement_b {
            constraint
                .point
                .extend(vec![EF::ZERO; n_vars_a + 1 - n_vars_b]);
            statement.push(constraint);
        }

        let combination_randomness_gen: EF = prover_state.sample();

        let mut polynomial = EF::zero_vec(polynomial_a.num_evals() * 2);
        polynomial
            .par_iter_mut()
            .step_by(1 << (1 + n_vars_a - n_vars_b))
            .enumerate()
            .for_each(|(i, eval)| {
                *eval = polynomial_b[i];
            });
        polynomial[1..]
            .par_iter_mut()
            .step_by(2)
            .enumerate()
            .for_each(|(i, eval)| {
                *eval = EF::from(polynomial_a[i]); // TODO embedding overhead
            });

        let dot_product_statement = dot_product_statement_a.map(|(slice, sum)| {
            let mut new_vec = EF::zero_vec(polynomial_a.num_evals() * 2);
            new_vec[1..]
                .par_iter_mut()
                .step_by(2)
                .zip(slice.par_iter())
                .for_each(|(a, b)| {
                    *a = *b;
                });
            (new_vec, sum)
        });

        let (sumcheck_prover, folding_randomness) = SumcheckSingle::from_base_evals(
            &polynomial,
            &statement,
            combination_randomness_gen,
            prover_state,
            prover.folding_factor.at_round(0) + 1,
            prover.starting_folding_pow_bits,
            dot_product_statement,
        );

        let randomness_vec = {
            let mut randomness_vec = Vec::with_capacity(prover.num_variables);
            randomness_vec.extend(folding_randomness.iter().rev().copied());
            randomness_vec.resize(prover.num_variables, EF::ZERO);
            randomness_vec
        };

        Ok(Self {
            domain_size: prover.starting_domain_size(),
            next_domain_gen: F::two_adic_generator(
                prover.starting_domain_size().ilog2() as usize - prover.folding_factor.at_round(0),
            ),
            sumcheck_prover,
            folding_randomness,
            merkle_prover_data: None,
            commitment_merkle_prover_data_a: witness_a.prover_data,
            commitment_merkle_prover_data_b: Some(witness_b.prover_data),
            randomness_vec,
            statement,
        })
    }
}

fn add_constraints_in_front<EF: Field>(
    statements: &mut Vec<Evaluation<EF>>,
    constraints: Vec<(MultilinearPoint<EF>, EF)>,
) {
    // Store the number of variables expected by this statement.
    let n = statements[0].num_variables();

    // Preallocate a vector for the converted constraints to avoid reallocations.
    let mut new_constraints = Vec::with_capacity(constraints.len());

    // Iterate through each (weights, sum) pair in the input.
    for (weights, sum) in constraints {
        // Ensure the number of variables in the weight matches the statement.
        assert_eq!(weights.num_variables(), n);

        // Convert the pair into a full `Constraint` with `defer_evaluation = false`.
        new_constraints.push(Evaluation {
            point: weights,
            value: sum,
        });
    }

    // Insert all new constraints at the beginning of the existing list.
    statements.splice(0..0, new_constraints);
}
