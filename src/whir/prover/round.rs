use p3_challenger::{FieldChallenger, GrindingChallenger};
use p3_field::{ExtensionField, TwoAdicField};
use rayon::iter::{IndexedParallelIterator, IntoParallelRefMutIterator, ParallelIterator};
use tracing::{info_span, instrument};

use super::Prover;
use crate::{
    PF,
    fiat_shamir::{errors::ProofResult, prover::ProverState, verifier::ChallengerState},
    poly::{evals::EvaluationsList, multilinear::MultilinearPoint},
    sumcheck::sumcheck_single::SumcheckSingle,
    whir::{
        committer::{RoundMerkleTree, Witness},
        statement::Statement,
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
pub(crate) struct RoundState<EF, F, const DIGEST_ELEMS: usize>
where
    F: TwoAdicField,
    EF: ExtensionField<F> + TwoAdicField,
{
    pub(crate) domain_size: usize,

    pub(crate) next_domain_gen: F,

    /// The sumcheck prover responsible for managing constraint accumulation and sumcheck rounds.
    /// Initialized in the first round (if applicable), and reused/updated in each subsequent round.
    pub(crate) sumcheck_prover: SumcheckSingle<F, EF>,

    /// The sampled folding randomness for this round, used to collapse a subset of variables.
    /// Length equals the folding factor at this round.
    pub(crate) folding_randomness: MultilinearPoint<EF>,

    /// Merkle commitment prover data for the **base field** polynomial from the first round.
    /// This is used to open values at queried locations.
    pub(crate) commitment_merkle_prover_data: RoundMerkleTree<PF<F>, F, DIGEST_ELEMS>,

    pub(crate) commitment_merkle_prover_data_b: RoundMerkleTree<PF<F>, F, DIGEST_ELEMS>,

    /// Merkle commitment prover data for the **extension field** polynomials (folded rounds).
    /// Present only after the first round.
    pub(crate) merkle_prover_data: Option<RoundMerkleTree<PF<F>, EF, DIGEST_ELEMS>>,

    /// Flat vector of challenge values used across all rounds.
    /// Populated progressively as folding randomness is sampled.
    /// The `i`-th index corresponds to variable `X_{n - 1 - i}`.
    pub(crate) randomness_vec: Vec<EF>,

    /// The accumulated set of linear equality constraints for this round.
    /// Used in computing the weighted sum for the sumcheck polynomial.
    pub(crate) statement: Statement<EF>,
}

#[allow(clippy::mismatching_type_param_order)]
impl<EF, F, const DIGEST_ELEMS: usize> RoundState<EF, F, DIGEST_ELEMS>
where
    F: TwoAdicField,
    EF: ExtensionField<F> + TwoAdicField,
    F: ExtensionField<PF<F>>,
    EF: ExtensionField<PF<F>>,
{
    #[instrument(skip_all)]
    pub(crate) fn initialize_first_round_state<MyChallenger, C, Challenger>(
        prover: &Prover<'_, EF, F, MyChallenger, C, Challenger>,
        prover_state: &mut ProverState<PF<F>, EF, Challenger>,
        statement_a: Statement<EF>,
        witness_a: Witness<EF, F, DIGEST_ELEMS>,
        statement_b: Statement<EF>,
        witness_b: Witness<EF, F, DIGEST_ELEMS>,
    ) -> ProofResult<Self>
    where
        Challenger: FieldChallenger<PF<F>> + GrindingChallenger<Witness = PF<F>> + ChallengerState,
    {
        let n_vars_a = statement_a.num_variables();
        let n_vars_b = statement_b.num_variables();

        let mut statement = Statement::new(statement_a.num_variables() + 1);

        for (point, evaluation) in witness_a.ood_points.into_iter().zip(witness_a.ood_answers) {
            let mut point = MultilinearPoint::expand_from_univariate(point, n_vars_a);
            point.push(EF::ONE);
            statement.add_constraint(point, evaluation);
        }
        for mut constraint in statement_a.constraints {
            constraint.weights.push(EF::ONE);
            statement.add_constraint(constraint.weights, constraint.sum);
        }
        for (point, evaluation) in witness_b.ood_points.into_iter().zip(witness_b.ood_answers) {
            let mut point = MultilinearPoint::expand_from_univariate(point, n_vars_b);
            point.extend(vec![EF::ZERO; n_vars_a + 1 - n_vars_b]);
            statement.add_constraint(point, evaluation);
        }
        for mut constraint in statement_b.constraints {
            constraint
                .weights
                .extend(vec![EF::ZERO; n_vars_a + 1 - n_vars_b]);
            statement.add_constraint(constraint.weights, constraint.sum);
        }

        let combination_randomness_gen: EF = prover_state.sample();

        let _span = info_span!("merging 2 batched polynomials", n_vars_a, n_vars_b,).entered();
        let mut polynomial = F::zero_vec(witness_a.polynomial.num_evals() * 2);
        polynomial
            .par_iter_mut()
            .step_by(1 << (1 + n_vars_a - n_vars_b))
            .enumerate()
            .for_each(|(i, eval)| {
                *eval = witness_b.polynomial.evals()[i];
            });
        polynomial[1..]
            .par_iter_mut()
            .step_by(2)
            .enumerate()
            .for_each(|(i, eval)| {
                *eval = witness_a.polynomial.evals()[i];
            });
        std::mem::drop(_span);

        let polynomial = EvaluationsList::new(polynomial);

        let (sumcheck_prover, folding_randomness) = SumcheckSingle::from_base_evals(
            &polynomial,
            &statement,
            combination_randomness_gen,
            prover_state,
            prover.folding_factor.at_round(0) + 1,
            prover.starting_folding_pow_bits,
        );

        let randomness_vec = info_span!("copy_across_random_vec").in_scope(|| {
            let mut randomness_vec = Vec::with_capacity(prover.mv_parameters.num_variables);
            randomness_vec.extend(folding_randomness.iter().rev().copied());
            randomness_vec.resize(prover.mv_parameters.num_variables, EF::ZERO);
            randomness_vec
        });

        Ok(Self {
            domain_size: prover.starting_domain_size(),
            next_domain_gen: F::two_adic_generator(
                prover.starting_domain_size().ilog2() as usize - prover.folding_factor.at_round(0),
            ),
            sumcheck_prover,
            folding_randomness,
            merkle_prover_data: None,
            commitment_merkle_prover_data: witness_a.prover_data,
            commitment_merkle_prover_data_b: witness_b.prover_data,
            randomness_vec,
            statement,
        })
    }
}
