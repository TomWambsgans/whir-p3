use p3_challenger::{FieldChallenger, GrindingChallenger};
use p3_field::{ExtensionField, TwoAdicField};
use tracing::{info_span, instrument};

use super::Prover;
use crate::{
    fiat_shamir::{errors::ProofResult, prover::ProverState},
    poly::multilinear::MultilinearPoint,
    sumcheck::{sumcheck_single::SumcheckSingle, K_SKIP_SUMCHECK},
    whir::{
        committer::{RoundMerkleTree, Witness},
        statement::{weights::Weights, Statement},
    }, PF,
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
    /// Initializes the prover’s state for the first round of the WHIR protocol.
    ///
    /// This function prepares all round-local state needed to begin the interactive proof:
    /// - If the WHIR protocol has an initial statement, it runs the first sumcheck round and
    ///   samples folding randomness using Fiat-Shamir.
    /// - Otherwise, it directly absorbs verifier-supplied randomness for folding.
    /// - It incorporates any out-of-domain (OOD) constraints derived from the witness,
    ///   and prepares the polynomial coefficients accordingly.
    ///
    /// Returns a fully-formed `RoundState` containing:
    /// - The active domain,
    /// - The initial polynomial (as coefficients),
    /// - The first sumcheck prover (if applicable),
    /// - The sampled folding randomness,
    /// - Constraint tracking data,
    /// - Merkle tree commitment data.
    ///
    /// This function should be called once at the beginning of the proof, before entering the
    /// main WHIR folding loop.
    #[instrument(skip_all)]
    pub(crate) fn initialize_first_round_state<MyChallenger, C, Challenger>(
        prover: &Prover<'_, EF, F, MyChallenger, C, Challenger>,
        prover_state: &mut ProverState<PF<F>, EF, Challenger>,
        mut statement: Statement<EF>,
        witness: Witness<EF, F, DIGEST_ELEMS>,
    ) -> ProofResult<Self>
    where
        Challenger: FieldChallenger<PF<F>> + GrindingChallenger<Witness = PF<F>>,
    {
        // Convert witness ood_points into constraints
        let new_constraints = witness
            .ood_points
            .into_iter()
            .zip(witness.ood_answers)
            .map(|(point, evaluation)| {
                let weights = Weights::evaluation(MultilinearPoint::expand_from_univariate(
                    point,
                    prover.mv_parameters.num_variables,
                ));
                (weights, evaluation)
            })
            .collect();

        statement.add_constraints_in_front(new_constraints);

        let (sumcheck_prover, folding_randomness) = if prover.initial_statement {
            // If there is initial statement, then we run the sum-check for
            // this initial statement.
            let combination_randomness_gen: EF = prover_state.sample();

            // Create the sumcheck prover
            let (sumcheck, folding_randomness) = if prover.univariate_skip {
                SumcheckSingle::with_skip(
                    &witness.polynomial,
                    &statement,
                    combination_randomness_gen,
                    prover_state,
                    prover.folding_factor.at_round(0),
                    prover.starting_folding_pow_bits,
                    K_SKIP_SUMCHECK,
                )
            } else {
                SumcheckSingle::from_base_evals(
                    &witness.polynomial,
                    &statement,
                    combination_randomness_gen,
                    prover_state,
                    prover.folding_factor.at_round(0),
                    prover.starting_folding_pow_bits,
                )
            };

            (sumcheck, folding_randomness)
        } else {
            // If there is no initial statement, there is no need to run the
            // initial rounds of the sum-check, and the verifier directly sends
            // the initial folding randomnesses.

            let folding_randomness = MultilinearPoint(
                (0..prover.folding_factor.at_round(0))
                    .map(|_| prover_state.sample())
                    .collect::<Vec<_>>(),
            );

            let poly = witness.polynomial.fold(&folding_randomness);
            let num_variables = poly.num_variables();

            // Create the sumcheck prover w/o running any rounds.
            let sumcheck =
                SumcheckSingle::from_extension_evals(poly, &Statement::new(num_variables), EF::ONE);

            prover_state.pow_grinding(prover.starting_folding_pow_bits);

            (sumcheck, folding_randomness)
        };

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
            commitment_merkle_prover_data: witness.prover_data,
            randomness_vec,
            statement,
        })
    }
}
