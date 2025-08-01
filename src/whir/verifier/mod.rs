use std::{fmt::Debug, ops::Deref};

use p3_commit::{BatchOpeningRef, ExtensionMmcs, Mmcs};
use p3_field::{BasedVectorSpace, ExtensionField, Field, TwoAdicField};
use p3_matrix::Dimensions;
use p3_merkle_tree::MerkleTreeMmcs;
use p3_symmetric::{CryptographicHasher, Hash, PseudoCompressionFunction};
use serde::{Deserialize, Serialize};
use tracing::instrument;

use super::{
    committer::reader::ParsedCommitment, statement::constraint::Constraint,
    utils::get_challenge_stir_queries,
};
use crate::{
    PF,
    fiat_shamir::{
        FSChallenger,
        errors::{ProofError, ProofResult},
        verifier::VerifierState,
    },
    poly::{evals::EvaluationsList, multilinear::MultilinearPoint},
    utils::pack_scalars_to_extension,
    whir::{
        config::{RoundConfig, WhirConfig},
        statement::Statement,
        verifier::sumcheck::verify_sumcheck_rounds,
    },
};

pub mod sumcheck;

/// Wrapper around the WHIR verifier configuration.
///
/// This type provides a lightweight, ergonomic interface to verification methods
/// by wrapping a reference to the `WhirConfig`.
#[derive(Debug)]
pub struct Verifier<'a, F, EF, H, C, const DIGEST_ELEMS: usize>(
    /// Reference to the verifier’s configuration containing all round parameters.
    pub(crate) &'a WhirConfig<F, EF, H, C, DIGEST_ELEMS>,
)
where
    F: Field,
    EF: ExtensionField<F>;

impl<'a, F, EF, H, C, const DIGEST_ELEMS: usize> Verifier<'a, F, EF, H, C, DIGEST_ELEMS>
where
    F: TwoAdicField,
    EF: ExtensionField<F> + TwoAdicField + ExtensionField<PF<F>>,
    F: ExtensionField<PF<F>>,
{
    pub const fn new(params: &'a WhirConfig<F, EF, H, C, DIGEST_ELEMS>) -> Self {
        Self(params)
    }

    #[instrument(skip_all)]
    #[allow(clippy::too_many_lines)]
    pub fn batch_verify(
        &self,
        verifier_state: &mut VerifierState<PF<F>, EF, impl FSChallenger<F>>,
        parsed_commitment_a: &ParsedCommitment<F, EF, DIGEST_ELEMS>,
        statement_a: &Statement<EF>,
        parsed_commitment_b: &ParsedCommitment<F, EF, DIGEST_ELEMS>,
        statement_b: &Statement<EF>,
    ) -> ProofResult<MultilinearPoint<EF>>
    where
        H: CryptographicHasher<PF<F>, [PF<F>; DIGEST_ELEMS]>
            + CryptographicHasher<PF<F>, [PF<F>; DIGEST_ELEMS]>
            + Sync,
        C: PseudoCompressionFunction<[PF<F>; DIGEST_ELEMS], 2>
            + PseudoCompressionFunction<[PF<F>; DIGEST_ELEMS], 2>
            + Sync,
        [PF<F>; DIGEST_ELEMS]: Serialize + for<'de> Deserialize<'de>,
    {
        // During the rounds we collect constraints, combination randomness, folding randomness
        // and we update the claimed sum of constraint evaluation.
        let mut round_constraints = Vec::new();
        let mut round_folding_randomness = Vec::new();
        let mut claimed_sum = EF::ZERO;
        let mut prev_commitment = None;

        // Combine OODS and statement constraints to claimed_sum
        let mut constraints: Vec<_> = parsed_commitment_a
            .oods_constraints()
            .into_iter()
            .chain(statement_a.constraints.iter().cloned())
            .map(|mut c| {
                c.weights.push(EF::ONE);
                c
            })
            .collect();

        constraints.extend(
            parsed_commitment_b
                .oods_constraints()
                .into_iter()
                .chain(statement_b.constraints.iter().cloned())
                .map(|mut c| {
                    let ending_zeros =
                        parsed_commitment_a.num_variables + 1 - parsed_commitment_b.num_variables;
                    c.weights.extend(vec![EF::ZERO; ending_zeros]);
                    c
                }),
        );

        let combination_randomness =
            self.combine_constraints(verifier_state, &mut claimed_sum, &constraints)?;
        round_constraints.push((combination_randomness, constraints));

        // Initial sumcheck
        let folding_randomness = verify_sumcheck_rounds::<F, EF, _>(
            verifier_state,
            &mut claimed_sum,
            self.folding_factor.at_round(0) + 1,
            self.starting_folding_pow_bits,
        )?;
        round_folding_randomness.push(folding_randomness);

        for round_index in 0..self.n_rounds() {
            // Fetch round parameters from config
            let round_params = &self.round_parameters[round_index];

            // Receive commitment to the folded polynomial (likely encoded at higher expansion)
            let new_commitment = ParsedCommitment::<F, EF, DIGEST_ELEMS>::parse(
                verifier_state,
                round_params.num_variables,
                round_params.ood_samples,
            )?;

            // Verify in-domain challenges on the previous commitment.
            let stir_constraints = if round_index == 0 {
                self.verify_stir_challenges_batched(
                    verifier_state,
                    round_params,
                    &parsed_commitment_a,
                    &parsed_commitment_b,
                    round_folding_randomness.last().unwrap(),
                    round_index,
                )?
            } else {
                self.verify_stir_challenges(
                    verifier_state,
                    round_params,
                    prev_commitment.as_ref().unwrap(),
                    round_folding_randomness.last().unwrap(),
                    round_index,
                )?
            };

            // Add out-of-domain and in-domain constraints to claimed_sum
            let constraints: Vec<Constraint<EF>> = new_commitment
                .oods_constraints()
                .into_iter()
                .chain(stir_constraints.into_iter())
                .collect();

            let combination_randomness =
                self.combine_constraints(verifier_state, &mut claimed_sum, &constraints)?;
            round_constraints.push((combination_randomness.clone(), constraints));

            let folding_randomness = verify_sumcheck_rounds::<F, EF, _>(
                verifier_state,
                &mut claimed_sum,
                self.folding_factor.at_round(round_index + 1),
                round_params.folding_pow_bits,
            )?;

            round_folding_randomness.push(folding_randomness);

            // Update round parameters
            prev_commitment = Some(new_commitment);
        }

        // In the final round we receive the full polynomial instead of a commitment.
        let n_final_coeffs = 1 << self.n_vars_of_final_polynomial();
        let final_coefficients = verifier_state.next_extension_scalars_vec(n_final_coeffs)?;
        let final_evaluations = EvaluationsList::new(final_coefficients);

        // Verify in-domain challenges on the previous commitment.
        let stir_constraints = self.verify_stir_challenges(
            verifier_state,
            &self.final_round_config(),
            prev_commitment.as_ref().unwrap(),
            round_folding_randomness.last().unwrap(),
            self.n_rounds(),
        )?;

        // Verify stir constraints directly on final polynomial
        stir_constraints
            .iter()
            .all(|c| c.verify(&final_evaluations))
            .then_some(())
            .ok_or(ProofError::InvalidProof)?;

        let final_sumcheck_randomness = verify_sumcheck_rounds::<F, EF, _>(
            verifier_state,
            &mut claimed_sum,
            self.final_sumcheck_rounds,
            self.final_folding_pow_bits,
        )?;
        round_folding_randomness.push(final_sumcheck_randomness.clone());

        // Compute folding randomness across all rounds.
        let folding_randomness = MultilinearPoint(
            round_folding_randomness
                .into_iter()
                .rev()
                .flat_map(|poly| poly.0.into_iter())
                .collect(),
        );

        let evaluation_of_weights =
            self.eval_constraints_poly(&round_constraints, folding_randomness.clone());

        // Check the final sumcheck evaluation
        let final_value = final_evaluations.evaluate(&final_sumcheck_randomness);
        if claimed_sum != evaluation_of_weights * final_value {
            return Err(ProofError::InvalidProof);
        }

        Ok(folding_randomness)
    }

    #[instrument(skip_all)]
    #[allow(clippy::too_many_lines)]
    pub fn verify(
        &self,
        verifier_state: &mut VerifierState<PF<F>, EF, impl FSChallenger<F>>,
        parsed_commitment: &ParsedCommitment<F, EF, DIGEST_ELEMS>,
        statement: &Statement<EF>,
    ) -> ProofResult<MultilinearPoint<EF>>
    where
        H: CryptographicHasher<PF<F>, [PF<F>; DIGEST_ELEMS]>
            + CryptographicHasher<PF<F>, [PF<F>; DIGEST_ELEMS]>
            + Sync,
        C: PseudoCompressionFunction<[PF<F>; DIGEST_ELEMS], 2>
            + PseudoCompressionFunction<[PF<F>; DIGEST_ELEMS], 2>
            + Sync,
        [PF<F>; DIGEST_ELEMS]: Serialize + for<'de> Deserialize<'de>,
    {
        // During the rounds we collect constraints, combination randomness, folding randomness
        // and we update the claimed sum of constraint evaluation.
        let mut round_constraints = Vec::new();
        let mut round_folding_randomness = Vec::new();
        let mut claimed_sum = EF::ZERO;
        let mut prev_commitment = parsed_commitment.clone();

        // Combine OODS and statement constraints to claimed_sum
        let constraints: Vec<_> = prev_commitment
            .oods_constraints()
            .into_iter()
            .chain(statement.constraints.iter().cloned())
            .collect();
        let combination_randomness =
            self.combine_constraints(verifier_state, &mut claimed_sum, &constraints)?;
        round_constraints.push((combination_randomness, constraints));

        // Initial sumcheck
        let folding_randomness = verify_sumcheck_rounds::<F, EF, _>(
            verifier_state,
            &mut claimed_sum,
            self.folding_factor.at_round(0),
            self.starting_folding_pow_bits,
        )?;
        round_folding_randomness.push(folding_randomness);

        for round_index in 0..self.n_rounds() {
            // Fetch round parameters from config
            let round_params = &self.round_parameters[round_index];

            // Receive commitment to the folded polynomial (likely encoded at higher expansion)
            let new_commitment = ParsedCommitment::<F, EF, DIGEST_ELEMS>::parse(
                verifier_state,
                round_params.num_variables,
                round_params.ood_samples,
            )?;

            // Verify in-domain challenges on the previous commitment.
            let stir_constraints = self.verify_stir_challenges(
                verifier_state,
                round_params,
                &prev_commitment,
                round_folding_randomness.last().unwrap(),
                round_index,
            )?;

            // Add out-of-domain and in-domain constraints to claimed_sum
            let constraints: Vec<Constraint<EF>> = new_commitment
                .oods_constraints()
                .into_iter()
                .chain(stir_constraints.into_iter())
                .collect();

            let combination_randomness =
                self.combine_constraints(verifier_state, &mut claimed_sum, &constraints)?;
            round_constraints.push((combination_randomness.clone(), constraints));

            let folding_randomness = verify_sumcheck_rounds::<F, EF, _>(
                verifier_state,
                &mut claimed_sum,
                self.folding_factor.at_round(round_index + 1),
                round_params.folding_pow_bits,
            )?;

            round_folding_randomness.push(folding_randomness);

            // Update round parameters
            prev_commitment = new_commitment;
        }

        // In the final round we receive the full polynomial instead of a commitment.
        let n_final_coeffs = 1 << self.n_vars_of_final_polynomial();
        let final_coefficients = verifier_state.next_extension_scalars_vec(n_final_coeffs)?;
        let final_evaluations = EvaluationsList::new(final_coefficients);

        // Verify in-domain challenges on the previous commitment.
        let stir_constraints = self.verify_stir_challenges(
            verifier_state,
            &self.final_round_config(),
            &prev_commitment,
            round_folding_randomness.last().unwrap(),
            self.n_rounds(),
        )?;

        // Verify stir constraints directly on final polynomial
        stir_constraints
            .iter()
            .all(|c| c.verify(&final_evaluations))
            .then_some(())
            .ok_or(ProofError::InvalidProof)?;

        let final_sumcheck_randomness = verify_sumcheck_rounds::<F, EF, _>(
            verifier_state,
            &mut claimed_sum,
            self.final_sumcheck_rounds,
            self.final_folding_pow_bits,
        )?;
        round_folding_randomness.push(final_sumcheck_randomness.clone());

        // Compute folding randomness across all rounds.
        let folding_randomness = MultilinearPoint(
            round_folding_randomness
                .into_iter()
                .rev()
                .flat_map(|poly| poly.0.into_iter())
                .collect(),
        );

        let evaluation_of_weights =
            self.eval_constraints_poly(&round_constraints, folding_randomness.clone());

        // Check the final sumcheck evaluation
        let final_value = final_evaluations.evaluate(&final_sumcheck_randomness);
        if claimed_sum != evaluation_of_weights * final_value {
            return Err(ProofError::InvalidProof);
        }

        Ok(folding_randomness)
    }

    /// Combine multiple constraints into a single claim using random linear combination.
    ///
    /// This method draws a challenge scalar from the Fiat-Shamir transcript and uses it
    /// to generate a sequence of powers, one for each constraint. These powers serve as
    /// coefficients in a random linear combination of the constraint sums.
    ///
    /// The resulting linear combination is added to `claimed_sum`, which becomes the new
    /// target value to verify in the sumcheck protocol.
    ///
    /// # Arguments
    /// - `verifier_state`: Fiat-Shamir transcript reader.
    /// - `claimed_sum`: Mutable reference to the running sum of combined constraints.
    /// - `constraints`: List of constraints to combine.
    ///
    /// # Returns
    /// A vector of randomness values used to weight each constraint.
    pub fn combine_constraints(
        &self,
        verifier_state: &mut VerifierState<PF<F>, EF, impl FSChallenger<F>>,
        claimed_sum: &mut EF,
        constraints: &[Constraint<EF>],
    ) -> ProofResult<Vec<EF>> {
        let combination_randomness_gen: EF = verifier_state.sample();
        let combination_randomness: Vec<_> = combination_randomness_gen
            .powers()
            .take(constraints.len())
            .collect();
        *claimed_sum += constraints
            .iter()
            .zip(&combination_randomness)
            .map(|(c, &rand)| rand * c.sum)
            .sum::<EF>();

        Ok(combination_randomness)
    }

    /// Verify STIR in-domain queries and produce associated constraints.
    ///
    /// This method runs the STIR query phase on a given commitment.
    /// It selects random in-domain indices (STIR challenges)
    /// and verifies Merkle proofs for the claimed values at these indices.
    ///
    /// After verification, it evaluates the folded polynomial at these queried points.
    /// It then packages the results as a list of `Constraint` objects,
    /// ready to be combined into the next round’s sumcheck.
    ///
    /// # Arguments
    /// - `verifier_state`: The verifier’s Fiat-Shamir state.
    /// - `params`: Parameters for the current STIR round (domain size, folding factor, etc.).
    /// - `commitment`: The prover’s commitment to the folded polynomial.
    /// - `folding_randomness`: Random point for folding the evaluations.
    /// - `leafs_base_field`: Whether the leaf data is in the base field or extension field.
    ///
    /// # Returns
    /// A vector of `Constraint` objects, each linking a queried domain point
    /// to its evaluated, folded value under the prover’s commitment.
    ///
    /// # Errors
    /// Returns `ProofError::InvalidProof` if Merkle proof verification fails
    /// or the prover’s data does not match the commitment.
    pub fn verify_stir_challenges(
        &self,
        verifier_state: &mut VerifierState<PF<F>, EF, impl FSChallenger<F>>,
        params: &RoundConfig<F>,
        commitment: &ParsedCommitment<F, EF, DIGEST_ELEMS>,
        folding_randomness: &MultilinearPoint<EF>,
        round_index: usize,
    ) -> ProofResult<Vec<Constraint<EF>>>
    where
        H: CryptographicHasher<PF<F>, [PF<F>; DIGEST_ELEMS]>
            + CryptographicHasher<PF<F>, [PF<F>; DIGEST_ELEMS]>
            + Sync,
        C: PseudoCompressionFunction<[PF<F>; DIGEST_ELEMS], 2>
            + PseudoCompressionFunction<[PF<F>; DIGEST_ELEMS], 2>
            + Sync,
        [PF<F>; DIGEST_ELEMS]: Serialize + for<'de> Deserialize<'de>,
    {
        let leafs_base_field = round_index == 0;

        // CRITICAL: Verify the prover's proof-of-work before generating challenges.
        //
        // This is the verifier's counterpart to the prover's grinding step and is essential
        // for protocol soundness.
        //
        // The query locations (`stir_challenges_indexes`) we are about to generate are derived
        // from the transcript, which includes the prover's commitment for this round. To prevent
        // a malicious prover from repeatedly trying different commitments until they find one that
        // produces "easy" queries, the protocol forces the prover to perform an expensive
        // proof-of-work (grinding) after they commit.
        //
        // By verifying that proof-of-work *now*, we confirm that the prover "locked in" their
        // commitment at a significant computational cost. This gives us confidence that the
        // challenges we generate are unpredictable and unbiased by a cheating prover.
        verifier_state.check_pow_grinding(params.pow_bits)?;

        let stir_challenges_indexes = get_challenge_stir_queries(
            params.domain_size >> params.folding_factor,
            params.num_queries,
            verifier_state,
        );

        // dbg!(&stir_challenges_indexes);
        // dbg!(verifier_state.challenger().state());

        let dimensions = vec![Dimensions {
            height: params.domain_size >> params.folding_factor,
            width: 1 << params.folding_factor,
        }];
        let answers = self.verify_merkle_proof(
            verifier_state,
            &commitment.root,
            &stir_challenges_indexes,
            &dimensions,
            leafs_base_field,
            round_index,
            0,
        )?;

        // Compute STIR Constraints
        let folds: Vec<_> = answers
            .into_iter()
            .map(|answers| EvaluationsList::new(answers).evaluate(folding_randomness))
            .collect();

        let stir_constraints = stir_challenges_indexes
            .iter()
            .map(|&index| params.folded_domain_gen.exp_u64(index as u64))
            .zip(&folds)
            .map(|(point, &value)| Constraint {
                weights: MultilinearPoint::expand_from_univariate(
                    EF::from(point),
                    params.num_variables,
                ),
                sum: value,
            })
            .collect();

        Ok(stir_constraints)
    }

    pub fn verify_stir_challenges_batched(
        &self,
        verifier_state: &mut VerifierState<PF<F>, EF, impl FSChallenger<F>>,
        params: &RoundConfig<F>,
        commitment_a: &ParsedCommitment<F, EF, DIGEST_ELEMS>,
        commitment_b: &ParsedCommitment<F, EF, DIGEST_ELEMS>,
        folding_randomness: &MultilinearPoint<EF>,
        round_index: usize,
    ) -> ProofResult<Vec<Constraint<EF>>>
    where
        H: CryptographicHasher<PF<F>, [PF<F>; DIGEST_ELEMS]>
            + CryptographicHasher<PF<F>, [PF<F>; DIGEST_ELEMS]>
            + Sync,
        C: PseudoCompressionFunction<[PF<F>; DIGEST_ELEMS], 2>
            + PseudoCompressionFunction<[PF<F>; DIGEST_ELEMS], 2>
            + Sync,
        [PF<F>; DIGEST_ELEMS]: Serialize + for<'de> Deserialize<'de>,
    {
        let leafs_base_field = round_index == 0;

        verifier_state.check_pow_grinding(params.pow_bits)?;

        let stir_challenges_indexes = get_challenge_stir_queries(
            params.domain_size >> params.folding_factor,
            params.num_queries,
            verifier_state,
        );

        // dbg!(&stir_challenges_indexes);
        // dbg!(verifier_state.challenger().state());

        let dimensions_a = vec![Dimensions {
            height: params.domain_size >> params.folding_factor,
            width: 1 << params.folding_factor,
        }];
        let answers_a = self.verify_merkle_proof(
            verifier_state,
            &commitment_a.root,
            &stir_challenges_indexes,
            &dimensions_a,
            leafs_base_field,
            round_index,
            0,
        )?;

        // WE ASSUME FOR SIMPLICITY THAT LOG_INV_RATE_A = LOG_INV_RATE_B
        let vars_diff = commitment_a.num_variables - commitment_b.num_variables;
        assert!(vars_diff < params.folding_factor);
        let dimensions_b = vec![Dimensions {
            height: params.domain_size >> params.folding_factor,
            width: 1 << (params.folding_factor - vars_diff),
        }];
        let answers_b = self.verify_merkle_proof(
            verifier_state,
            &commitment_b.root,
            &stir_challenges_indexes,
            &dimensions_b,
            leafs_base_field,
            round_index,
            vars_diff,
        )?;

        // Compute STIR Constraints
        let folds: Vec<_> = answers_a
            .into_iter()
            .zip(answers_b)
            .map(|(answer_a, answer_b)| {
                let eval_a = EvaluationsList::new(answer_a).evaluate(&MultilinearPoint(
                    folding_randomness[..folding_randomness.len() - 1].to_vec(),
                ));
                let vars_b = answer_b.len().ilog2() as usize;
                let eval_b = EvaluationsList::new(answer_b)
                    .evaluate(&MultilinearPoint(folding_randomness[..vars_b].to_vec()));
                eval_a * folding_randomness[folding_randomness.len() - 1]
                    + eval_b
                        * folding_randomness[vars_b..]
                            .iter()
                            .map(|&x| EF::ONE - x)
                            .product::<EF>()
            })
            .collect();

        let stir_constraints = stir_challenges_indexes
            .iter()
            .map(|&index| params.folded_domain_gen.exp_u64(index as u64))
            .zip(&folds)
            .map(|(point, &value)| Constraint {
                weights: MultilinearPoint::expand_from_univariate(
                    EF::from(point),
                    params.num_variables,
                ),
                sum: value,
            })
            .collect();

        Ok(stir_constraints)
    }

    pub fn verify_merkle_proof(
        &self,
        verifier_state: &mut VerifierState<PF<F>, EF, impl FSChallenger<F>>,
        root: &Hash<PF<F>, PF<F>, DIGEST_ELEMS>,
        indices: &[usize],
        dimensions: &[Dimensions],
        leafs_base_field: bool,
        round_index: usize,
        var_shift: usize,
    ) -> ProofResult<Vec<Vec<EF>>>
    where
        H: CryptographicHasher<PF<F>, [PF<F>; DIGEST_ELEMS]>
            + CryptographicHasher<PF<F>, [PF<F>; DIGEST_ELEMS]>
            + Sync,
        C: PseudoCompressionFunction<[PF<F>; DIGEST_ELEMS], 2>
            + PseudoCompressionFunction<[PF<F>; DIGEST_ELEMS], 2>
            + Sync,
        [PF<F>; DIGEST_ELEMS]: Serialize + for<'de> Deserialize<'de>,
    {
        let mmcs = MerkleTreeMmcs::<PF<F>, PF<F>, H, C, DIGEST_ELEMS>::new(
            self.merkle_hash.clone(),
            self.merkle_compress.clone(),
        );
        let extension_mmcs_f = ExtensionMmcs::<PF<F>, F, _>::new(mmcs.clone());
        let extension_mmcs_ef = ExtensionMmcs::<PF<F>, EF, _>::new(mmcs.clone());

        // Branch depending on whether the committed leafs are base field or extension field.
        let res = if leafs_base_field {
            // Merkle leaves
            let mut answers = Vec::<Vec<F>>::new();
            let merkle_leaf_size = 1 << (self.folding_factor.at_round(round_index) - var_shift);
            for _ in 0..indices.len() {
                answers.push(pack_scalars_to_extension::<PF<F>, F>(
                    &verifier_state.receive_hint_base_scalars(
                        merkle_leaf_size * <F as BasedVectorSpace<PF<F>>>::DIMENSION,
                    )?,
                ));
            }

            // Merkle proofs
            let mut merkle_proofs = Vec::new();
            for _ in 0..indices.len() {
                let mut merkle_path = vec![];
                for _ in 0..self.merkle_tree_height(round_index) {
                    let digest: [PF<F>; DIGEST_ELEMS] = verifier_state
                        .receive_hint_base_scalars(DIGEST_ELEMS)?
                        .try_into()
                        .unwrap();
                    merkle_path.push(digest);
                }
                merkle_proofs.push(merkle_path);
            }

            // For each queried index:
            for (i, &index) in indices.iter().enumerate() {
                // Verify the Merkle opening for the claimed leaf against the Merkle root.
                extension_mmcs_f
                    .verify_batch(
                        root,
                        dimensions,
                        index,
                        BatchOpeningRef {
                            opened_values: &[answers[i].clone()],
                            opening_proof: &merkle_proofs[i],
                        },
                    )
                    .map_err(|_| ProofError::InvalidProof)?;
            }

            // Convert the base field values to EF and collect them into a result vector.
            answers
                .into_iter()
                .map(|inner| inner.iter().map(|&f_el| f_el.into()).collect())
                .collect()
        } else {
            // Merkle leaves
            let mut answers = vec![];
            let merkle_leaf_size = 1 << self.folding_factor.at_round(round_index);
            for _ in 0..indices.len() {
                answers.push(verifier_state.receive_hint_extension_scalars(merkle_leaf_size)?);
            }

            // Merkle proofs
            let mut merkle_proofs = Vec::new();
            for _ in 0..indices.len() {
                let mut merkle_path = vec![];
                for _ in 0..self.merkle_tree_height(round_index) {
                    let digest: [PF<F>; DIGEST_ELEMS] = verifier_state
                        .receive_hint_base_scalars(DIGEST_ELEMS)?
                        .try_into()
                        .unwrap();
                    merkle_path.push(digest);
                }
                merkle_proofs.push(merkle_path);
            }

            // For each queried index:
            for (i, &index) in indices.iter().enumerate() {
                // Verify the Merkle opening against the extension MMCS.
                extension_mmcs_ef
                    .verify_batch(
                        root,
                        dimensions,
                        index,
                        BatchOpeningRef {
                            opened_values: &[answers[i].clone()],
                            opening_proof: &merkle_proofs[i],
                        },
                    )
                    .map_err(|_| ProofError::InvalidProof)?;
            }

            // Return the extension field answers as-is.
            answers
        };

        // Return the verified leaf values.
        Ok(res)
    }

    /// Evaluate a batch of constraint polynomials at a given multilinear point.
    ///
    /// This function computes the combined weighted value of constraints across all rounds.
    /// Each constraint is either directly evaluated at the input point (`MultilinearPoint`)
    /// or substituted with a deferred evaluation result, depending on the constraint type.
    ///
    /// The final result is the sum of each constraint's value, scaled by its corresponding
    /// challenge randomness (used in the linear combination step of the sumcheck protocol).
    ///
    /// # Arguments
    /// - `constraints`: A list of tuples, where each tuple corresponds to a round and contains:
    ///     - A vector of challenge randomness values (used to weight each constraint),
    ///     - A vector of `Constraint<EF>` objects for that round.
    /// - `deferred`: Precomputed evaluations used for deferred constraints.
    /// - `point`: The multilinear point at which to evaluate the constraint polynomials.
    ///
    /// # Returns
    /// The combined evaluation result of all weighted constraints across rounds at the given point.
    ///
    /// # Panics
    /// Panics if:
    /// - Any round's `randomness.len()` does not match `constraints.len()`,
    /// - A deferred constraint is encountered but `deferred` has been exhausted.
    fn eval_constraints_poly(
        &self,
        constraints: &[(Vec<EF>, Vec<Constraint<EF>>)],
        mut point: MultilinearPoint<EF>,
    ) -> EF {
        let mut num_variables = self.num_variables;
        let mut value = EF::ZERO;

        for (round, (randomness, constraints)) in constraints.iter().enumerate() {
            assert_eq!(randomness.len(), constraints.len());
            if round > 0 {
                num_variables -= self.folding_factor.at_round(round - 1);
                point = MultilinearPoint(point[..num_variables].to_vec());
            }
            value += constraints
                .iter()
                .zip(randomness)
                .map(|(constraint, &randomness)| {
                    let value = constraint.weights.eq_poly_outside(&point);
                    value * randomness
                })
                .sum::<EF>();
        }
        value
    }
}

impl<F, EF, H, C, const DIGEST_ELEMS: usize> Deref for Verifier<'_, F, EF, H, C, DIGEST_ELEMS>
where
    F: Field,
    EF: ExtensionField<F>,
{
    type Target = WhirConfig<F, EF, H, C, DIGEST_ELEMS>;

    fn deref(&self) -> &Self::Target {
        self.0
    }
}
