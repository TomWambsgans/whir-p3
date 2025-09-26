use fiat_shamir::*;
use multilinear_toolkit::prelude::*;
use p3_commit::{BatchOpeningRef, ExtensionMmcs, Mmcs};
use p3_field::{BasedVectorSpace, ExtensionField, Field, TwoAdicField};
use p3_matrix::Dimensions;
use p3_merkle_tree::MerkleTreeMmcs;
use p3_symmetric::{CryptographicHasher, Hash, PseudoCompressionFunction};
use serde::{Deserialize, Serialize};
use std::{fmt::Debug, marker::PhantomData};

use crate::*;

#[derive(Debug, Clone)]
pub struct ParsedCommitment<F: Field, EF: ExtensionField<F>, const DIGEST_ELEMS: usize> {
    pub num_variables: usize,
    pub root: Hash<PF<EF>, PF<EF>, DIGEST_ELEMS>,
    pub ood_points: Vec<EF>,
    pub ood_answers: Vec<EF>,
    pub base_field: PhantomData<F>,
}

impl<F: Field, EF: ExtensionField<F>, const DIGEST_ELEMS: usize>
    ParsedCommitment<F, EF, DIGEST_ELEMS>
{
    pub fn parse(
        verifier_state: &mut VerifierState<PF<EF>, EF, impl FSChallenger<EF>>,
        num_variables: usize,
        ood_samples: usize,
    ) -> ProofResult<Self>
    where
        F: TwoAdicField,
        EF: ExtensionField<F> + TwoAdicField,
        EF: ExtensionField<PF<EF>>,
    {
        let root = verifier_state
            .next_base_scalars_const::<DIGEST_ELEMS>()?
            .into();
        let mut ood_points = EF::zero_vec(ood_samples);
        let ood_answers = if ood_samples > 0 {
            for ood_point in &mut ood_points {
                *ood_point = verifier_state.sample();
            }

            verifier_state.next_extension_scalars_vec(ood_samples)?
        } else {
            Vec::new()
        };
        Ok(Self {
            num_variables,
            root,
            ood_points,
            ood_answers,
            base_field: PhantomData,
        })
    }

    pub fn oods_constraints(&self) -> Vec<Evaluation<EF>> {
        self.ood_points
            .iter()
            .zip(&self.ood_answers)
            .map(|(&point, &eval)| Evaluation {
                point: MultilinearPoint::expand_from_univariate(point, self.num_variables),
                value: eval,
            })
            .collect()
    }
}

impl<'a, F, EF, H, C, const DIGEST_ELEMS: usize> WhirConfig<F, EF, H, C, DIGEST_ELEMS>
where
    F: TwoAdicField,
    EF: ExtensionField<F> + TwoAdicField + ExtensionField<PF<EF>>,
{
    pub fn parse_commitment(
        &self,
        verifier_state: &mut VerifierState<PF<EF>, EF, impl FSChallenger<EF>>,
    ) -> ProofResult<ParsedCommitment<F, EF, DIGEST_ELEMS>> {
        ParsedCommitment::<F, EF, DIGEST_ELEMS>::parse(
            verifier_state,
            self.num_variables,
            self.committment_ood_samples,
        )
    }
}

impl<'a, F, EF, H, C, const DIGEST_ELEMS: usize> WhirConfig<F, EF, H, C, DIGEST_ELEMS>
where
    F: TwoAdicField,
    EF: ExtensionField<F> + TwoAdicField + ExtensionField<PF<EF>>,
    F: ExtensionField<PF<EF>>,
{
    #[allow(clippy::too_many_lines)]
    pub fn batch_verify(
        &self,
        verifier_state: &mut VerifierState<PF<EF>, EF, impl FSChallenger<EF>>,
        parsed_commitment_a: &ParsedCommitment<F, EF, DIGEST_ELEMS>,
        statement_a: Vec<Evaluation<EF>>,
        parsed_commitment_b: &ParsedCommitment<EF, EF, DIGEST_ELEMS>,
        statement_b: Vec<Evaluation<EF>>,
    ) -> ProofResult<MultilinearPoint<EF>>
    where
        H: CryptographicHasher<PF<EF>, [PF<EF>; DIGEST_ELEMS]>
            + CryptographicHasher<PF<EF>, [PF<EF>; DIGEST_ELEMS]>
            + Sync,
        C: PseudoCompressionFunction<[PF<EF>; DIGEST_ELEMS], 2>
            + PseudoCompressionFunction<[PF<EF>; DIGEST_ELEMS], 2>
            + Sync,
        [PF<EF>; DIGEST_ELEMS]: Serialize + for<'de> Deserialize<'de>,
    {
        assert!(
            statement_a
                .iter()
                .all(|c| c.point.len() == parsed_commitment_a.num_variables)
        );
        assert!(
            statement_b
                .iter()
                .all(|c| c.point.len() == parsed_commitment_b.num_variables)
        );

        let mut round_constraints = Vec::new();
        let mut round_folding_randomness = Vec::new();
        let mut claimed_sum = EF::ZERO;
        let mut prev_commitment = None;

        // Combine OODS and statement constraints to claimed_sum
        let mut constraints: Vec<_> = parsed_commitment_a
            .oods_constraints()
            .into_iter()
            .chain(statement_a)
            .map(|mut c| {
                c.point.push(EF::ONE);
                c
            })
            .collect();

        constraints.extend(
            parsed_commitment_b
                .oods_constraints()
                .into_iter()
                .chain(statement_b)
                .map(|mut c| {
                    let ending_zeros =
                        parsed_commitment_a.num_variables + 1 - parsed_commitment_b.num_variables;
                    c.point.extend(vec![EF::ZERO; ending_zeros]);
                    c
                }),
        );

        let combination_randomness =
            self.combine_constraints(verifier_state, &mut claimed_sum, &constraints)?;
        round_constraints.push((combination_randomness, constraints));

        // Initial sumcheck
        let folding_randomness = verify_sumcheck_rounds::<F, EF>(
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
            let constraints: Vec<Evaluation<EF>> = new_commitment
                .oods_constraints()
                .into_iter()
                .chain(stir_constraints.into_iter())
                .collect();

            let combination_randomness =
                self.combine_constraints(verifier_state, &mut claimed_sum, &constraints)?;
            round_constraints.push((combination_randomness.clone(), constraints));

            let folding_randomness = verify_sumcheck_rounds::<F, EF>(
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
        let final_evaluations = verifier_state.next_extension_scalars_vec(n_final_coeffs)?;

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
            .all(|c| verify_constraint(c, &final_evaluations))
            .then_some(())
            .ok_or(ProofError::InvalidProof)?;

        let final_sumcheck_randomness = verify_sumcheck_rounds::<F, EF>(
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

    #[allow(clippy::too_many_lines)]
    pub fn verify(
        &self,
        verifier_state: &mut VerifierState<PF<EF>, EF, impl FSChallenger<EF>>,
        parsed_commitment: &ParsedCommitment<F, EF, DIGEST_ELEMS>,
        statement: Vec<Evaluation<EF>>,
    ) -> ProofResult<MultilinearPoint<EF>>
    where
        H: CryptographicHasher<PF<EF>, [PF<EF>; DIGEST_ELEMS]>
            + CryptographicHasher<PF<EF>, [PF<EF>; DIGEST_ELEMS]>
            + Sync,
        C: PseudoCompressionFunction<[PF<EF>; DIGEST_ELEMS], 2>
            + PseudoCompressionFunction<[PF<EF>; DIGEST_ELEMS], 2>
            + Sync,
        [PF<EF>; DIGEST_ELEMS]: Serialize + for<'de> Deserialize<'de>,
    {
        assert!(
            statement
                .iter()
                .all(|c| c.point.len() == parsed_commitment.num_variables)
        );

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
            .chain(statement)
            .collect();
        let combination_randomness =
            self.combine_constraints(verifier_state, &mut claimed_sum, &constraints)?;
        round_constraints.push((combination_randomness, constraints));

        // Initial sumcheck
        let folding_randomness = verify_sumcheck_rounds::<F, EF>(
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
            let constraints: Vec<Evaluation<EF>> = new_commitment
                .oods_constraints()
                .into_iter()
                .chain(stir_constraints)
                .collect();

            let combination_randomness =
                self.combine_constraints(verifier_state, &mut claimed_sum, &constraints)?;
            round_constraints.push((combination_randomness.clone(), constraints));

            let folding_randomness = verify_sumcheck_rounds::<F, EF>(
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
        let final_evaluations = verifier_state.next_extension_scalars_vec(n_final_coeffs)?;

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
            .all(|c| verify_constraint(c, &final_evaluations))
            .then_some(())
            .ok_or(ProofError::InvalidProof)
            .unwrap();

        let final_sumcheck_randomness = verify_sumcheck_rounds::<F, EF>(
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
                .flat_map(|poly| poly.0.into_iter())
                .collect(),
        );

        let evaluation_of_weights =
            self.eval_constraints_poly(&round_constraints, folding_randomness.clone());

        // Check the final sumcheck evaluation
        let final_value = final_evaluations.evaluate(&final_sumcheck_randomness);
        if claimed_sum != evaluation_of_weights * final_value {
            panic!();
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
    pub(crate) fn combine_constraints(
        &self,
        verifier_state: &mut VerifierState<PF<EF>, EF, impl FSChallenger<EF>>,
        claimed_sum: &mut EF,
        constraints: &[Evaluation<EF>],
    ) -> ProofResult<Vec<EF>> {
        let combination_randomness_gen: EF = verifier_state.sample();
        let combination_randomness: Vec<_> = combination_randomness_gen
            .powers()
            .take(constraints.len())
            .collect();
        *claimed_sum += constraints
            .iter()
            .zip(&combination_randomness)
            .map(|(c, &rand)| rand * c.value)
            .sum::<EF>();

        Ok(combination_randomness)
    }

    fn verify_stir_challenges(
        &self,
        verifier_state: &mut VerifierState<PF<EF>, EF, impl FSChallenger<EF>>,
        params: &RoundConfig<F>,
        commitment: &ParsedCommitment<F, EF, DIGEST_ELEMS>,
        folding_randomness: &MultilinearPoint<EF>,
        round_index: usize,
    ) -> ProofResult<Vec<Evaluation<EF>>>
    where
        H: CryptographicHasher<PF<EF>, [PF<EF>; DIGEST_ELEMS]>
            + CryptographicHasher<PF<EF>, [PF<EF>; DIGEST_ELEMS]>
            + Sync,
        C: PseudoCompressionFunction<[PF<EF>; DIGEST_ELEMS], 2>
            + PseudoCompressionFunction<[PF<EF>; DIGEST_ELEMS], 2>
            + Sync,
        [PF<EF>; DIGEST_ELEMS]: Serialize + for<'de> Deserialize<'de>,
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
            .map(|answers| answers.evaluate(folding_randomness))
            .collect();

        let stir_constraints = stir_challenges_indexes
            .iter()
            .map(|&index| params.folded_domain_gen.exp_u64(index as u64))
            .zip(&folds)
            .map(|(point, &value)| {
                Evaluation::new(
                    MultilinearPoint::expand_from_univariate(EF::from(point), params.num_variables),
                    value,
                )
            })
            .collect();

        Ok(stir_constraints)
    }

    fn verify_stir_challenges_batched(
        &self,
        verifier_state: &mut VerifierState<PF<EF>, EF, impl FSChallenger<EF>>,
        params: &RoundConfig<F>,
        commitment_a: &ParsedCommitment<F, EF, DIGEST_ELEMS>,
        commitment_b: &ParsedCommitment<EF, EF, DIGEST_ELEMS>,
        folding_randomness: &MultilinearPoint<EF>,
        round_index: usize,
    ) -> ProofResult<Vec<Evaluation<EF>>>
    where
        H: CryptographicHasher<PF<EF>, [PF<EF>; DIGEST_ELEMS]>
            + CryptographicHasher<PF<EF>, [PF<EF>; DIGEST_ELEMS]>
            + Sync,
        C: PseudoCompressionFunction<[PF<EF>; DIGEST_ELEMS], 2>
            + PseudoCompressionFunction<[PF<EF>; DIGEST_ELEMS], 2>
            + Sync,
        [PF<EF>; DIGEST_ELEMS]: Serialize + for<'de> Deserialize<'de>,
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
        let answers_b = self
            .verify_merkle_proof(
                verifier_state,
                &commitment_b.root,
                &stir_challenges_indexes,
                &dimensions_b,
                false,
                round_index,
                vars_diff,
            )
            .unwrap();

        // Compute STIR Constraints
        let folds: Vec<_> = answers_a
            .into_iter()
            .zip(answers_b)
            .map(|(answer_a, answer_b)| {
                let eval_a = answer_a.evaluate(&MultilinearPoint(
                    folding_randomness[..folding_randomness.len() - 1].to_vec(),
                ));
                let vars_b = answer_b.len().ilog2() as usize;
                let eval_b =
                    answer_b.evaluate(&MultilinearPoint(folding_randomness[..vars_b].to_vec()));
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
            .map(|(point, &value)| {
                Evaluation::new(
                    MultilinearPoint::expand_from_univariate(EF::from(point), params.num_variables),
                    value,
                )
            })
            .collect();

        Ok(stir_constraints)
    }

    fn verify_merkle_proof(
        &self,
        verifier_state: &mut VerifierState<PF<EF>, EF, impl FSChallenger<EF>>,
        root: &Hash<PF<EF>, PF<EF>, DIGEST_ELEMS>,
        indices: &[usize],
        dimensions: &[Dimensions],
        leafs_base_field: bool,
        round_index: usize,
        var_shift: usize,
    ) -> ProofResult<Vec<Vec<EF>>>
    where
        H: CryptographicHasher<PF<EF>, [PF<EF>; DIGEST_ELEMS]>
            + CryptographicHasher<PF<EF>, [PF<EF>; DIGEST_ELEMS]>
            + Sync,
        C: PseudoCompressionFunction<[PF<EF>; DIGEST_ELEMS], 2>
            + PseudoCompressionFunction<[PF<EF>; DIGEST_ELEMS], 2>
            + Sync,
        [PF<EF>; DIGEST_ELEMS]: Serialize + for<'de> Deserialize<'de>,
    {
        let mmcs = MerkleTreeMmcs::<PF<EF>, PF<EF>, H, C, DIGEST_ELEMS>::new(
            self.merkle_hash.clone(),
            self.merkle_compress.clone(),
        );
        let extension_mmcs_f = ExtensionMmcs::<PF<EF>, F, _>::new(mmcs.clone());
        let extension_mmcs_ef = ExtensionMmcs::<PF<EF>, EF, _>::new(mmcs.clone());

        // Branch depending on whether the committed leafs are base field or extension field.
        let res = if leafs_base_field {
            // Merkle leaves
            let mut answers = Vec::<Vec<F>>::new();
            let merkle_leaf_size = 1 << (self.folding_factor.at_round(round_index) - var_shift);
            for _ in 0..indices.len() {
                answers.push(pack_scalars_to_extension::<PF<EF>, F>(
                    &verifier_state.receive_hint_base_scalars(
                        merkle_leaf_size * <F as BasedVectorSpace<PF<EF>>>::DIMENSION,
                    )?,
                ));
            }

            // Merkle proofs
            let mut merkle_proofs = Vec::new();
            for _ in 0..indices.len() {
                let mut merkle_path = vec![];
                for _ in 0..self.merkle_tree_height(round_index) {
                    let digest: [PF<EF>; DIGEST_ELEMS] = verifier_state
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
            let merkle_leaf_size = if round_index == 0 {
                1 << (self.folding_factor.at_round(round_index) - var_shift)
            } else {
                1 << self.folding_factor.at_round(round_index)
            };
            for _ in 0..indices.len() {
                answers.push(verifier_state.receive_hint_extension_scalars(merkle_leaf_size)?);
            }

            // Merkle proofs
            let mut merkle_proofs = Vec::new();
            for _ in 0..indices.len() {
                let mut merkle_path = vec![];
                for _ in 0..self.merkle_tree_height(round_index) {
                    let digest: [PF<EF>; DIGEST_ELEMS] = verifier_state
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

    fn eval_constraints_poly(
        &self,
        constraints: &[(Vec<EF>, Vec<Evaluation<EF>>)],
        mut point: MultilinearPoint<EF>,
    ) -> EF {
        let mut value = EF::ZERO;

        for (round, (randomness, constraints)) in constraints.iter().enumerate() {
            assert_eq!(randomness.len(), constraints.len());
            if round > 0 {
                point = MultilinearPoint(point[self.folding_factor.at_round(round - 1)..].to_vec());
            }
            value += constraints
                .iter()
                .zip(randomness)
                .map(|(constraint, &randomness)| {
                    let value = constraint.point.eq_poly_outside(&point);
                    value * randomness
                })
                .sum::<EF>();
        }
        value
    }
}

fn verify_constraint<EF: Field>(constraint: &Evaluation<EF>, poly: &[EF]) -> bool {
    poly.evaluate(&constraint.point) == constraint.value
}

/// The full vector of folding randomness values, in reverse round order.
type SumcheckRandomness<F> = MultilinearPoint<F>;

pub(crate) fn verify_sumcheck_rounds<F, EF>(
    verifier_state: &mut VerifierState<PF<EF>, EF, impl FSChallenger<EF>>,
    claimed_sum: &mut EF,
    rounds: usize,
    _pow_bits: usize,
) -> ProofResult<SumcheckRandomness<EF>>
where
    F: TwoAdicField,
    EF: ExtensionField<F> + TwoAdicField + ExtensionField<PF<EF>>,
{
    // Preallocate vector to hold the randomness values
    let mut randomness = Vec::with_capacity(rounds);

    for _ in 0..rounds {
        // Extract the 3 evaluations of the quadratic sumcheck polynomial h(X)
        let coeffs: [_; 3] = verifier_state.next_extension_scalars_const()?;

        let poly = DensePolynomial::new(coeffs.to_vec());

        // Verify claimed sum is consistent with polynomial
        if poly.evaluate(EF::ZERO) + poly.evaluate(EF::ONE) != *claimed_sum {
            return Err(ProofError::InvalidProof);
        }

        // TODO: re-enable PoW grinding
        // verifier_state.check_pow_grinding(pow_bits)?;

        // Sample the next verifier folding randomness rᵢ
        let rand: EF = verifier_state.sample();

        // Update claimed sum using folding randomness
        *claimed_sum = poly.evaluate(rand);

        // Store this round’s randomness
        randomness.push(rand);
    }

    Ok(MultilinearPoint(randomness))
}
