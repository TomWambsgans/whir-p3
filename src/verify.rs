use std::{fmt::Debug, marker::PhantomData};

use multilinear_toolkit::prelude::*;
use p3_field::{BasedVectorSpace, ExtensionField, Field, TwoAdicField};
use p3_matrix::Dimensions;

use crate::*;

#[derive(Debug, Clone)]
pub struct ParsedCommitment<F: Field, EF: ExtensionField<F>> {
    pub num_variables: usize,
    pub root: [PF<EF>; DIGEST_ELEMS],
    pub ood_points: Vec<EF>,
    pub ood_answers: Vec<EF>,
    pub base_field: PhantomData<F>,
}

impl<F: Field, EF: ExtensionField<F>> ParsedCommitment<F, EF> {
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

impl<'a, EF> WhirConfig<EF>
where
    EF: TwoAdicField + ExtensionField<PF<EF>>,
{
    pub fn parse_commitment<F: TwoAdicField>(
        &self,
        verifier_state: &mut VerifierState<PF<EF>, EF, impl FSChallenger<EF>>,
    ) -> ProofResult<ParsedCommitment<F, EF>>
    where
        EF: ExtensionField<F>,
    {
        ParsedCommitment::<F, EF>::parse(
            verifier_state,
            self.num_variables,
            self.committment_ood_samples,
        )
    }
}

impl<'a, EF> WhirConfig<EF>
where
    EF: TwoAdicField + ExtensionField<PF<EF>>,
    PF<EF>: TwoAdicField,
{
    #[allow(clippy::too_many_lines)]
    pub fn batch_verify<F: TwoAdicField>(
        &self,
        verifier_state: &mut VerifierState<PF<EF>, EF, impl FSChallenger<EF>>,
        parsed_commitment_a: &ParsedCommitment<F, EF>,
        statement_a: Vec<Evaluation<EF>>,
        parsed_commitment_b: &ParsedCommitment<EF, EF>,
        statement_b: Vec<Evaluation<EF>>,
    ) -> ProofResult<MultilinearPoint<EF>>
    where
        EF: ExtensionField<F>,
        F: ExtensionField<PF<EF>>,
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
                c.point.insert(0, EF::ONE);
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
                    c.point.splice(0..0, vec![EF::ZERO; ending_zeros]);
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
            let new_commitment = ParsedCommitment::<F, EF>::parse(
                verifier_state,
                round_params.num_variables,
                round_params.ood_samples,
            )?;

            // Verify in-domain challenges on the previous commitment.
            let stir_constraints = if round_index == 0 {
                self.verify_stir_challenges_batched::<F>(
                    verifier_state,
                    round_params,
                    parsed_commitment_a,
                    parsed_commitment_b,
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
            self.eval_constraints_poly(&round_constraints, folding_randomness.clone(), true);

        // Check the final sumcheck evaluation
        let final_value = final_evaluations.evaluate(&final_sumcheck_randomness);
        if claimed_sum != evaluation_of_weights * final_value {
            return Err(ProofError::InvalidProof);
        }

        Ok(folding_randomness)
    }

    #[allow(clippy::too_many_lines)]
    pub fn verify<F: TwoAdicField>(
        &self,
        verifier_state: &mut VerifierState<PF<EF>, EF, impl FSChallenger<EF>>,
        parsed_commitment: &ParsedCommitment<F, EF>,
        statement: Vec<Evaluation<EF>>,
    ) -> ProofResult<MultilinearPoint<EF>>
    where
        EF: ExtensionField<F>,
        F: ExtensionField<PF<EF>>,
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
            let new_commitment = ParsedCommitment::<F, EF>::parse(
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
            self.eval_constraints_poly(&round_constraints, folding_randomness.clone(), false);

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

    fn verify_stir_challenges<F: Field>(
        &self,
        verifier_state: &mut VerifierState<PF<EF>, EF, impl FSChallenger<EF>>,
        params: &RoundConfig<EF>,
        commitment: &ParsedCommitment<F, EF>,
        folding_randomness: &MultilinearPoint<EF>,
        round_index: usize,
    ) -> ProofResult<Vec<Evaluation<EF>>>
    where
        EF: ExtensionField<F>,
        F: ExtensionField<PF<EF>>,
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
        let answers = self.verify_merkle_proof::<F>(
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

    fn verify_stir_challenges_batched<F: Field>(
        &self,
        verifier_state: &mut VerifierState<PF<EF>, EF, impl FSChallenger<EF>>,
        params: &RoundConfig<EF>,
        commitment_a: &ParsedCommitment<F, EF>,
        commitment_b: &ParsedCommitment<EF, EF>,
        folding_randomness: &MultilinearPoint<EF>,
        round_index: usize,
    ) -> ProofResult<Vec<Evaluation<EF>>>
    where
        EF: ExtensionField<F>,
        F: ExtensionField<PF<EF>>,
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
        let answers_a = self.verify_merkle_proof::<F>(
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
            .verify_merkle_proof::<EF>(
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
                let vars_a = log2_strict_usize(answer_a.len());
                let vars_b = log2_strict_usize(answer_b.len());
                let a_trunc = folding_randomness[1..].to_vec();
                let eval_a = answer_a.evaluate(&MultilinearPoint(a_trunc));
                let b_trunc = folding_randomness[vars_a - vars_b + 1..].to_vec();
                let eval_b = answer_b.evaluate(&MultilinearPoint(b_trunc));
                let last_fold_rand_a = folding_randomness[0];
                let last_fold_rand_b = folding_randomness[..vars_a - vars_b + 1]
                    .iter()
                    .map(|&x| EF::ONE - x)
                    .product::<EF>();
                eval_a * last_fold_rand_a + eval_b * last_fold_rand_b
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

    fn verify_merkle_proof<F: Field>(
        &self,
        verifier_state: &mut VerifierState<PF<EF>, EF, impl FSChallenger<EF>>,
        root: &[PF<EF>; DIGEST_ELEMS],
        indices: &[usize],
        dimensions: &[Dimensions],
        leafs_base_field: bool,
        round_index: usize,
        var_shift: usize,
    ) -> ProofResult<Vec<Vec<EF>>>
    where
        EF: ExtensionField<F>,
        F: ExtensionField<PF<EF>>,
    {
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
                let merkle_path = verifier_state.receive_hint_merkle_path()?;
                if merkle_path.len() != self.merkle_tree_height(round_index) {
                    return Err(ProofError::InvalidProof);
                }
                merkle_proofs.push(merkle_path);
            }

            // For each queried index:
            for (i, &index) in indices.iter().enumerate() {
                // Verify the Merkle opening for the claimed leaf against the Merkle root.
                if !merkle_verify::<PF<EF>, F>(
                    *root,
                    index,
                    dimensions[0],
                    answers[i].clone(),
                    &merkle_proofs[i],
                ) {
                    return Err(ProofError::InvalidProof);
                }
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
                let merkle_path = verifier_state.receive_hint_merkle_path()?;
                if merkle_path.len() != self.merkle_tree_height(round_index) {
                    return Err(ProofError::InvalidProof);
                }
                merkle_proofs.push(merkle_path);
            }

            // For each queried index:
            for (i, &index) in indices.iter().enumerate() {
                // Verify the Merkle opening against the extension MMCS.
                if !merkle_verify::<PF<EF>, EF>(
                    *root,
                    index,
                    dimensions[0],
                    answers[i].clone(),
                    &merkle_proofs[i],
                ) {
                    return Err(ProofError::InvalidProof);
                }
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
        batched: bool,
    ) -> EF {
        let mut value = EF::ZERO;

        for (round, (randomness, constraints)) in constraints.iter().enumerate() {
            assert_eq!(randomness.len(), constraints.len());
            if round > 0 {
                let mut k = self.folding_factor.at_round(round - 1);
                if round == 1 && batched {
                    k += 1;
                }
                point = MultilinearPoint(point[k..].to_vec());
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
