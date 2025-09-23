use p3_commit::{ExtensionMmcs, Mmcs};
use p3_field::{ExtensionField, Field, TwoAdicField};
use p3_matrix::dense::RowMajorMatrix;
use p3_merkle_tree::MerkleTreeMmcs;
use p3_symmetric::{CryptographicHasher, PseudoCompressionFunction};
use p3_util::log2_ceil_usize;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use tracing::{info_span, instrument};

use crate::{config::WhirConfig, sumcheck::SumcheckSingle, *};

pub type Proof<W, const DIGEST_ELEMS: usize> = Vec<Vec<[W; DIGEST_ELEMS]>>;
pub type Leafs<F> = Vec<Vec<F>>;

impl<F, EF, H, C, const DIGEST_ELEMS: usize> WhirConfig<F, EF, H, C, DIGEST_ELEMS>
where
    F: TwoAdicField,
    EF: ExtensionField<F> + TwoAdicField,
    F: ExtensionField<PF<EF>>,
    EF: ExtensionField<PF<EF>>,
{
    /// Validates that the total number of variables expected by the prover configuration
    /// matches the number implied by the folding schedule and the final rounds.
    ///
    /// This ensures that the recursive folding in the sumcheck protocol terminates
    /// precisely at the expected number of final variables.
    ///
    /// # Returns
    /// `true` if the parameter configuration is consistent, `false` otherwise.
    fn validate_parameters(&self) -> bool {
        self.num_variables
            == self.folding_factor.total_number(self.n_rounds()) + self.final_sumcheck_rounds
    }

    /// Validates that the public statement is compatible with the configured number of variables.
    ///
    /// Ensures the following:
    /// - The number of variables in the statement matches the prover's expectations
    /// - If no initial statement is used, the statement must be empty
    ///
    /// # Parameters
    /// - `statement`: The public constraints that the prover will use
    ///
    /// # Returns
    /// `true` if the statement structure is valid for this protocol instance.
    fn validate_statement(&self, statement: &[Evaluation<EF>]) -> bool {
        statement
            .iter()
            .all(|e| e.num_variables() == self.num_variables)
    }

    /// Validates that the witness satisfies the structural requirements of the WHIR prover.
    ///
    /// Checks the following conditions:
    /// - The number of OOD (out-of-domain) points equals the number of OOD answers
    /// - If no initial statement is used, the OOD data must be empty
    /// - The multilinear witness polynomial must match the expected number of variables
    ///
    /// # Parameters
    /// - `witness`: The private witness to be verified for structural consistency
    ///
    /// # Returns
    /// `true` if the witness structure matches expectations.
    ///
    /// # Panics
    /// - Panics if OOD lengths are inconsistent
    /// - Panics if OOD data is non-empty despite `initial_statement = false`
    fn validate_witness(&self, witness: &Witness<F, EF, DIGEST_ELEMS>, polynomial: &[F]) -> bool {
        assert_eq!(witness.ood_points.len(), witness.ood_answers.len());
        polynomial.num_variables() == self.num_variables
    }

    #[instrument(name = "WHIR prove", skip_all)]
    pub fn prove(
        &self,
        dft: &EvalsDft<PF<EF>>,
        prover_state: &mut ProverState<PF<EF>, EF, impl FSChallenger<EF>>,
        statement: Vec<Evaluation<EF>>,
        witness: Witness<F, EF, DIGEST_ELEMS>,
        polynomial: &[F],
    ) -> MultilinearPoint<EF>
    where
        H: CryptographicHasher<PF<EF>, [PF<EF>; DIGEST_ELEMS]>
            + CryptographicHasher<PFPacking<EF>, [PFPacking<EF>; DIGEST_ELEMS]>
            + Sync,
        C: PseudoCompressionFunction<[PF<EF>; DIGEST_ELEMS], 2>
            + PseudoCompressionFunction<[PFPacking<EF>; DIGEST_ELEMS], 2>
            + Sync,
        [PF<EF>; DIGEST_ELEMS]: Serialize + for<'de> Deserialize<'de>,
        PF<EF>: TwoAdicField,
    {
        // Validate parameters
        assert!(
            self.validate_parameters()
                && self.validate_statement(&statement)
                && self.validate_witness(&witness, &polynomial),
            "Invalid prover parameters, statement, or witness"
        );

        // Initialize the round state with inputs and initial polynomial data
        let mut round_state = RoundState::initialize_first_round_state(
            self,
            prover_state,
            statement,
            witness,
            polynomial,
        )
        .unwrap();

        // Run the WHIR protocol round-by-round
        for round in 0..=self.n_rounds() {
            self.round(round, dft, prover_state, &mut round_state)
                .unwrap();
        }

        // Reverse the vector of verifier challenges (used as evaluation point)
        //
        // These challenges were pushed in round order; we reverse them to use as a single
        // evaluation point for final statement consistency checks.
        let constraint_eval = MultilinearPoint(round_state.randomness_vec);

        constraint_eval
    }

    #[instrument(name = "WHIR batch prove", skip_all)]
    pub fn batch_prove(
        &self,
        dft: &EvalsDft<PF<EF>>,
        prover_state: &mut ProverState<PF<EF>, EF, impl FSChallenger<EF>>,
        statement_a: Vec<Evaluation<EF>>,
        witness_a: Witness<F, EF, DIGEST_ELEMS>,
        polynomial_a: &[F],
        statement_b: Vec<Evaluation<EF>>,
        witness_b: Witness<EF, EF, DIGEST_ELEMS>,
        polynomial_b: &[EF],
    ) -> MultilinearPoint<EF>
    where
        H: CryptographicHasher<PF<EF>, [PF<EF>; DIGEST_ELEMS]>
            + CryptographicHasher<PFPacking<EF>, [PFPacking<EF>; DIGEST_ELEMS]>
            + Sync,
        C: PseudoCompressionFunction<[PF<EF>; DIGEST_ELEMS], 2>
            + PseudoCompressionFunction<[PFPacking<EF>; DIGEST_ELEMS], 2>
            + Sync,
        [PF<EF>; DIGEST_ELEMS]: Serialize + for<'de> Deserialize<'de>,
        PF<EF>: TwoAdicField,
    {
        // Xn PolA + (1 - Xn) PolB

        assert_eq!(polynomial_a.num_variables(), self.num_variables,);
        assert!(polynomial_a.num_variables() >= polynomial_b.num_variables());
        // Initialize the round state with inputs and initial polynomial data
        let mut round_state = RoundState::initialize_first_round_state_batch(
            self,
            prover_state,
            statement_a,
            witness_a,
            polynomial_a,
            statement_b,
            witness_b,
            polynomial_b,
        )
        .unwrap();

        // Run the WHIR protocol round-by-round
        for round in 0..=self.n_rounds() {
            self.round(round, dft, prover_state, &mut round_state)
                .unwrap();
        }

        // Reverse the vector of verifier challenges (used as evaluation point)
        //
        // These challenges were pushed in round order; we reverse them to use as a single
        // evaluation point for final statement consistency checks.
        let constraint_eval = MultilinearPoint(round_state.randomness_vec);

        constraint_eval
    }

    fn round(
        &self,
        round_index: usize,
        dft: &EvalsDft<PF<EF>>,
        prover_state: &mut ProverState<PF<EF>, EF, impl FSChallenger<EF>>,
        round_state: &mut RoundState<F, EF, DIGEST_ELEMS>,
    ) -> ProofResult<()>
    where
        H: CryptographicHasher<PF<EF>, [PF<EF>; DIGEST_ELEMS]>
            + CryptographicHasher<PFPacking<EF>, [PFPacking<EF>; DIGEST_ELEMS]>
            + Sync,
        C: PseudoCompressionFunction<[PF<EF>; DIGEST_ELEMS], 2>
            + PseudoCompressionFunction<[PFPacking<EF>; DIGEST_ELEMS], 2>
            + Sync,
        [PF<EF>; DIGEST_ELEMS]: Serialize + for<'de> Deserialize<'de>,
        PF<EF>: TwoAdicField,
    {
        let folded_evaluations = &round_state.sumcheck_prover.evals;
        let num_variables = self.num_variables - self.folding_factor.total_number(round_index);
        assert_eq!(num_variables, folded_evaluations.num_variables());

        // Base case: final round reached
        if round_index == self.n_rounds() {
            return self.final_round(round_index, prover_state, round_state);
        }

        let round_params = &self.round_parameters[round_index];

        // Compute the folding factors for later use
        let folding_factor_next = self.folding_factor.at_round(round_index + 1);

        // Compute polynomial evaluations and build Merkle tree
        let domain_reduction = 1 << self.rs_reduction_factor(round_index);
        let new_domain_size = round_state.domain_size / domain_reduction;
        let inv_rate = new_domain_size / folded_evaluations.num_evals();
        let folded_matrix = info_span!("fold matrix").in_scope(|| {
            let evals_repeated = info_span!("repeating evals")
                .in_scope(|| parallel_repeat(folded_evaluations, inv_rate));
            // Do DFT on only interleaved polys to be folded.
            info_span!(
                "dft",
                height = evals_repeated.len() >> folding_factor_next,
                width = 1 << folding_factor_next
            )
            .in_scope(|| {
                dft.dft_algebra_batch_by_evals(RowMajorMatrix::new(
                    evals_repeated,
                    1 << folding_factor_next,
                ))
            })
        });

        let mmcs = MerkleTreeMmcs::<PFPacking<EF>, PFPacking<EF>, H, C, DIGEST_ELEMS>::new(
            self.merkle_hash.clone(),
            self.merkle_compress.clone(),
        );
        let extension_mmcs_f = ExtensionMmcs::<PF<EF>, F, _>::new(mmcs.clone());
        let extension_mmcs_ef = ExtensionMmcs::<PF<EF>, EF, _>::new(mmcs.clone());

        let (root, prover_data) =
            info_span!("commit matrix").in_scope(|| extension_mmcs_ef.commit_matrix(folded_matrix));

        prover_state.add_base_scalars(root.as_ref());

        // Handle OOD (Out-Of-Domain) samples
        let (ood_points, ood_answers) = sample_ood_points::<F, EF, _>(
            prover_state,
            round_params.ood_samples,
            num_variables,
            |point| info_span!("ood evaluation").in_scope(|| folded_evaluations.evaluate(point)),
        );

        prover_state.pow_grinding(round_params.pow_bits);

        // STIR Queries
        let (ood_challenges, stir_challenges, stir_challenges_indexes) = self
            .compute_stir_queries(
                prover_state,
                round_state,
                num_variables,
                round_params,
                &ood_points,
                round_index,
            )?;

        // Collect Merkle proofs for stir queries
        let stir_evaluations = match &round_state.merkle_prover_data {
            None => {
                if round_state.commitment_merkle_prover_data_b.is_some() {
                    let mut answers_a = Vec::<Vec<F>>::new();
                    let mut answers_b = Vec::<Vec<EF>>::new();

                    {
                        let mut merkle_proofs = Vec::new();
                        for challenge in &stir_challenges_indexes {
                            let commitment = extension_mmcs_f.open_batch(
                                *challenge,
                                &round_state.commitment_merkle_prover_data_a,
                            );
                            answers_a.push(commitment.opened_values[0].clone());
                            merkle_proofs.push(commitment.opening_proof);
                        }

                        // merkle leaves
                        for answer in &answers_a {
                            prover_state.hint_base_scalars(&flatten_scalars_to_base(answer));
                        }

                        // merkle authentication proof
                        for merkle_proof in &merkle_proofs {
                            for digest in merkle_proof {
                                prover_state.hint_base_scalars(digest);
                            }
                        }
                    }
                    {
                        let mut merkle_proofs = Vec::new();
                        for challenge in &stir_challenges_indexes {
                            let commitment = extension_mmcs_ef.open_batch(
                                *challenge,
                                round_state
                                    .commitment_merkle_prover_data_b
                                    .as_ref()
                                    .unwrap(),
                            );
                            answers_b.push(commitment.opened_values[0].clone());
                            merkle_proofs.push(commitment.opening_proof);
                        }

                        // merkle leaves
                        for answer in &answers_b {
                            prover_state.hint_base_scalars(&flatten_scalars_to_base(answer));
                        }

                        // merkle authentication proof
                        for merkle_proof in &merkle_proofs {
                            for digest in merkle_proof {
                                prover_state.hint_base_scalars(digest);
                            }
                        }
                    }

                    let mut stir_evaluations = Vec::new();
                    for (answer_a, answer_b) in answers_a.iter().zip(&answers_b) {
                        let a_trunc = round_state.folding_randomness
                            [..round_state.folding_randomness.len() - 1]
                            .to_vec();
                        let vars_b = answer_b.len().ilog2() as usize;
                        let eval_a = answer_a.evaluate(&MultilinearPoint(a_trunc));
                        let b_trunc = round_state.folding_randomness[..vars_b].to_vec();
                        let eval_b = answer_b.evaluate(&MultilinearPoint(b_trunc));
                        let last_fold_rand_a = round_state.folding_randomness
                            [round_state.folding_randomness.len() - 1];
                        let last_fold_rand_b = round_state.folding_randomness[vars_b..]
                            .iter()
                            .map(|&x| EF::ONE - x)
                            .product::<EF>();
                        stir_evaluations
                            .push(eval_a * last_fold_rand_a + eval_b * last_fold_rand_b);
                    }

                    stir_evaluations
                } else {
                    let mut answers = Vec::<Vec<F>>::with_capacity(stir_challenges_indexes.len());
                    let mut merkle_proofs = Vec::with_capacity(stir_challenges_indexes.len());
                    for challenge in &stir_challenges_indexes {
                        let commitment = extension_mmcs_f
                            .open_batch(*challenge, &round_state.commitment_merkle_prover_data_a);
                        answers.push(commitment.opened_values[0].clone());
                        merkle_proofs.push(commitment.opening_proof);

                        dbg!(1);

                        let mut folded_rev = round_state.folding_randomness.clone();
                        folded_rev.reverse();
                        let folded_eval =
                            commitment.opened_values[0].evaluate(&folded_rev);
                        assert_eq!(
                            folded_eval,
                            round_state.sumcheck_prover.evals.evaluate(
                                &MultilinearPoint::expand_from_univariate(
                                    EF::from(
                                        round_state.next_domain_gen.exp_u64(*challenge as u64)
                                    ),
                                    log2_ceil_usize(round_state.sumcheck_prover.evals.len())
                                )
                            )
                        );
                    }

                    // merkle leaves
                    for answer in &answers {
                        prover_state.hint_base_scalars(&flatten_scalars_to_base(answer));
                    }

                    // merkle authentication proof
                    for merkle_proof in &merkle_proofs {
                        for digest in merkle_proof {
                            prover_state.hint_base_scalars(digest);
                        }
                    }

                    // Evaluate answers in the folding randomness.
                    let mut stir_evaluations = Vec::with_capacity(answers.len());
                    for answer in &answers {
                        stir_evaluations.push(answer.evaluate(&round_state.folding_randomness));
                    }

                    stir_evaluations
                }
            }

            Some(data) => {
                let mut answers = Vec::with_capacity(stir_challenges_indexes.len());
                let mut merkle_proofs = Vec::with_capacity(stir_challenges_indexes.len());
                for challenge in &stir_challenges_indexes {
                    let commitment = extension_mmcs_ef.open_batch(*challenge, data);
                    answers.push(commitment.opened_values[0].clone());
                    merkle_proofs.push(commitment.opening_proof);

                    dbg!(2);

                    let folded_eval =
                        commitment.opened_values[0].evaluate(&round_state.folding_randomness);
                    assert_eq!(
                        folded_eval,
                        round_state.sumcheck_prover.evals.evaluate(
                            &MultilinearPoint::expand_from_univariate(
                                EF::from(round_state.next_domain_gen.exp_u64(*challenge as u64)),
                                log2_ceil_usize(round_state.sumcheck_prover.evals.len())
                            )
                        )
                    );
                }

                // merkle leaves
                for answer in &answers {
                    prover_state.hint_extension_scalars(answer);
                }

                // merkle authentication proof
                for merkle_proof in &merkle_proofs {
                    for digest in merkle_proof {
                        prover_state.hint_base_scalars(digest);
                    }
                }

                // Evaluate answers in the folding randomness.
                let mut stir_evaluations = Vec::with_capacity(answers.len());
                for answer in &answers {
                    stir_evaluations.push(answer.evaluate(&round_state.folding_randomness));
                }

                stir_evaluations
            }
        };

        // Randomness for combination
        let combination_randomness_gen: EF = prover_state.sample();
        let ood_combination_randomness: Vec<_> = combination_randomness_gen
            .powers()
            .collect_n(ood_challenges.len());
        round_state.sumcheck_prover.add_new_equality(
            &ood_challenges,
            &ood_answers,
            &ood_combination_randomness,
        );
        let stir_combination_randomness = combination_randomness_gen
            .powers()
            .skip(ood_challenges.len())
            .take(stir_challenges.len())
            .collect::<Vec<_>>();

        round_state.sumcheck_prover.add_new_base_equality(
            &stir_challenges,
            &stir_evaluations,
            &stir_combination_randomness,
        );

        let folding_randomness = round_state
            .sumcheck_prover
            .run_sumcheck_many_rounds::<PF<EF>>(
                prover_state,
                folding_factor_next,
                round_params.folding_pow_bits,
            );

        let start_idx = self.folding_factor.total_number(round_index);
        let dst_randomness =
            &mut round_state.randomness_vec[start_idx..][..folding_randomness.len()];

        for (dst, src) in dst_randomness.iter_mut().zip(folding_randomness.iter()) {
            *dst = *src;
        }

        // Update round state
        round_state.domain_size = new_domain_size;
        round_state.next_domain_gen =
            F::two_adic_generator(new_domain_size.ilog2() as usize - folding_factor_next);
        round_state.folding_randomness = folding_randomness;
        round_state.merkle_prover_data = Some(prover_data);

        Ok(())
    }

    fn final_round(
        &self,
        round_index: usize,
        prover_state: &mut ProverState<PF<EF>, EF, impl FSChallenger<EF>>,
        round_state: &mut RoundState<F, EF, DIGEST_ELEMS>,
    ) -> ProofResult<()>
    where
        H: CryptographicHasher<PF<EF>, [PF<EF>; DIGEST_ELEMS]>
            + CryptographicHasher<PFPacking<EF>, [PFPacking<EF>; DIGEST_ELEMS]>
            + Sync,
        C: PseudoCompressionFunction<[PF<EF>; DIGEST_ELEMS], 2>
            + PseudoCompressionFunction<[PFPacking<EF>; DIGEST_ELEMS], 2>
            + Sync,
        [PF<EF>; DIGEST_ELEMS]: Serialize + for<'de> Deserialize<'de>,
    {
        // Directly send coefficients of the polynomial to the verifier.
        prover_state.add_extension_scalars(&round_state.sumcheck_prover.evals);

        prover_state.pow_grinding(self.final_pow_bits);

        // Final verifier queries and answers. The indices are over the folded domain.
        let final_challenge_indexes = get_challenge_stir_queries(
            // The size of the original domain before folding
            round_state.domain_size >> self.folding_factor.at_round(round_index),
            self.final_queries,
            prover_state,
        );

        // Every query requires opening these many in the previous Merkle tree
        let mmcs = MerkleTreeMmcs::<PFPacking<EF>, PFPacking<EF>, H, C, DIGEST_ELEMS>::new(
            self.merkle_hash.clone(),
            self.merkle_compress.clone(),
        );
        let extension_mmcs = ExtensionMmcs::new(mmcs.clone());

        match &round_state.merkle_prover_data {
            None => {
                let mut answers = Vec::<Vec<PF<EF>>>::with_capacity(final_challenge_indexes.len());
                let mut merkle_proofs = Vec::with_capacity(final_challenge_indexes.len());

                for challenge in final_challenge_indexes {
                    let commitment =
                        mmcs.open_batch(challenge, &round_state.commitment_merkle_prover_data_a);
                    answers.push(commitment.opened_values[0].clone());
                    merkle_proofs.push(commitment.opening_proof);
                }

                // merkle leaves
                for answer in &answers {
                    prover_state.hint_base_scalars(answer);
                }

                // merkle authentication proof
                for merkle_proof in &merkle_proofs {
                    for digest in merkle_proof {
                        prover_state.hint_base_scalars(digest);
                    }
                }
            }

            Some(data) => {
                let mut answers = Vec::with_capacity(final_challenge_indexes.len());
                let mut merkle_proofs = Vec::with_capacity(final_challenge_indexes.len());
                for challenge in final_challenge_indexes {
                    let commitment = extension_mmcs.open_batch(challenge, data);
                    answers.push(commitment.opened_values[0].clone());
                    merkle_proofs.push(commitment.opening_proof);

                    let folded_eval =
                        commitment.opened_values[0].evaluate(&round_state.folding_randomness);
                    assert_eq!(
                        folded_eval,
                        round_state.sumcheck_prover.evals.evaluate(
                            &MultilinearPoint::expand_from_univariate(
                                EF::from(round_state.next_domain_gen.exp_u64(challenge as u64)),
                                log2_ceil_usize(round_state.sumcheck_prover.evals.len())
                            )
                        )
                    );
                }

                // merkle leaves
                for answer in &answers {
                    prover_state.hint_extension_scalars(answer);
                }

                // merkle authentication proof
                for merkle_proof in &merkle_proofs {
                    for digest in merkle_proof {
                        prover_state.hint_base_scalars(digest);
                    }
                }
            }
        }

        dbg!(prover_state.challenger().state());
        dbg!(&round_state.folding_randomness);

        // Run final sumcheck if required
        if self.final_sumcheck_rounds > 0 {
            let final_folding_randomness = round_state
                .sumcheck_prover
                .run_sumcheck_many_rounds::<PF<EF>>(
                    prover_state,
                    self.final_sumcheck_rounds,
                    self.final_folding_pow_bits,
                );
            let start_idx = self.folding_factor.total_number(round_index);
            let rand_dst = &mut round_state.randomness_vec
                [start_idx..start_idx + final_folding_randomness.len()];

            for (dst, src) in rand_dst.iter_mut().zip(final_folding_randomness.iter()) {
                *dst = *src;
            }
        }

        Ok(())
    }

    #[allow(clippy::type_complexity)]
    fn compute_stir_queries(
        &self,
        prover_state: &mut ProverState<PF<EF>, EF, impl FSChallenger<EF>>,
        round_state: &RoundState<F, EF, DIGEST_ELEMS>,
        num_variables: usize,
        round_params: &RoundConfig<F>,
        ood_points: &[EF],
        round_index: usize,
    ) -> ProofResult<(
        Vec<MultilinearPoint<EF>>,
        Vec<MultilinearPoint<F>>,
        Vec<usize>,
    )> {
        let stir_challenges_indexes = get_challenge_stir_queries(
            round_state.domain_size >> self.folding_factor.at_round(round_index),
            round_params.num_queries,
            prover_state,
        );

        // Compute the generator of the folded domain, in the extension field
        let domain_scaled_gen = round_state.next_domain_gen;
        let ood_challenges = ood_points
            .iter()
            .map(|univariate| MultilinearPoint::expand_from_univariate(*univariate, num_variables))
            .collect();
        let stir_challenges = stir_challenges_indexes
            .iter()
            .map(|i| {
                MultilinearPoint::expand_from_univariate(
                    domain_scaled_gen.exp_u64(*i as u64),
                    num_variables,
                )
            })
            .collect();

        Ok((ood_challenges, stir_challenges, stir_challenges_indexes))
    }
}

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

        let (sumcheck_prover, folding_randomness) = SumcheckSingle::run_initial_sumcheck_rounds(
            polynomial,
            &statement,
            combination_randomness_gen,
            prover_state,
            prover.folding_factor.at_round(0),
            prover.starting_folding_pow_bits,
        );

        let randomness_vec = {
            let mut randomness_vec = folding_randomness.0.clone();
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

        let (sumcheck_prover, folding_randomness) = SumcheckSingle::run_initial_sumcheck_rounds(
            &polynomial,
            &statement,
            combination_randomness_gen,
            prover_state,
            prover.folding_factor.at_round(0) + 1,
            prover.starting_folding_pow_bits,
        );

        let randomness_vec = {
            let mut randomness_vec = folding_randomness.0.clone();
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
