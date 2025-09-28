use multilinear_toolkit::prelude::*;
use p3_field::{ExtensionField, Field, TwoAdicField};
use p3_util::log2_strict_usize;
use tracing::{info_span, instrument};

use crate::{config::WhirConfig, *};

impl<EF> WhirConfig<EF>
where
    EF: ExtensionField<PF<EF>>,
    PF<EF>: TwoAdicField,
{
    fn validate_parameters(&self) -> bool {
        self.num_variables
            == self.folding_factor.total_number(self.n_rounds()) + self.final_sumcheck_rounds
    }

    fn validate_statement(&self, statement: &[Evaluation<EF>]) -> bool {
        statement
            .iter()
            .all(|e| e.num_variables() == self.num_variables)
    }

    fn validate_witness(&self, witness: &Witness<EF>, polynomial: &MleRef<'_, EF>) -> bool {
        assert_eq!(witness.ood_points.len(), witness.ood_answers.len());
        polynomial.n_vars() == self.num_variables
    }

    #[instrument(name = "WHIR prove", skip_all)]
    pub fn prove(
        &self,
        prover_state: &mut ProverState<PF<EF>, EF, impl FSChallenger<EF>>,
        statement: Vec<Evaluation<EF>>,
        witness: Witness<EF>,
        polynomial: &MleRef<'_, EF>,
    ) -> MultilinearPoint<EF> {
        assert!(
            self.validate_parameters()
                && self.validate_statement(&statement)
                && self.validate_witness(&witness, &polynomial),
            "Invalid prover parameters, statement, or witness"
        );

        let mut round_state = RoundState::initialize_first_round_state(
            self,
            prover_state,
            statement,
            witness,
            polynomial,
        )
        .unwrap();

        for round in 0..=self.n_rounds() {
            self.round(round, prover_state, &mut round_state).unwrap();
        }

        MultilinearPoint(round_state.randomness_vec)
    }

    #[instrument(name = "WHIR batch prove", skip_all)]
    pub fn batch_prove(
        &self,
        prover_state: &mut ProverState<PF<EF>, EF, impl FSChallenger<EF>>,
        statement_a: Vec<Evaluation<EF>>,
        witness_a: Witness<EF>,
        polynomial_a: &MleRef<'_, EF>,
        statement_b: Vec<Evaluation<EF>>,
        witness_b: Witness<EF>,
        polynomial_b: &MleRef<'_, EF>,
    ) -> MultilinearPoint<EF>
    where
        PF<EF>: TwoAdicField,
    {
        // (1 - X).PolB + X.PolA

        assert_eq!(polynomial_a.n_vars(), self.num_variables);
        assert!(polynomial_a.n_vars() >= polynomial_b.n_vars());
        let mut round_state = RoundState::initialize_first_round_state_batch(
            self,
            prover_state,
            statement_a,
            witness_a,
            polynomial_a,
            statement_b,
            witness_b,
            polynomial_b,
        );

        for round in 0..=self.n_rounds() {
            self.round(round, prover_state, &mut round_state).unwrap();
        }
        MultilinearPoint(round_state.randomness_vec)
    }

    fn round(
        &self,
        round_index: usize,
        prover_state: &mut ProverState<PF<EF>, EF, impl FSChallenger<EF>>,
        round_state: &mut RoundState<EF>,
    ) -> ProofResult<()> {
        let folded_evaluations = &round_state.sumcheck_prover.evals;
        let num_variables = self.num_variables - self.folding_factor.total_number(round_index);

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
        let inv_rate = new_domain_size >> num_variables;
        let folded_matrix = info_span!("FFT").in_scope(|| {
            reorder_and_dft(
                &folded_evaluations.by_ref(),
                folding_factor_next,
                log2_strict_usize(inv_rate),
            )
        });

        let (prover_data, root) = MerkleData::build(folded_matrix);

        prover_state.add_base_scalars(&root);

        // Handle OOD (Out-Of-Domain) samples
        let (ood_points, ood_answers) = sample_ood_points::<EF, _>(
            prover_state,
            round_params.ood_samples,
            num_variables,
            |point| info_span!("ood evaluation").in_scope(|| folded_evaluations.evaluate(point)),
        );

        prover_state.pow_grinding(round_params.pow_bits);

        let (ood_challenges, stir_challenges, stir_challenges_indexes) = self
            .compute_stir_queries(
                prover_state,
                round_state,
                num_variables,
                round_params,
                &ood_points,
                round_index,
            )?;

        let folding_randomness = round_state.folding_randomness(
            self.folding_factor.at_round(round_index)
                + round_state.commitment_merkle_prover_data_b.is_some() as usize,
        );

        let stir_evaluations = if round_state.commitment_merkle_prover_data_b.is_some() {
            let answers_a = open_merkle_tree_at_challenges(
                &round_state.merkle_prover_data,
                prover_state,
                &stir_challenges_indexes,
            );
            let answers_b = open_merkle_tree_at_challenges(
                round_state
                    .commitment_merkle_prover_data_b
                    .as_ref()
                    .unwrap(),
                prover_state,
                &stir_challenges_indexes,
            );
            let mut stir_evaluations = Vec::new();
            for (answer_a, answer_b) in answers_a.iter().zip(&answers_b) {
                let vars_a = answer_a.by_ref().n_vars();
                let vars_b = answer_b.by_ref().n_vars();
                let a_trunc = folding_randomness[1..].to_vec();
                let eval_a = answer_a.evaluate(&MultilinearPoint(a_trunc));
                let b_trunc = folding_randomness[vars_a - vars_b + 1..].to_vec();
                let eval_b = answer_b.evaluate(&MultilinearPoint(b_trunc));
                let last_fold_rand_a = folding_randomness[0];
                let last_fold_rand_b = folding_randomness[..vars_a - vars_b + 1]
                    .iter()
                    .map(|&x| EF::ONE - x)
                    .product::<EF>();
                stir_evaluations.push(eval_a * last_fold_rand_a + eval_b * last_fold_rand_b);
            }

            stir_evaluations
        } else {
            open_merkle_tree_at_challenges(
                &round_state.merkle_prover_data,
                prover_state,
                &stir_challenges_indexes,
            )
            .iter()
            .map(|answer| answer.evaluate(&folding_randomness))
            .collect()
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

        let next_folding_randomness = round_state.sumcheck_prover.run_sumcheck_many_rounds(
            prover_state,
            folding_factor_next,
            round_params.folding_pow_bits,
        );

        round_state
            .randomness_vec
            .extend_from_slice(&next_folding_randomness.0);

        // Update round state
        round_state.domain_size = new_domain_size;
        round_state.next_domain_gen =
            PF::<EF>::two_adic_generator(log2_strict_usize(new_domain_size) - folding_factor_next);
        round_state.merkle_prover_data = prover_data;
        round_state.commitment_merkle_prover_data_b = None;

        Ok(())
    }

    fn final_round(
        &self,
        round_index: usize,
        prover_state: &mut ProverState<PF<EF>, EF, impl FSChallenger<EF>>,
        round_state: &mut RoundState<EF>,
    ) -> ProofResult<()> {
        // Directly send coefficients of the polynomial to the verifier.
        prover_state.add_extension_scalars(&unpack_extension(
            &round_state
                .sumcheck_prover
                .evals
                .as_extension_packed()
                .unwrap(),
        ));

        prover_state.pow_grinding(self.final_pow_bits);

        // Final verifier queries and answers. The indices are over the folded domain.
        let final_challenge_indexes = get_challenge_stir_queries(
            // The size of the original domain before folding
            round_state.domain_size >> self.folding_factor.at_round(round_index),
            self.final_queries,
            prover_state,
        );

        {
            let mut answers = Vec::<MleOwned<EF>>::with_capacity(final_challenge_indexes.len());
            let mut merkle_proofs = Vec::with_capacity(final_challenge_indexes.len());

            for challenge in final_challenge_indexes {
                let (answer, proof) = round_state.merkle_prover_data.open(challenge);
                answers.push(answer);
                merkle_proofs.push(proof);
            }

            // merkle leaves
            for answer in &answers {
                match answer {
                    MleOwned::Base(answer) => {
                        prover_state.hint_base_scalars(answer);
                    }
                    MleOwned::Extension(answer) => {
                        prover_state.hint_extension_scalars(answer);
                    }
                    _ => unreachable!(),
                }
            }

            // merkle authentication proof
            for merkle_proof in &merkle_proofs {
                for digest in merkle_proof {
                    prover_state.hint_base_scalars(digest);
                }
            }
        }

        // Run final sumcheck if required
        if self.final_sumcheck_rounds > 0 {
            let final_folding_randomness = round_state.sumcheck_prover.run_sumcheck_many_rounds(
                prover_state,
                self.final_sumcheck_rounds,
                self.final_folding_pow_bits,
            );

            round_state
                .randomness_vec
                .extend(final_folding_randomness.0);
        }

        Ok(())
    }

    #[allow(clippy::type_complexity)]
    fn compute_stir_queries(
        &self,
        prover_state: &mut ProverState<PF<EF>, EF, impl FSChallenger<EF>>,
        round_state: &RoundState<EF>,
        num_variables: usize,
        round_params: &RoundConfig<EF>,
        ood_points: &[EF],
        round_index: usize,
    ) -> ProofResult<(
        Vec<MultilinearPoint<EF>>,
        Vec<MultilinearPoint<PF<EF>>>,
        Vec<usize>,
    )> {
        let stir_challenges_indexes = get_challenge_stir_queries(
            round_state.domain_size >> self.folding_factor.at_round(round_index),
            round_params.num_queries,
            prover_state,
        );

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

fn open_merkle_tree_at_challenges<EF: ExtensionField<PF<EF>>>(
    merkle_tree: &MerkleData<EF>,
    prover_state: &mut ProverState<PF<EF>, EF, impl FSChallenger<EF>>,
    stir_challenges_indexes: &[usize],
) -> Vec<MleOwned<EF>> {
    let mut merkle_proofs = Vec::new();
    let mut answers = Vec::new();
    for challenge in stir_challenges_indexes {
        let (answer, proof) = merkle_tree.open(*challenge);
        answers.push(answer);
        merkle_proofs.push(proof);
    }

    // merkle leaves
    for answer in &answers {
        match answer {
            MleOwned::Base(answer) => {
                prover_state.hint_base_scalars(answer);
            }
            MleOwned::Extension(answer) => {
                prover_state.hint_extension_scalars(answer);
            }
            _ => unreachable!(),
        }
    }

    // merkle authentication proof
    for merkle_proof in &merkle_proofs {
        for digest in merkle_proof {
            prover_state.hint_base_scalars(digest);
        }
    }

    answers
}

#[derive(Debug, Clone)]
pub struct SumcheckSingle<EF: ExtensionField<PF<EF>>> {
    /// Evaluations of the polynomial `p(X)`.
    pub(crate) evals: MleOwned<EF>,
    /// Evaluations of the equality polynomial used for enforcing constraints.
    pub(crate) weights: MleOwned<EF>,
    /// Accumulated sum incorporating equality constraints.
    pub(crate) sum: EF,
}

impl<EF: Field> SumcheckSingle<EF>
where
    EF: ExtensionField<PF<EF>>,
{
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
                compute_eval_eq_packed::<_, true>(
                    point,
                    &mut self.weights.as_extension_packed_mut().unwrap(),
                    rand,
                );
            });

        self.sum += combination_randomness
            .iter()
            .zip(evaluations.iter())
            .map(|(&rand, &eval)| rand * eval)
            .sum::<EF>();
    }

    pub(crate) fn add_new_base_equality(
        &mut self,
        points: &[MultilinearPoint<PF<EF>>],
        evaluations: &[EF],
        combination_randomness: &[EF],
    ) {
        assert_eq!(combination_randomness.len(), points.len());
        assert_eq!(evaluations.len(), points.len());

        // Parallel update of weight buffer

        points
            .iter()
            .zip(combination_randomness.iter())
            .for_each(|(point, &rand)| {
                compute_eval_eq_base_packed::<_, _, true>(
                    point,
                    &mut self.weights.as_extension_packed_mut().unwrap(),
                    rand,
                );
            });

        // Accumulate the weighted sum (cheap, done sequentially)
        self.sum += combination_randomness
            .iter()
            .zip(evaluations.iter())
            .map(|(&rand, &eval)| rand * eval)
            .sum::<EF>();
    }

    fn run_sumcheck_many_rounds(
        &mut self,
        prover_state: &mut ProverState<PF<EF>, EF, impl FSChallenger<EF>>,
        folding_factor: usize,
        _pow_bits: usize, // TODO pow grinding
    ) -> MultilinearPoint<EF> {
        let num_vars_start = self.evals.by_ref().n_vars();
        let (challenges, folds, new_sum) = sumcheck_prove_many_rounds(
            1,
            MleGroupOwned::merge(vec![
                std::mem::take(&mut self.evals),
                std::mem::take(&mut self.weights),
            ]),
            &ProductComputation,
            &ProductComputation,
            &[],
            None,
            false,
            prover_state,
            self.sum,
            None,
            folding_factor,
        );

        self.sum = new_sum;
        [self.evals, self.weights] = folds.as_owned().unwrap().split().try_into().unwrap();

        assert_eq!(
            self.evals.by_ref().n_vars(),
            num_vars_start - folding_factor
        );

        challenges
    }

    pub(crate) fn run_initial_sumcheck_rounds(
        evals: &MleRef<'_, EF>,
        statement: &[Evaluation<EF>],
        combination_randomness: EF,
        prover_state: &mut ProverState<PF<EF>, EF, impl FSChallenger<EF>>,
        folding_factor: usize,
        _pow_bits: usize, // TODO
    ) -> (Self, MultilinearPoint<EF>) {
        assert_ne!(folding_factor, 0);
        let mut res = Vec::with_capacity(folding_factor);

        let (mut weights, mut sum) =
            combine_statement::<EF>(statement, combination_randomness, EF::ONE);

        let sumcheck_poly = match evals {
            MleRef::Base(evals) => compute_product_sumcheck_polynomial(
                PFPacking::<EF>::pack_slice(&evals),
                &weights,
                sum,
                |e| EFPacking::<EF>::to_ext_iter([e]).collect(),
            ),
            MleRef::ExtensionPacked(evals) => {
                compute_product_sumcheck_polynomial(evals, &weights, sum, |e| {
                    EFPacking::<EF>::to_ext_iter([e]).collect()
                })
            }
            _ => unimplemented!(),
        };
        prover_state.add_extension_scalars(&sumcheck_poly.coeffs);

        // TODO: re-enable PoW grinding
        // prover_state.pow_grinding(pow_bits);

        let r: EF = prover_state.sample();

        let compressed_evals: Vec<EFPacking<EF>> =
            info_span!("initial compression").in_scope(|| {
                fold_multilinear_in_place(&mut weights, &[(EF::ONE - r), r]);
                match evals {
                    MleRef::Base(evals) => {
                        let folded = fold_multilinear(&evals, &[(EF::ONE - r), r], &|a, b| b * a);
                        pack_extension(&folded) // TODO avoid the intermediate allocation "folded"
                    }
                    MleRef::ExtensionPacked(evals) => {
                        fold_multilinear(&evals, &[(EF::ONE - r), r], &|a, b| a * b)
                    }
                    _ => unimplemented!(),
                }
            });

        sum = sumcheck_poly.evaluate(r);

        res.push(r);

        let mut sumcheck = Self {
            evals: MleOwned::ExtensionPacked(compressed_evals),
            weights: MleOwned::ExtensionPacked(weights),
            sum,
        };

        // Apply rest of sumcheck rounds
        let remaining_challenges = info_span!("remaining initial sumcheck rounds").in_scope(|| {
            sumcheck.run_sumcheck_many_rounds(prover_state, folding_factor - 1, _pow_bits)
        });
        res.extend(remaining_challenges.0);

        (sumcheck, MultilinearPoint(res))
    }

    pub(crate) fn run_initial_sumcheck_rounds_batched(
        pol_a: &MleRef<'_, EF>,
        pol_b: &MleRef<'_, EF>,
        statement_a: Vec<Evaluation<EF>>,
        statement_b: Vec<Evaluation<EF>>,
        combination_randomness: EF,
        prover_state: &mut ProverState<PF<EF>, EF, impl FSChallenger<EF>>,
        folding_factor: usize,
        _pow_bits: usize, // TODO
    ) -> (Self, MultilinearPoint<EF>) {
        let mut res = Vec::with_capacity(folding_factor);

        let (weights_a, linear_a_f_1) =
            combine_statement(&statement_a, combination_randomness, EF::ONE);
        let (weights_b, linear_b_f_0) = combine_statement(
            &statement_b,
            combination_randomness,
            combination_randomness.exp_u64(statement_a.len() as u64),
        );

        let pol_a = match pol_a {
            MleRef::Base(evals) => evals,
            _ => unimplemented!(),
        };
        let pol_b = match pol_b {
            MleRef::ExtensionPacked(evals) => evals,
            _ => unimplemented!(),
        };

        let linear_b_f_1: EF = dot_product_ef_packed_par(&weights_a[..pol_b.len()], &pol_b);
        let linear_a_f_0: EF = dot_product_ef_packed_par(
            &weights_b,
            &PFPacking::<EF>::pack_slice(pol_a)[..pol_b.len()],
        );

        let one_minus_x = DensePolynomial::new(vec![EF::ONE, -EF::ONE]);
        let x = DensePolynomial::new(vec![EF::ZERO, EF::ONE]);
        let sumcheck_poly_a = DensePolynomial::new(vec![linear_a_f_0, linear_a_f_1 - linear_a_f_0]);
        let sumcheck_poly_b = DensePolynomial::new(vec![linear_b_f_0, linear_b_f_1 - linear_b_f_0]);
        let sumcheck_poly = &(&one_minus_x * &sumcheck_poly_b) + &(&x * &sumcheck_poly_a);

        prover_state.add_extension_scalars(&sumcheck_poly.coeffs);

        // TODO: re-enable PoW grinding
        // prover_state.pow_grinding(pow_bits);

        let r: EF = prover_state.sample();

        let (compressed_evals, compressed_weights) =
            info_span!("initial compression").in_scope(|| {
                let pol_a_packed = PFPacking::<EF>::pack_slice(pol_a);
                let r_packed = EFPacking::<EF>::from(r);
                let compressed_evals: Vec<EFPacking<EF>> = pol_a_packed
                    .par_iter()
                    .zip(pol_b.par_iter())
                    .map(|(&a, &b)| (-b + a) * r + b)
                    .chain(
                        pol_a_packed[pol_b.len()..]
                            .par_iter()
                            .map(|&a| r_packed * a),
                    )
                    .collect();

                let compressed_weights: Vec<EFPacking<EF>> = weights_a
                    .par_iter()
                    .zip(weights_b.par_iter())
                    .map(|(&a, &b)| (-b + a) * r + b)
                    .chain(weights_a[pol_b.len()..].par_iter().map(|&a| r_packed * a))
                    .collect();

                (compressed_evals, compressed_weights)
            });

        let sum = sumcheck_poly.evaluate(r);

        res.push(r);

        let mut sumcheck = Self {
            evals: MleOwned::ExtensionPacked(compressed_evals),
            weights: MleOwned::ExtensionPacked(compressed_weights),
            sum,
        };

        // Apply rest of sumcheck rounds
        let remaining_challenges = info_span!("remaining initial sumcheck rounds").in_scope(|| {
            sumcheck.run_sumcheck_many_rounds(prover_state, folding_factor - 1, _pow_bits)
        });
        res.extend(remaining_challenges.0);

        (sumcheck, MultilinearPoint(res))
    }
}

#[derive(Debug)]
pub(crate) struct RoundState<EF>
where
    EF: ExtensionField<PF<EF>>,
{
    domain_size: usize,
    next_domain_gen: PF<EF>,
    sumcheck_prover: SumcheckSingle<EF>,
    commitment_merkle_prover_data_b: Option<MerkleData<EF>>,
    merkle_prover_data: MerkleData<EF>,
    randomness_vec: Vec<EF>,
}

#[allow(clippy::mismatching_type_param_order)]
impl<EF> RoundState<EF>
where
    EF: ExtensionField<PF<EF>>,
    PF<EF>: TwoAdicField,
{
    pub(crate) fn initialize_first_round_state(
        prover: &WhirConfig<EF>,
        prover_state: &mut ProverState<PF<EF>, EF, impl FSChallenger<EF>>,
        mut statement: Vec<Evaluation<EF>>,
        witness: Witness<EF>,
        polynomial: &MleRef<'_, EF>,
    ) -> ProofResult<Self> {
        let ood_statements = witness
            .ood_points
            .into_iter()
            .zip(witness.ood_answers)
            .map(|(point, evaluation)| {
                Evaluation::new(
                    MultilinearPoint::expand_from_univariate(point, prover.num_variables),
                    evaluation,
                )
            })
            .collect::<Vec<_>>();

        statement.splice(0..0, ood_statements);

        let combination_randomness_gen: EF = prover_state.sample();

        let (sumcheck_prover, folding_randomness) = SumcheckSingle::run_initial_sumcheck_rounds(
            polynomial,
            &statement,
            combination_randomness_gen,
            prover_state,
            prover.folding_factor.at_round(0),
            prover.starting_folding_pow_bits,
        );

        Ok(Self {
            domain_size: prover.starting_domain_size(),
            next_domain_gen: PF::<EF>::two_adic_generator(
                log2_strict_usize(prover.starting_domain_size())
                    - prover.folding_factor.at_round(0),
            ),
            sumcheck_prover,
            merkle_prover_data: witness.prover_data,
            commitment_merkle_prover_data_b: None,
            randomness_vec: folding_randomness.0.clone(),
        })
    }

    pub(crate) fn initialize_first_round_state_batch(
        prover: &WhirConfig<EF>,
        prover_state: &mut ProverState<PF<EF>, EF, impl FSChallenger<EF>>,
        mut statement_a: Vec<Evaluation<EF>>,
        witness_a: Witness<EF>,
        polynomial_a: &MleRef<'_, EF>,
        mut statement_b: Vec<Evaluation<EF>>,
        witness_b: Witness<EF>,
        polynomial_b: &MleRef<'_, EF>,
    ) -> Self {
        let ood_statements_a = witness_a
            .ood_points
            .into_iter()
            .zip(witness_a.ood_answers)
            .map(|(point, evaluation)| {
                Evaluation::new(
                    MultilinearPoint::expand_from_univariate(point, polynomial_a.n_vars()),
                    evaluation,
                )
            })
            .collect::<Vec<_>>();
        statement_a.splice(0..0, ood_statements_a);

        let ood_statements_b = witness_b
            .ood_points
            .into_iter()
            .zip(witness_b.ood_answers)
            .map(|(point, evaluation)| {
                Evaluation::new(
                    MultilinearPoint::expand_from_univariate(point, polynomial_b.n_vars()),
                    evaluation,
                )
            })
            .collect::<Vec<_>>();
        statement_b.splice(0..0, ood_statements_b);

        let combination_randomness_gen: EF = prover_state.sample();

        let (sumcheck_prover, folding_randomness) =
            SumcheckSingle::run_initial_sumcheck_rounds_batched(
                polynomial_a,
                polynomial_b,
                statement_a,
                statement_b,
                combination_randomness_gen,
                prover_state,
                prover.folding_factor.at_round(0) + 1,
                prover.starting_folding_pow_bits,
            );

        Self {
            domain_size: prover.starting_domain_size(),
            next_domain_gen: PF::<EF>::two_adic_generator(
                log2_strict_usize(prover.starting_domain_size())
                    - prover.folding_factor.at_round(0),
            ),
            sumcheck_prover,
            merkle_prover_data: witness_a.prover_data,
            commitment_merkle_prover_data_b: Some(witness_b.prover_data),
            randomness_vec: folding_randomness.0.clone(),
        }
    }

    fn folding_randomness(&self, folding_factor: usize) -> MultilinearPoint<EF> {
        MultilinearPoint(self.randomness_vec[self.randomness_vec.len() - folding_factor..].to_vec())
    }
}

#[instrument(skip_all, fields(num_constraints = statement.len(), n_vars = statement[0].num_variables()))]
fn combine_statement<EF>(
    statement: &[Evaluation<EF>],
    challenge: EF,
    start: EF,
) -> (Vec<EFPacking<EF>>, EF)
where
    EF: ExtensionField<PF<EF>>,
{
    let num_variables = statement[0].num_variables();
    assert!(statement.iter().all(|e| e.num_variables() == num_variables));

    let mut combined_evals =
        EFPacking::<EF>::zero_vec(1 << (num_variables - packing_log_width::<EF>()));
    let (combined_sum, _) =
        statement
            .iter()
            .fold((EF::ZERO, start), |(mut acc_sum, gamma_pow), constraint| {
                compute_sparse_eval_eq_packed::<EF>(
                    &constraint.point,
                    &mut combined_evals,
                    gamma_pow,
                );
                acc_sum += constraint.value * gamma_pow;
                (acc_sum, gamma_pow * challenge)
            });

    (combined_evals, combined_sum)
}

#[instrument(skip_all, fields(num_constraints = statement.len(), n_vars = statement[0].num_variables()))]
fn combine_statement_batched<EF>(
    statement: &[Evaluation<EF>],
    challenge: EF,
) -> (Vec<EFPacking<EF>>, EF, EF)
where
    EF: ExtensionField<PF<EF>>,
{
    let num_variables = statement[0].num_variables();
    assert!(statement.iter().all(|e| e.num_variables() == num_variables));

    let mut combined_evals =
        EFPacking::<EF>::zero_vec(1 << (num_variables - packing_log_width::<EF>()));
    let (combined_sum_a, combined_sum_b, _) = statement.iter().fold(
        (EF::ZERO, EF::ZERO, EF::ONE),
        |(mut acc_sum_a, mut acc_sum_b, gamma_pow), constraint| {
            compute_sparse_eval_eq_packed::<EF>(&constraint.point, &mut combined_evals, gamma_pow);
            if constraint.point[0] == EF::ZERO {
                acc_sum_b += constraint.value * gamma_pow;
            } else {
                assert_eq!(constraint.point[0], EF::ONE);
                acc_sum_a += constraint.value * gamma_pow;
            }
            (acc_sum_a, acc_sum_b, gamma_pow * challenge)
        },
    );

    (combined_evals, combined_sum_a, combined_sum_b)
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
