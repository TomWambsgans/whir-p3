use std::time::{Duration, Instant};

use clap::Parser;
use p3_baby_bear::BabyBear;
use p3_challenger::DuplexChallenger;
use p3_dft::Radix2DFTSmallBatch;
use p3_field::PrimeCharacteristicRing;
use p3_field::extension::BinomialExtensionField;
use p3_goldilocks::Goldilocks;
use p3_koala_bear::{KoalaBear, Poseidon2KoalaBear};
use p3_symmetric::{PaddingFreeSponge, TruncatedPermutation};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng, rngs::SmallRng};
use whir_p3::{
    fiat_shamir::domain_separator::DomainSeparator,
    parameters::{DEFAULT_MAX_POW, FoldingFactor, ProtocolParameters, errors::SecurityAssumption},
    poly::{evals::EvaluationsList, multilinear::MultilinearPoint},
    whir::{
        committer::{reader::CommitmentReader, writer::CommitmentWriter},
        constraints::statement::EqStatement,
        parameters::{InitialPhaseConfig, WhirConfig},
        proof::WhirProof,
        prover::Prover,
        verifier::Verifier,
    },
};

type F = KoalaBear;
type EF = BinomialExtensionField<F, 4>;
type _F = BabyBear;
type _EF = BinomialExtensionField<_F, 5>;
type __F = Goldilocks;
type __EF = BinomialExtensionField<__F, 2>;

type Poseidon16 = Poseidon2KoalaBear<16>;
type Poseidon24 = Poseidon2KoalaBear<24>;

type MerkleHash = PaddingFreeSponge<Poseidon24, 24, 16, 8>; // leaf hashing
type MerkleCompress = TruncatedPermutation<Poseidon16, 2, 8, 16>; // 2-to-1 compression
type MyChallenger = DuplexChallenger<F, Poseidon16, 16, 8>;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    #[arg(short = 'l', long, default_value = "90")]
    security_level: usize,

    #[arg(short = 'p', long)]
    pow_bits: Option<usize>,

    #[arg(short = 'e', long = "evaluations", default_value = "1")]
    num_evaluations: usize,

    #[arg(short = 'r', long, default_value = "1")]
    rate: usize,

    #[arg(short = 'k', long = "fold", default_value = "5")]
    folding_factor: usize,

    #[arg(long = "sec", default_value = "CapacityBound")]
    soundness_type: SecurityAssumption,

    #[arg(long = "initial-rs-reduction", default_value = "3")]
    rs_domain_initial_reduction_factor: usize,
}

#[allow(clippy::too_many_lines)]
fn main() {
    let mut args = Args::parse();

    if args.pow_bits.is_none() {
        args.pow_bits = Some(DEFAULT_MAX_POW);
    }

    let security_level = args.security_level;
    let pow_bits = args.pow_bits.unwrap();
    let starting_rate = args.rate;
    let folding_factor = FoldingFactor::Constant(args.folding_factor);
    let soundness_type = args.soundness_type;
    let rs_domain_initial_reduction_factor = args.rs_domain_initial_reduction_factor;
    let n_reps: usize = 20;

    // Create hash and compression functions for the Merkle tree
    let mut rng = SmallRng::seed_from_u64(1);
    let poseidon16 = Poseidon16::new_from_rng_128(&mut rng);
    let poseidon24 = Poseidon24::new_from_rng_128(&mut rng);

    let merkle_hash = MerkleHash::new(poseidon24);
    let merkle_compress = MerkleCompress::new(poseidon16.clone());

    for num_variables in 13..26 {
        let num_coeffs = 1 << num_variables;

        // Construct WHIR protocol parameters
        let whir_params = ProtocolParameters {
            initial_phase_config: InitialPhaseConfig::WithStatementClassic,
            security_level,
            pow_bits,
            folding_factor: folding_factor.clone(),
            merkle_hash: merkle_hash.clone(),
            merkle_compress: merkle_compress.clone(),
            soundness_type,
            starting_log_inv_rate: starting_rate,
            rs_domain_initial_reduction_factor,
        };

        let params = WhirConfig::<EF, F, MerkleHash, MerkleCompress, MyChallenger>::new(
            num_variables,
            whir_params.clone(),
        );

        let mut rng = StdRng::seed_from_u64(0);
        let polynomial = EvaluationsList::<F>::new((0..num_coeffs).map(|_| rng.random()).collect());

        let mut points: Vec<_> = vec![];

        // Add all-ones and all-zeros points
        points.push(MultilinearPoint(vec![EF::ONE; num_variables]));

        // Construct statement with equality constraints
        let mut statement = EqStatement::<EF>::initialize(num_variables);

        for point in &points {
            statement.add_unevaluated_constraint_hypercube(point.clone(), &polynomial);
        }

        // Define the Fiat-Shamir domain separator pattern
        let mut domainsep = DomainSeparator::new(vec![]);
        domainsep.commit_statement::<_, _, _, 32>(&params);
        domainsep.add_whir_proof::<_, _, _, 32>(&params);

        let challenger = MyChallenger::new(poseidon16.clone());
        let dft = Radix2DFTSmallBatch::<F>::new(1 << params.max_fft_size());

        let mut commit_time: Duration = Default::default();
        let mut opening_time: Duration = Default::default();

        for _ in 0..n_reps {
            let mut prover_challenger = challenger.clone();
            let mut prover_state = domainsep.to_prover_state(challenger.clone());
            domainsep.observe_domain_separator(&mut prover_challenger);

            let committer = CommitmentWriter::new(&params);
            let mut proof =
                WhirProof::<F, EF, 8>::from_protocol_parameters(&whir_params, num_variables);

            let time = Instant::now();
            let witness = committer
                .commit(
                    &dft,
                    &mut prover_state,
                    &mut proof,
                    &mut prover_challenger,
                    polynomial.clone(),
                )
                .unwrap();
            commit_time += time.elapsed();

            let prover = Prover(&params);

            let time = Instant::now();
            prover
                .prove(
                    &dft,
                    &mut prover_state,
                    &mut proof,
                    &mut prover_challenger,
                    statement.clone(),
                    witness,
                )
                .unwrap();
            opening_time += time.elapsed();

            // Verify to ensure correctness
            let commitment_reader = CommitmentReader::new(&params);
            let verifier = Verifier::new(&params);

            let mut verifier_challenger = challenger.clone();
            let mut verifier_state =
                domainsep.to_verifier_state(prover_state.proof_data().to_vec(), challenger.clone());

            let parsed_commitment = commitment_reader
                .parse_commitment::<8>(&mut verifier_state, &proof, &mut verifier_challenger)
                .unwrap();

            verifier
                .verify(
                    &mut verifier_state,
                    &parsed_commitment,
                    statement.clone(),
                    &proof,
                    &mut verifier_challenger,
                )
                .unwrap();
        }

        commit_time /= n_reps as u32;
        opening_time /= n_reps as u32;

        let total_time = commit_time + opening_time;

        let total_time_per_field_element = total_time.as_secs_f64() / (1 << num_variables) as f64;

        println!(
            "num WHIR variables = {}, time per field element (commit + open): {:.3} µs",
            num_variables,
            total_time_per_field_element * 1e6
        );
    }

    println!("(num repetitions per setting: {})", n_reps);
}
