use std::time::Instant;

use p3_challenger::DuplexChallenger;
use p3_field::{PrimeCharacteristicRing, PrimeField64, extension::BinomialExtensionField};
use p3_koala_bear::{KoalaBear, Poseidon2KoalaBear};
use p3_symmetric::{PaddingFreeSponge, TruncatedPermutation};
use rand::{Rng, SeedableRng, rngs::StdRng};
use tracing_forest::{ForestLayer, util::LevelFilter};
use tracing_subscriber::{EnvFilter, Registry, layer::SubscriberExt, util::SubscriberInitExt};
use whir_p3::{
    dft::EvalsDft,
    fiat_shamir::{prover::ProverState, verifier::VerifierState},
    parameters::{
        DEFAULT_MAX_POW, FoldingFactor, MultivariateParameters, ProtocolParameters,
        errors::SecurityAssumption,
    },
    poly::{evals::EvaluationsList, multilinear::MultilinearPoint},
    whir::{
        committer::{reader::CommitmentReader, writer::CommitmentWriter},
        parameters::WhirConfig,
        prover::Prover,
        statement::Statement,
        verifier::Verifier,
    },
};

type F = KoalaBear;
type EF = BinomialExtensionField<KoalaBear, 8>;

type FPrimeSubfield = <F as PrimeCharacteristicRing>::PrimeSubfield;

type Poseidon16 = Poseidon2KoalaBear<16>;
type Poseidon24 = Poseidon2KoalaBear<24>;

type MerkleHash = PaddingFreeSponge<Poseidon24, 24, 16, 8>; // leaf hashing
type MerkleCompress = TruncatedPermutation<Poseidon16, 2, 8, 16>; // 2-to-1 compression
type MyChallenger = DuplexChallenger<FPrimeSubfield, Poseidon16, 16, 8>;

fn main() {
    let env_filter = EnvFilter::builder()
        .with_default_directive(LevelFilter::INFO.into())
        .from_env_lossy();

    Registry::default()
        .with(env_filter)
        .with(ForestLayer::default())
        .init();

    // Create hash and compression functions for the Merkle tree
    let poseidon16 = Poseidon16::new_from_rng_128(&mut StdRng::seed_from_u64(0));
    let poseidon24 = Poseidon24::new_from_rng_128(&mut StdRng::seed_from_u64(0));

    let merkle_hash = MerkleHash::new(poseidon24);
    let merkle_compress = MerkleCompress::new(poseidon16.clone());

    // let vars_diff = 1;

    let num_variables_a = 25;
    // let num_variables_b = num_variables_a - vars_diff;

    let num_coeffs_a = 1 << num_variables_a;
    // let num_coeffs_b = 1 << num_variables_b;

    let mv_params_a = MultivariateParameters::<EF>::new(num_variables_a);
    // let mv_params_b = MultivariateParameters::<EF>::new(num_variables_b);

    // Construct WHIR protocol parameters
    let whir_params_a = ProtocolParameters {
        security_level: 128,
        max_num_variables_to_send_coeffs: 6,
        pow_bits: DEFAULT_MAX_POW,
        folding_factor: FoldingFactor::ConstantFromSecondRound(7, 4),
        merkle_hash,
        merkle_compress,
        soundness_type: SecurityAssumption::CapacityBound,
        starting_log_inv_rate: 1,
        rs_domain_initial_reduction_factor: 5,
    };

    // let mut whir_params_b = whir_params_a.clone();
    // whir_params_b.folding_factor = FoldingFactor::Constant(4 - vars_diff);

    let params_a = WhirConfig::<EF, F, MerkleHash, MerkleCompress, MyChallenger>::new(
        mv_params_a,
        whir_params_a,
    );
    // let params_b = WhirConfig::<EF, F, MerkleHash, MerkleCompress, MyChallenger>::new(
    //     mv_params_b,
    //     whir_params_b,
    // );

    // println!("Using parameters:\n{}", params.to_string());

    let mut rng = StdRng::seed_from_u64(0);
    let polynomial_a = EvaluationsList::<F>::new((0..num_coeffs_a).map(|_| rng.random()).collect());
    // let polynomial_b = EvaluationsList::<F>::new((0..num_coeffs_b).map(|_| rng.random()).collect());

    // Sample `num_points` random multilinear points in the Boolean hypercube
    let points_a = (0..3)
        .map(|_| MultilinearPoint::rand(&mut rng, num_variables_a))
        .collect::<Vec<_>>();
    // let points_b = (0..2)
    //     .map(|_| MultilinearPoint::rand(&mut rng, num_variables_b))
    //     .collect::<Vec<_>>();

    // Construct a new statement with the correct number of variables
    let mut statement_a = Statement::<EF>::new(num_variables_a);
    // let mut statement_b = Statement::<EF>::new(num_variables_b);

    // Add constraints for each sampled point (equality constraints)
    for point_a in &points_a {
        let eval = polynomial_a.evaluate(point_a);
        statement_a.add_constraint(point_a.clone(), eval);
    }
    // for point_b in &points_b {
    //     let eval = polynomial_b.evaluate(point_b);
    //     statement_b.add_constraint(point_b.clone(), eval);
    // }

    // Define the Fiat-Shamir domain separator pattern for committing and proving

    let challenger = MyChallenger::new(poseidon16);

    // Initialize the Merlin transcript from the IOPattern
    let mut prover_state = ProverState::new(challenger.clone());

    // Commit to the polynomial and produce a witness

    let dft = EvalsDft::<FPrimeSubfield>::new(1 << params_a.max_fft_size());

    let time = Instant::now();
    let witness_a = CommitmentWriter::new(&params_a)
        .commit(&dft, &mut prover_state, polynomial_a)
        .unwrap();
    let commit_time_a = time.elapsed();

    // let time = Instant::now();
    // let witness_b = CommitmentWriter::new(&params_b)
    //     .commit(&dft, &mut prover_state, polynomial_b)
    //     .unwrap();
    // let commit_time_b = time.elapsed();

    // Generate a proof for the given statement and witness
    let time = Instant::now();
    Prover(&params_a)
        .prove(
            &dft,
            &mut prover_state,
            statement_a.clone(),
            witness_a,
            // statement_b.clone(),
            // witness_b,
        )
        .unwrap();
    let opening_time = time.elapsed();

    // Reconstruct verifier's view of the transcript using the DomainSeparator and prover's data
    let mut verifier_state = VerifierState::new(prover_state.proof_data().to_vec(), challenger);

    // Parse the commitment
    let parsed_commitment_a = CommitmentReader::new(&params_a)
        .parse_commitment::<8>(&mut verifier_state)
        .unwrap();
    // let parsed_commitment_b = CommitmentReader::new(&params_b)
    //     .parse_commitment::<8>(&mut verifier_state)
    //     .unwrap();

    let verif_time = Instant::now();
    Verifier::new(&params_a)
        .verify(
            &mut verifier_state,
            &parsed_commitment_a,
            &statement_a,
            // &parsed_commitment_b,
            // &statement_b,
        )
        .unwrap();
    let verify_time = verif_time.elapsed();

    println!(
        "\nProving time: {} ms (commit: {} ms, opening: {} ms)",
        commit_time_a.as_millis() + opening_time.as_millis(),
        commit_time_a.as_millis(),
        // commit_time_b.as_millis(),
        opening_time.as_millis()
    );
    let proof_size =
        prover_state.proof_data().len() as f64 * (FPrimeSubfield::ORDER_U64 as f64).log2() / 8.0;
    println!("proof size: {:.2} KiB", proof_size / 1024.0);
    println!("Verification time: {} μs", verify_time.as_micros());
}
