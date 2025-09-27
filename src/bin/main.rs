use std::time::Instant;

use multilinear_toolkit::prelude::*;
use p3_challenger::DuplexChallenger;
use p3_field::{extension::BinomialExtensionField, PrimeCharacteristicRing, PrimeField64};
use p3_koala_bear::{KoalaBear, Poseidon2KoalaBear};
use p3_symmetric::{PaddingFreeSponge, TruncatedPermutation};
use rand::{Rng, SeedableRng, rngs::StdRng};
// use tracing_forest::{ForestLayer, util::LevelFilter};
// use tracing_subscriber::{EnvFilter, Registry, layer::SubscriberExt, util::SubscriberInitExt};
use whir_p3::*;

// Commit A in F, B in EF
// TODO there is a big overhead embedding overhead in the sumcheck

type F = KoalaBear;
type EF = BinomialExtensionField<KoalaBear, 4>;

type EFPrimeSubfield = <EF as PrimeCharacteristicRing>::PrimeSubfield;

type Poseidon16 = Poseidon2KoalaBear<16>;
type Poseidon24 = Poseidon2KoalaBear<24>;

type MerkleHash = PaddingFreeSponge<Poseidon24, 24, 16, 8>; // leaf hashing
type MerkleCompress = TruncatedPermutation<Poseidon16, 2, 8, 16>; // 2-to-1 compression
type MyChallenger = DuplexChallenger<EFPrimeSubfield, Poseidon16, 16, 8>;

fn main() {
    // let env_filter: EnvFilter = EnvFilter::builder()
    //     .with_default_directive(LevelFilter::INFO.into())
    //     .from_env_lossy();

    // Registry::default()
    //     .with(env_filter)
    //     .with(ForestLayer::default())
    //     .init();

    // Create hash and compression functions for the Merkle tree
    let poseidon16 = Poseidon16::new_from_rng_128(&mut StdRng::seed_from_u64(0));
    let poseidon24 = Poseidon24::new_from_rng_128(&mut StdRng::seed_from_u64(0));

    let merkle_hash = MerkleHash::new(poseidon24);
    let merkle_compress = MerkleCompress::new(poseidon16.clone());

    type BaseFieldA = F;
    type BaseFieldB = EF;


    let num_variables_a = 25;

    let num_coeffs_a = 1 << num_variables_a;

    // Construct WHIR protocol parameters
    let params_a = WhirConfigBuilder {
        security_level: 90,
        max_num_variables_to_send_coeffs: 6,
        pow_bits: DEFAULT_MAX_POW,
        folding_factor: FoldingFactor::new(5, 5),
        merkle_hash: merkle_hash.clone(),
        merkle_compress: merkle_compress.clone(),
        soundness_type: SecurityAssumption::CapacityBound,
        starting_log_inv_rate: 1,
        rs_domain_initial_reduction_factor: 3,
    };
    
    let params_a = WhirConfig::new(params_a.clone(), num_variables_a);

    // println!("Using parameters:\n{}", params.to_string());

    let mut rng = StdRng::seed_from_u64(0);
    let polynomial_a = (0..num_coeffs_a)
        .map(|_| rng.random())
        .collect::<Vec<BaseFieldA>>();


    // Sample `num_points` random multilinear points in the Boolean hypercube
    let points_a: Vec<MultilinearPoint<EF>> = (0..1)
        .map(|_| MultilinearPoint((0..num_variables_a).map(|_| rng.random()).collect()))
        .collect();

    // Construct a new statement with the correct number of variables
    let mut statement_a = Vec::new();

    // Add constraints for each sampled point (equality constraints)
    for point_a in &points_a {
        let eval = polynomial_a.evaluate(point_a);
        statement_a.push(Evaluation::new(point_a.clone(), eval));
    }

    // Define the Fiat-Shamir domain separator pattern for committing and proving

    let challenger = MyChallenger::new(poseidon16);

    // Initialize the Merlin transcript from the IOPattern
    let mut prover_state = ProverState::new(challenger.clone());

    // Commit to the polynomial and produce a witness

    let dft = EvalsDft::<EFPrimeSubfield>::new(1 << params_a.max_fft_size());

    let polynomial_a = MleOwned::Base(polynomial_a);
    let time = Instant::now();
    let witness_a = params_a.commit(&dft, &mut prover_state, &polynomial_a);
    let commit_time_a = time.elapsed();

    // let time = Instant::now();
    // let witness_b = params_b.commit(&dft, &mut prover_state, &polynomial_b);
    // let commit_time_b = time.elapsed();

    // Generate a proof for the given statement and witness
    let time = Instant::now();
    params_a.prove(
        &dft,
        &mut prover_state,
        statement_a.clone(),
        witness_a,
        &polynomial_a.by_ref(),
        // statement_b.clone(),
        // witness_b,
        // &polynomial_b.by_ref(),
    );

    let opening_time = time.elapsed();

    // Reconstruct verifier's view of the transcript using the DomainSeparator and prover's data
    let mut verifier_state = VerifierState::new(prover_state.proof_data().to_vec(), challenger);

    // // Parse the commitment
     let parsed_commitment_a = params_a.parse_commitment::<F>(&mut verifier_state).unwrap();
    // let parsed_commitment_b = params_b
    //     .parse_commitment::<EF>(&mut verifier_state)
    //     .unwrap();

    let verif_time = Instant::now();
    params_a
        .verify(
            &mut verifier_state,
            &parsed_commitment_a,
            statement_a,
            // &parsed_commitment_b,
            // statement_b,
        )
        .unwrap();
    let verify_time = verif_time.elapsed();

    println!(
        "\nProving time: {} ms (commit A: {} ms,  opening: {} ms)",
        commit_time_a.as_millis()+ opening_time.as_millis(),
        commit_time_a.as_millis(),
        opening_time.as_millis()
    );
    let proof_size =
        prover_state.proof_size() as f64 * (EFPrimeSubfield::ORDER_U64 as f64).log2() / 8.0;
    println!("proof size: {:.2} KiB", proof_size / 1024.0);
    println!("Verification time: {} μs", verify_time.as_micros());
}
