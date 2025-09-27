use std::time::Instant;

use fiat_shamir::*;
use multilinear_toolkit::prelude::*;
use p3_challenger::DuplexChallenger;
use p3_field::{PrimeCharacteristicRing, PrimeField64};
use p3_koala_bear::{KoalaBear, Poseidon2KoalaBear, QuinticExtensionFieldKB};
use p3_symmetric::{PaddingFreeSponge, TruncatedPermutation};
use rand::{Rng, SeedableRng, rngs::StdRng};
// use tracing_forest::{ForestLayer, util::LevelFilter};
// use tracing_subscriber::{EnvFilter, Registry, layer::SubscriberExt, util::SubscriberInitExt};
use whir_p3::*;

// Commit A in F, B in EF
// TODO there is a big overhead embedding overhead in the sumcheck

type F = KoalaBear;
type EF = QuinticExtensionFieldKB;

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

    let vars_diff = 3;

    let num_variables_a = 25;
    let num_variables_b = num_variables_a - vars_diff;

    let num_coeffs_a = 1 << num_variables_a;
    let num_coeffs_b = 1 << num_variables_b;

    // Construct WHIR protocol parameters
    let params_a = WhirConfigBuilder {
        security_level: 128,
        max_num_variables_to_send_coeffs: 6,
        pow_bits: DEFAULT_MAX_POW,
        folding_factor: FoldingFactor::new(7, 4),
        merkle_hash: merkle_hash.clone(),
        merkle_compress: merkle_compress.clone(),
        soundness_type: SecurityAssumption::CapacityBound,
        starting_log_inv_rate: 1,
        rs_domain_initial_reduction_factor: 5,
    };

    let params_a = WhirConfig::new(params_a.clone(), num_variables_a);

    // println!("Using parameters:\n{}", params.to_string());

    let mut rng = StdRng::seed_from_u64(0);
    let polynomial_a = (0..num_coeffs_a)
        .map(|_| rng.random())
        .collect::<Vec<BaseFieldA>>();
    let polynomial_b = (0..num_coeffs_b)
        .map(|_| rng.random())
        .collect::<Vec<BaseFieldB>>();

    let random_sparse_point = |rng: &mut StdRng, num_variables: usize| {
        let mut point = (0..num_variables)
            .map(|_| rng.random())
            .collect::<Vec<EF>>();
        let initial_booleans = rng.random_range(0..num_variables / 4);
        for i in 0..initial_booleans {
            point[i] = EF::from_usize(rng.random_range(0..2));
        }
        let final_booleans = rng.random_range(0..num_variables / 4);
        for i in (num_variables - final_booleans)..num_variables {
            point[i] = EF::from_usize(rng.random_range(0..2));
        }
        MultilinearPoint(point)
    };

    // Sample `num_points` random multilinear points in the Boolean hypercube
    let mut points_a = (0..7)
        .map(|_| random_sparse_point(&mut rng, num_variables_a))
        .collect::<Vec<_>>();
    points_a.push(MultilinearPoint(vec![EF::ONE; num_variables_a]));
    points_a.push(MultilinearPoint(vec![EF::ZERO; num_variables_a]));
    let points_b = (0..9)
        .map(|_| random_sparse_point(&mut rng, num_variables_b))
        .collect::<Vec<_>>();

    // Construct a new statement with the correct number of variables
    let mut statement_a = Vec::new();
    let mut statement_b = Vec::new();

    // Add constraints for each sampled point (equality constraints)
    for point_a in &points_a {
        let eval = polynomial_a.evaluate(point_a);
        statement_a.push(Evaluation::new(point_a.clone(), eval));
    }
    for point_b in &points_b {
        let eval = polynomial_b.evaluate(point_b);
        statement_b.push(Evaluation::new(point_b.clone(), eval));
    }

    // Define the Fiat-Shamir domain separator pattern for committing and proving

    let challenger = MyChallenger::new(poseidon16);

    // Initialize the Merlin transcript from the IOPattern
    let mut prover_state = ProverState::new(challenger.clone());

    // Commit to the polynomial and produce a witness

    let dft = EvalsDft::<EFPrimeSubfield>::new(1 << params_a.max_fft_size());

    let time = Instant::now();
    let witness_a = params_a.commit(&dft, &mut prover_state, &polynomial_a);
    let commit_time_a = time.elapsed();

    let time = Instant::now();
    let commit_time_b = time.elapsed();

    // Generate a proof for the given statement and witness
    let time = Instant::now();
    params_a.prove(
        &dft,
        &mut prover_state,
        statement_a.clone(),
        witness_a,
        &polynomial_a,
    );

    let opening_time = time.elapsed();

    // Reconstruct verifier's view of the transcript using the DomainSeparator and prover's data
    let mut verifier_state = VerifierState::new(prover_state.proof_data().to_vec(), challenger);

    // Parse the commitment
    let parsed_commitment_a = params_a.parse_commitment(&mut verifier_state).unwrap();

    let verif_time = Instant::now();
    params_a
        .verify(&mut verifier_state, &parsed_commitment_a, statement_a)
        .unwrap();
    let verify_time = verif_time.elapsed();

    println!(
        "\nProving time: {} ms (commit A: {} ms, commit B: {} ms, opening: {} ms)",
        commit_time_a.as_millis() + commit_time_b.as_millis() + opening_time.as_millis(),
        commit_time_a.as_millis(),
        commit_time_b.as_millis(),
        opening_time.as_millis()
    );
    let proof_size =
        prover_state.proof_size() as f64 * (EFPrimeSubfield::ORDER_U64 as f64).log2() / 8.0;
    println!("proof size: {:.2} KiB", proof_size / 1024.0);
    println!("Verification time: {} μs", verify_time.as_micros());
}
