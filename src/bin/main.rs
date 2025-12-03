use std::time::Instant;

use multilinear_toolkit::prelude::*;
use p3_challenger::DuplexChallenger;
use p3_field::{PrimeCharacteristicRing, PrimeField64};
use p3_koala_bear::{
    KoalaBear, Poseidon2KoalaBear, QuinticExtensionFieldKB, default_koalabear_poseidon2_16,
};
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

    let poseidon16 = default_koalabear_poseidon2_16();

    type BaseField = F;

    let num_variables_a = 25;
    let num_coeffs_a = 1 << num_variables_a;
    let num_non_zero_coeffs_a = num_coeffs_a * 3 / 5;

    // Construct WHIR protocol parameters
    let params_a = WhirConfigBuilder {
        security_level: 128,
        max_num_variables_to_send_coeffs: 6,
        pow_bits: DEFAULT_MAX_POW,
        folding_factor: FoldingFactor::new(7, 4),
        soundness_type: SecurityAssumption::CapacityBound,
        starting_log_inv_rate: 1,
        rs_domain_initial_reduction_factor: 5,
    };
    let params_a = WhirConfig::new(params_a.clone(), num_variables_a);

    let mut rng = StdRng::seed_from_u64(0);
    let mut polynomial_a = (0..num_coeffs_a)
        .map(|_| rng.random())
        .collect::<Vec<BaseField>>();
    polynomial_a[num_non_zero_coeffs_a..].fill(BaseField::ZERO);

    let random_sparse_point = |rng: &mut StdRng, num_variables: usize| {
        let mut point = (0..num_variables)
            .map(|_| rng.random())
            .collect::<Vec<EF>>();
        let initial_booleans = rng.random_range(0..num_variables / 4);
        for i in 0..initial_booleans {
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

    let mut statement_a = Vec::new();

    // Add constraints for each sampled point (equality constraints)
    for point_a in &points_a {
        let eval = polynomial_a.evaluate(point_a);
        statement_a.push(Evaluation::new(point_a.clone(), eval));
    }

    let challenger = MyChallenger::new(poseidon16);

    let mut prover_state = ProverState::new(challenger.clone());

    precompute_dft_twiddles::<F>(1 << F::TWO_ADICITY);

    let polynomial_a = MleOwned::Base(polynomial_a);
    let time = Instant::now();
    let witness_a = params_a.commit(&mut prover_state, &polynomial_a);
    let commit_time_a = time.elapsed();

    let witness_a_clone = witness_a.clone();
    let time = Instant::now();
    params_a.prove(
        &mut prover_state,
        statement_a.clone(),
        witness_a_clone,
        &polynomial_a.by_ref(),
    );
    let opening_time_single = time.elapsed();
    let proof_size_single =
        prover_state.proof_size() as f64 * (EFPrimeSubfield::ORDER_U64 as f64).log2() / 8.0;

    let mut verifier_state = VerifierState::new(prover_state.into_proof(), challenger);

    let parsed_commitment_a = params_a.parse_commitment::<F>(&mut verifier_state).unwrap();

    params_a
        .verify::<F>(
            &mut verifier_state,
            &parsed_commitment_a,
            statement_a.clone(),
        )
        .unwrap();

    println!(
        "\nSingle proving time: {} ms (commit: {} ms, opening: {} ms)",
        commit_time_a.as_millis() + opening_time_single.as_millis(),
        commit_time_a.as_millis(),
        opening_time_single.as_millis()
    );

    println!(
        "\nTotal proving time: {} ms (commit: {} ms, opening: {} ms)",
        commit_time_a.as_millis() + opening_time_single.as_millis(),
        commit_time_a.as_millis(),
        opening_time_single.as_millis()
    );
    println!("proof size: {:.2} KiB", proof_size_single / 1024.0);
}
