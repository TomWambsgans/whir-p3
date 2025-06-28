use crate::fiat_shamir::domain_separator::DomainSeparator;
use p3_challenger::{DuplexChallenger, GrindingChallenger};
use p3_field::PrimeCharacteristicRing;
use p3_field::extension::BinomialExtensionField;
use p3_keccak::KeccakF;
use p3_koala_bear::{KoalaBear, Poseidon2KoalaBear};
use p3_symmetric::Permutation;
use rand::{SeedableRng, rngs::SmallRng};

type F = KoalaBear;
type EF = BinomialExtensionField<F, 4>;

#[test]
fn bench_grinding() {
    // Define the Fiat-Shamir domain separator pattern for committing and proving
    let domainsep = DomainSeparator::<EF, F>::new(vec![]);
    let mut rng = SmallRng::seed_from_u64(1);
    let challenger =
        DuplexChallenger::<F, _, 16, 8>::new(Poseidon2KoalaBear::<16>::new_from_rng_128(&mut rng));

    // Initialize the Merlin transcript from the IOPattern
    let mut prover_state = domainsep.to_prover_state::<_, 8>(challenger.clone());
    let bits = 20;
    let time = std::time::Instant::now();
    prover_state.challenger.grind(bits);
    println!(
        "Grinding time: {} ms for {} bits",
        time.elapsed().as_millis(),
        bits
    );
}


#[test]
fn bench_poseidon() {
    let mut rng = SmallRng::seed_from_u64(1);
    let perm = Poseidon2KoalaBear::<16>::new_from_rng_128(&mut rng);
    let mut array = [F::ZERO; 16];
    let n = 1 << 20;
    let time = std::time::Instant::now();
    for _ in 0..n {
        perm.permute_mut(&mut array);
    }
    println!(
        "Poseidon_16 permutation time: {} ms for {} iterations",
        time.elapsed().as_millis(),
        n
    );

    let mut rng = SmallRng::seed_from_u64(1);
    let perm = Poseidon2KoalaBear::<24>::new_from_rng_128(&mut rng);
    let mut array = [F::ZERO; 24];
    let n = 1 << 20;
    let time = std::time::Instant::now();
    for _ in 0..n {
        perm.permute_mut(&mut array);
    }
    println!(
        "Poseidon_24 permutation time: {} ms for {} iterations",
        time.elapsed().as_millis(),
        n
    );


    let perm = KeccakF;
    let mut array = [0u8; 200];
    let n = 1 << 20;
    let time = std::time::Instant::now();
    for _ in 0..n {
        perm.permute_mut(&mut array);
    }
    println!(
        "Keccak permutation time: {} ms for {} iterations",
        time.elapsed().as_millis(),
        n
    );
}
