use p3_field::{ExtensionField, Field};

use crate::{
    PF,
    fiat_shamir::{BitsSampler, FSChallenger, prover::ProverState},
    poly::multilinear::MultilinearPoint,
};

/// Computes the optimal workload size for `T` to fit in L1 cache (32 KB).
///
/// Ensures efficient memory access by dividing the cache size by `T`'s size.
/// The result represents how many elements of `T` can be processed per thread.
///
/// Helps minimize cache misses and improve performance in parallel workloads.
#[must_use]
pub const fn workload_size<T: Sized>() -> usize {
    const L1_CACHE_SIZE: usize = 1 << 15; // 32 KB
    L1_CACHE_SIZE / size_of::<T>()
}

/// Samples a list of unique query indices from a folded evaluation domain, using transcript randomness.
///
/// This function is used to select random query locations for verifying proximity to a folded codeword.
/// The folding reduces the domain size exponentially (e.g. by 2^folding_factor), so we sample indices
/// in the reduced "folded" domain.
///
/// ## Parameters
/// - `domain_size`: The size of the original evaluation domain (e.g., 2^22).
/// - `folding_factor`: The number of folding rounds applied (e.g., k = 1 means domain halves).
/// - `num_queries`: The number of query *indices* we want to obtain.
/// - `challenger`: A Fiat–Shamir transcript used to sample randomness deterministically.
///
/// ## Returns
/// A sorted and deduplicated list of random query indices in the folded domain.
pub fn get_challenge_stir_queries<F: Field, Chal: BitsSampler<F>>(
    folded_domain_size: usize,
    num_queries: usize,
    prover_state: &mut Chal,
) -> Vec<usize> {
    (0..num_queries)
        .map(|_| prover_state.sample_bits(folded_domain_size.ilog2() as usize))
        .collect()
}

/// A utility function to sample Out-of-Domain (OOD) points and evaluate them.
///
/// This should be used on the prover side.
pub fn sample_ood_points<F: Field, EF: ExtensionField<F>, E>(
    prover_state: &mut ProverState<PF<EF>, EF, impl FSChallenger<EF>>,
    num_samples: usize,
    num_variables: usize,
    evaluate_fn: E,
) -> (Vec<EF>, Vec<EF>)
where
    E: Fn(&MultilinearPoint<EF>) -> EF,
    EF: ExtensionField<PF<EF>>,
{
    let mut ood_points = EF::zero_vec(num_samples);
    let mut ood_answers = Vec::with_capacity(num_samples);

    if num_samples > 0 {
        // Generate OOD points from ProverState randomness
        for ood_point in &mut ood_points {
            *ood_point = prover_state.sample();
        }

        // Evaluate the function at each OOD point
        ood_answers.extend(ood_points.iter().map(|ood_point| {
            evaluate_fn(&MultilinearPoint::expand_from_univariate(
                *ood_point,
                num_variables,
            ))
        }));

        prover_state.add_extension_scalars(&ood_answers);
    }

    (ood_points, ood_answers)
}
