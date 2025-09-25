use fiat_shamir::*;
use fiat_shamir::{ChallengeSampler, FSChallenger, ProverState};
use multilinear_toolkit::MultilinearPoint;
use p3_field::*;
use p3_util::log2_strict_usize;
use rayon::prelude::*;

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
pub(crate) fn get_challenge_stir_queries<F: Field, Chal: ChallengeSampler<F>>(
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
pub(crate) fn sample_ood_points<F: Field, EF: ExtensionField<F>, E>(
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

pub(crate) fn prepare_evals_for_fft<A: Copy + Send + Sync>(
    evals: &[A],
    folding_factor: usize,
    log_inv_rate: usize,
) -> Vec<A> {
    assert!(evals.len() % (1 << folding_factor) == 0);
    let n_blocks = 1 << folding_factor;
    let full_len = evals.len() << log_inv_rate;
    let block_size = full_len / n_blocks;
    let log_block_size = log2_strict_usize(block_size);
    let n_blocks_mask = n_blocks - 1;

    (0..full_len)
        .into_par_iter()
        .map(|i| {
            let block_index = i & n_blocks_mask;
            let offset_in_block = i >> folding_factor;
            let src_index = ((block_index << log_block_size) + offset_in_block) >> log_inv_rate;
            unsafe { *evals.get_unchecked(src_index) }
        })
        .collect()
}
