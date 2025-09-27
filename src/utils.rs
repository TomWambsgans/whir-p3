use multilinear_toolkit::prelude::*;
use p3_matrix::{Matrix, dense::RowMajorMatrix};
use p3_util::log2_strict_usize;

use crate::EvalsDft;

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
pub(crate) fn sample_ood_points<EF: ExtensionField<PF<EF>>, E>(
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

pub(crate) enum DftInput<EF: Field> {
    Base(Vec<PF<EF>>),
    Extension(Vec<EF>),
}

pub(crate) enum DftOutput<EF: Field> {
    Base(RowMajorMatrix<PF<EF>>),
    Extension(RowMajorMatrix<EF>),
}

pub(crate) fn reorder_and_dft<EF: ExtensionField<PF<EF>>>(
    evals: &MleRef<'_, EF>,
    dft: &EvalsDft<PF<EF>>,
    folding_factor: usize,
    log_inv_rate: usize,
) -> DftOutput<EF>
where
    PF<EF>: TwoAdicField,
{
    let prepared_evals = prepare_evals_for_fft(&evals, folding_factor, log_inv_rate);
    let width = 1 << folding_factor;
    match prepared_evals {
        DftInput::Base(evals) => DftOutput::Base(
            dft.dft_algebra_batch_by_evals(RowMajorMatrix::new(evals, width))
                .to_row_major_matrix(),
        ),
        DftInput::Extension(evals) => DftOutput::Extension(
            dft.dft_algebra_batch_by_evals(RowMajorMatrix::new(evals, width))
                .to_row_major_matrix(),
        ),
    }
}

fn prepare_evals_for_fft<EF: ExtensionField<PF<EF>>>(
    evals: &MleRef<'_, EF>,
    folding_factor: usize,
    log_inv_rate: usize,
) -> DftInput<EF> {
    match evals {
        MleRef::Base(evals) => DftInput::Base(prepare_evals_for_fft_unpacked(
            evals,
            folding_factor,
            log_inv_rate,
        )),
        MleRef::BasePacked(evals) => DftInput::Base(prepare_evals_for_fft_unpacked(
            PFPacking::<EF>::unpack_slice(evals),
            folding_factor,
            log_inv_rate,
        )),
        MleRef::Extension(evals) => DftInput::Extension(prepare_evals_for_fft_unpacked(
            evals,
            folding_factor,
            log_inv_rate,
        )),
        MleRef::ExtensionPacked(evals) => DftInput::Extension(prepare_evals_for_fft_packed_extension(
            evals,
            folding_factor,
            log_inv_rate,
        )),
    }
}

fn prepare_evals_for_fft_unpacked<A: Copy + Send + Sync>(
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

fn prepare_evals_for_fft_packed_extension<EF: ExtensionField<PF<EF>>>(
    evals: &[EFPacking<EF>],
    folding_factor: usize,
    log_inv_rate: usize,
) -> Vec<EF> {
    let log_packing = packing_log_width::<EF>();
    assert!(evals.len() << log_packing % (1 << folding_factor) == 0);
    let n_blocks = 1 << folding_factor;
    let full_len = evals.len() << (log_inv_rate + log_packing);
    let block_size = full_len / n_blocks;
    let log_block_size = log2_strict_usize(block_size);
    let n_blocks_mask = n_blocks - 1;
    let packing_mask = (1 << log_packing) - 1;

    (0..full_len)
        .into_par_iter()
        .map(|i| {
            let block_index = i & n_blocks_mask;
            let offset_in_block = i >> folding_factor;
            let src_index = ((block_index << log_block_size) + offset_in_block) >> log_inv_rate;
            let packed_src_index = src_index >> log_packing;
            let offset_in_packing = src_index & packing_mask;
            let packed = unsafe { evals.get_unchecked(packed_src_index) };
            let unpacked: &[PFPacking<EF>] = packed.as_basis_coefficients_slice();
            EF::from_basis_coefficients_fn(|i| unsafe {
                let u: &PFPacking<EF> = unpacked.get_unchecked(i);
                *u.as_slice().get_unchecked(offset_in_packing)
            })
        })
        .collect()
}
