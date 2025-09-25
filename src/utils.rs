use p3_field::{
    Algebra, BasedVectorSpace, ExtensionField, Field, PackedFieldExtension, PackedValue,
    PrimeCharacteristicRing,
};
use p3_util::{iter_array_chunks_padded, log2_strict_usize};
use rayon::prelude::*;

use crate::*;

/// Log of number of threads to spawn.
/// Long term this should be a modifiable parameter and potentially be in an optimization file somewhere.
/// I've chosen 32 here as my machine has 20 logical cores.
const LOG_NUM_THREADS: usize = 5;

/// The number of threads to spawn for parallel computations.
const NUM_THREADS: usize = 1 << LOG_NUM_THREADS;

#[inline]
pub fn compute_sparse_eval_eq<F, EF>(eval: &[EF], out: &mut [EF], scalar: EF)
where
    F: Field,
    EF: ExtensionField<F>,
{
    let boolean_starts = eval
        .iter()
        .take_while(|&&x| x.is_zero() || x.is_one())
        .map(|&x| x.is_one())
        .collect::<Vec<_>>();
    let starts_big_endian = boolean_starts
        .iter()
        .fold(0, |acc, &bit| (acc << 1) | (bit as usize));

    if boolean_starts.len() == eval.len() {
        // full of booleans
        out[starts_big_endian] += scalar;
        return;
    }

    let mut boolean_ends = eval
        .iter()
        .rev()
        .take_while(|&&x| x.is_zero() || x.is_one())
        .map(|&x| x.is_one())
        .collect::<Vec<_>>();
    boolean_ends.reverse();
    let ends_big_endian = boolean_ends
        .iter()
        .fold(0, |acc, &bit| (acc << 1) | (bit as usize));

    let eval = &eval[boolean_starts.len()..];
    let new_out_size = 1 << eval.len();
    let out = &mut out[starts_big_endian * new_out_size..(starts_big_endian + 1) * new_out_size];

    if boolean_ends.len() == 0 {
        compute_eval_eq::<F, EF, true>(eval, out, scalar);
    } else {
        let mut buff = unsafe { uninitialized_vec::<EF>(out.len() >> boolean_ends.len()) };
        compute_eval_eq::<F, EF, false>(
            &eval[..eval.len() - boolean_ends.len()],
            &mut buff,
            scalar,
        );
        out[ends_big_endian..]
            .par_iter_mut()
            .step_by(1 << boolean_ends.len())
            .zip(buff.into_par_iter())
            .for_each(|(o, v)| {
                *o += v;
            });
    }
}

/// Computes the equality polynomial evaluations efficiently.
///
/// Given an evaluation point vector `eval`, the function computes
/// the equality polynomial recursively using the formula:
///
/// ```text
/// eq(X) = ∏ (1 - X_i + 2X_i z_i)
/// ```
///
/// where `z_i` are the constraint points.
///
/// If INITIALIZED is:
/// - false: the result is directly set to the `out` buffer
/// - true: the result is added to the `out` buffer
#[inline]
pub fn compute_eval_eq<F, EF, const INITIALIZED: bool>(eval: &[EF], out: &mut [EF], scalar: EF)
where
    F: Field,
    EF: ExtensionField<F>,
{
    // It's possible for this to be called with F = EF (Despite F actually being an extension field).
    //
    // IMPORTANT: We previously checked here that `packing_width > 1`,
    // but this check is **not viable** for Goldilocks on Neon or when not using `target-cpu=native`.
    //
    // Why? Because Neon SIMD vectors are 128 bits and Goldilocks elements are already 64 bits,
    // so no packing happens (width stays 1), and there's no performance advantage.
    //
    // Be careful: this means code relying on packing optimizations should **not assume**
    // `packing_width > 1` is always true.
    let packing_width = F::Packing::WIDTH;
    // debug_assert!(packing_width > 1);

    // Ensure that the output buffer size is correct:
    // It should be of size `2^n`, where `n` is the number of variables.
    debug_assert_eq!(out.len(), 1 << eval.len());

    // If the number of variables is small, there is no need to use
    // parallelization or packings.
    if eval.len() <= packing_width + 1 + LOG_NUM_THREADS {
        // A basic recursive approach.
        eval_eq_basic::<_, _, _, INITIALIZED>(eval, out, scalar);
    } else {
        let log_packing_width = log2_strict_usize(packing_width);
        let eval_len_min_packing = eval.len() - log_packing_width;

        // We split eval into three parts:
        // - eval[..LOG_NUM_THREADS] (the first LOG_NUM_THREADS elements)
        // - eval[LOG_NUM_THREADS..eval_len_min_packing] (the middle elements)
        // - eval[eval_len_min_packing..] (the last log_packing_width elements)

        // The middle elements are the ones which will be computed in parallel.
        // The last log_packing_width elements are the ones which will be packed.

        // We make a buffer of elements of size `NUM_THREADS`.
        let mut parallel_buffer = EF::ExtensionPacking::zero_vec(NUM_THREADS);
        let out_chunk_size = out.len() / NUM_THREADS;

        // Compute the equality polynomial corresponding to the last log_packing_width elements
        // and pack these.
        parallel_buffer[0] = packed_eq_poly(&eval[eval_len_min_packing..], scalar);

        // Update the buffer so it contains the evaluations of the equality polynomial
        // with respect to parts one and three.
        fill_buffer(eval[..LOG_NUM_THREADS].iter().rev(), &mut parallel_buffer);

        // Finally do all computations involving the middle elements in parallel.
        out.par_chunks_exact_mut(out_chunk_size)
            .zip(parallel_buffer.par_iter())
            .for_each(|(out_chunk, buffer_val)| {
                eval_eq_packed::<_, _, INITIALIZED>(
                    &eval[LOG_NUM_THREADS..(eval.len() - log_packing_width)],
                    out_chunk,
                    *buffer_val,
                );
            });
    }
}

/// Computes the equality polynomial evaluations efficiently.
///
/// This function is similar to [`eval_eq`], but it assumes that we want to evaluate
/// at a base field point instead of an extension field point. This leads to a different
/// strategy which can better minimize data transfers.
///
/// Given an evaluation point vector `eval`, the function computes
/// the equality polynomial recursively using the formula:
///
/// ```text
/// eq(X) = ∏ (1 - X_i + 2X_i z_i)
/// ```
///
/// where `z_i` are the constraint points.
///
/// If INITIALIZED is:
/// - false: the result is directly set to the `out` buffer
/// - true: the result is added to the `out` buffer
#[inline]
pub fn compute_eval_eq_base<F, EF, const INITIALIZED: bool>(eval: &[F], out: &mut [EF], scalar: EF)
where
    F: Field,
    EF: ExtensionField<F>,
{
    // we assume that packing_width is a power of 2.
    let packing_width = F::Packing::WIDTH;

    // Ensure that the output buffer size is correct:
    // It should be of size `2^n`, where `n` is the number of variables.
    debug_assert_eq!(out.len(), 1 << eval.len());

    // If the number of variables is small, there is no need to use
    // parallelization or packings.
    if eval.len() <= packing_width + 1 + LOG_NUM_THREADS {
        // A basic recursive approach.
        eval_eq_basic::<_, _, _, INITIALIZED>(eval, out, scalar);
    } else {
        let log_packing_width = log2_strict_usize(packing_width);
        let eval_len_min_packing = eval.len() - log_packing_width;

        // We split eval into three parts:
        // - eval[..LOG_NUM_THREADS] (the first LOG_NUM_THREADS elements)
        // - eval[LOG_NUM_THREADS..eval_len_min_packing] (the middle elements)
        // - eval[eval_len_min_packing..] (the last log_packing_width elements)

        // The middle elements are the ones which will be computed in parallel.
        // The last log_packing_width elements are the ones which will be packed.

        // We make a buffer of PackedField elements of size `NUM_THREADS`.
        // Note that this is a slightly different strategy to `eval_eq` which instead
        // uses PackedExtensionField elements. Whilst this involves slightly more mathematical
        // operations, it seems to be faster in practice due to less data moving around.
        let mut parallel_buffer = F::Packing::zero_vec(NUM_THREADS);
        let out_chunk_size = out.len() / NUM_THREADS;

        // Compute the equality polynomial corresponding to the last log_packing_width elements
        // and pack these.
        parallel_buffer[0] = packed_eq_poly(&eval[eval_len_min_packing..], F::ONE);

        // Update the buffer so it contains the evaluations of the equality polynomial
        // with respect to parts one and three.
        fill_buffer(eval[..LOG_NUM_THREADS].iter().rev(), &mut parallel_buffer);

        // Finally do all computations involving the middle elements in parallel.
        out.par_chunks_exact_mut(out_chunk_size)
            .zip(parallel_buffer.par_iter())
            .for_each(|(out_chunk, buffer_val)| {
                base_eval_eq_packed::<_, _, INITIALIZED>(
                    &eval[LOG_NUM_THREADS..(eval.len() - log_packing_width)],
                    out_chunk,
                    *buffer_val,
                    scalar,
                );
            });
    }
}

/// Fills the `buffer` with evaluations of the equality polynomial
/// of degree `points.len()` multiplied by the value at `buffer[0]`.
///
/// Assume that `buffer[0]` contains `{eq(i, x)}` for `i \in \{0, 1\}^j` packed into a single
/// PackedExtensionField element. This function fills out the remainder of the buffer so that
/// `buffer[ind]` contains `{eq(ind, points) * eq(i, x)}` for `i \in \{0, 1\}^j`. Note that
/// `ind` is interpreted as an element of `\{0, 1\}^{points.len()}`.
#[allow(clippy::inline_always)] // Adding inline(always) seems to give a small performance boost.
#[inline(always)]
fn fill_buffer<'a, F, A>(points: impl ExactSizeIterator<Item = &'a F>, buffer: &mut [A])
where
    F: Field,
    A: Algebra<F> + Copy,
{
    for (ind, &entry) in points.enumerate() {
        let stride = 1 << ind;

        for index in 0..stride {
            let val = buffer[index];
            let scaled_val = val * entry;
            let new_val = val - scaled_val;

            buffer[index] = new_val;
            buffer[index + stride] = scaled_val;
        }
    }
}

/// Compute the scaled multilinear equality polynomial over `{0,1}` for a single variable.
///
/// This is the hardcoded base case for the equality polynomial `eq(x, z)`
/// in the case of a single variable `z = [z_0] ∈ 𝔽`, and returns:
///
/// \begin{equation}
/// [α ⋅ (1 - z_0), α ⋅ z_0]
/// \end{equation}
///
/// corresponding to the evaluations:
///
/// \begin{equation}
/// [α ⋅ eq(0, z), α ⋅ eq(1, z)]
/// \end{equation}
///
/// where the multilinear equality function is:
///
/// \begin{equation}
/// eq(x, z) = x ⋅ z + (1 - x)(1 - z)
/// \end{equation}
///
/// Concretely:
/// - For `x = 0`, we have:
///   \begin{equation}
///   eq(0, z_0) = 0 ⋅ z_0 + (1 - 0)(1 - z_0) = 1 - z_0
///   \end{equation}
/// - For `x = 1`, we have:
///   \begin{equation}
///   eq(1, z_0) = 1 ⋅ z_0 + (1 - 1)(1 - z_0) = z_0
///   \end{equation}
///
/// So the return value is:
/// - `[α ⋅ (1 - z_0), α ⋅ z_0]`
///
/// # Arguments
/// - `eval`: Slice containing the evaluation point `[z_0]` (must have length 1)
/// - `scalar`: A scalar multiplier `α` to scale the result by
///
/// # Returns
/// An array `[α ⋅ (1 - z_0), α ⋅ z_0]` representing the scaled evaluations
/// of `eq(x, z)` for `x ∈ {0,1}`.
#[allow(clippy::inline_always)] // Adding inline(always) seems to give a small performance boost.
#[inline(always)]
fn eval_eq_1<F, FP>(eval: &[F], scalar: FP) -> [FP; 2]
where
    F: Field,
    FP: Algebra<F> + Copy,
{
    assert_eq!(eval.len(), 1);

    // Extract the evaluation point z_0
    let z_0 = eval[0];

    // Compute α ⋅ z_0 = α ⋅ eq(1, z) and α ⋅ (1 - z_0) = α - α ⋅ z_0 = α ⋅ eq(0, z)
    let eq_1 = scalar * z_0;
    let eq_0 = scalar - eq_1;

    [eq_0, eq_1]
}

/// Compute the scaled multilinear equality polynomial over `{0,1}^2`.
///
/// This is the hardcoded base case for the multilinear equality polynomial `eq(x, z)`
/// when the evaluation point has 2 variables: `z = [z_0, z_1] ∈ 𝔽²`.
///
/// It computes and returns the vector:
///
/// \begin{equation}
/// [α ⋅ eq((0,0), z), α ⋅ eq((0,1), z), α ⋅ eq((1,0), z), α ⋅ eq((1,1), z)]
/// \end{equation}
///
/// where the multilinear equality polynomial is:
///
/// \begin{equation}
/// eq(x, z) = ∏_{i=0}^{1} (x_i ⋅ z_i + (1 - x_i)(1 - z_i))
/// \end{equation}
///
/// Concretely, this gives:
/// - `eq((0,0), z) = (1 - z_0)(1 - z_1)`
/// - `eq((0,1), z) = (1 - z_0)(z_1)`
/// - `eq((1,0), z) = z_0(1 - z_1)`
/// - `eq((1,1), z) = z_0(z_1)`
///
/// Then all outputs are scaled by `α`.
///
/// # Arguments
/// - `eval`: Slice `[z_0, z_1]`, the evaluation point in `𝔽²`
/// - `scalar`: The scalar multiplier `α ∈ 𝔽`
///
/// # Returns
/// An array `[α ⋅ eq((0,0), z), α ⋅ eq((0,1), z), α ⋅ eq((1,0), z), α ⋅ eq((1,1), z)]`
#[allow(clippy::inline_always)] // Helps with performance in tight loops
#[inline(always)]
fn eval_eq_2<F, FP>(eval: &[F], scalar: FP) -> [FP; 4]
where
    F: Field,
    FP: Algebra<F> + Copy,
{
    assert_eq!(eval.len(), 2);

    // Extract z_0, z_1 from the evaluation point
    let z_0 = eval[0];
    let z_1 = eval[1];

    // Compute s1 = α ⋅ z_0 = α ⋅ eq(1, -) and s0 = α - s1 = α ⋅ (1 - z_0) = α ⋅ eq(0, -)
    let s1 = scalar * z_0;
    let s0 = scalar - s1;

    // For x_0 = 0:
    // - s01 = s0 ⋅ z_1 = α ⋅ (1 - z_0) ⋅ z_1 = α ⋅ eq((0,1), z)
    // - s00 = s0 - s01 = α ⋅ (1 - z_0)(1 - z_1) = α ⋅ eq((0,0), z)
    let s01 = s0 * z_1;
    let s00 = s0 - s01;

    // For x_0 = 1:
    // - s11 = s1 ⋅ z_1 = α ⋅ z_0 ⋅ z_1 = α ⋅ eq((1,1), z)
    // - s10 = s1 - s11 = α ⋅ z_0(1 - z_1) = α ⋅ eq((1,0), z)
    let s11 = s1 * z_1;
    let s10 = s1 - s11;

    // Return values in lexicographic order of x = (x_0, x_1)
    [s00, s01, s10, s11]
}

/// Compute the scaled multilinear equality polynomial over `{0,1}³` for 3 variables.
///
/// This is the hardcoded base case for the equality polynomial `eq(x, z)`
/// in the case of three variables `z = [z_0, z_1, z_2] ∈ 𝔽³`, and returns:
///
/// \begin{equation}
/// [α ⋅ eq((0,0,0), z), α ⋅ eq((0,0,1), z), ..., α ⋅ eq((1,1,1), z)]
/// \end{equation}
///
/// where the multilinear equality function is defined as:
///
/// \begin{equation}
/// \mathrm{eq}(x, z) = \prod_{i=0}^{2} \left( x_i z_i + (1 - x_i)(1 - z_i) \right)
/// \end{equation}
///
/// For each binary vector `x ∈ {0,1}³`, this returns the scaled evaluation `α ⋅ eq(x, z)`,
/// in lexicographic order: `(0,0,0), (0,0,1), ..., (1,1,1)`.
///
/// # Arguments
/// - `eval`: A slice containing `[z_0, z_1, z_2]`, the evaluation point.
/// - `scalar`: A scalar multiplier `α` to apply to all results.
///
/// # Returns
/// An array of 8 values `[α ⋅ eq(x, z)]` for all `x ∈ {0,1}³`, in lex order.
#[allow(clippy::inline_always)] // Adding inline(always) seems to give a small performance boost.
#[inline(always)]
fn eval_eq_3<F, FP>(eval: &[F], scalar: FP) -> [FP; 8]
where
    F: Field,
    FP: Algebra<F> + Copy,
{
    assert_eq!(eval.len(), 3);

    // Extract z_0, z_1, z_2 from the evaluation point
    let z_0 = eval[0];
    let z_1 = eval[1];
    let z_2 = eval[2];

    // First dimension split: scalar * z_0 and scalar * (1 - z_0)
    let s1 = scalar * z_0; // α ⋅ z_0
    let s0 = scalar - s1; // α ⋅ (1 - z_0)

    // Second dimension split:
    // Group (0, x1) branch using s0 = α ⋅ (1 - z_0)
    let s01 = s0 * z_1; // α ⋅ (1 - z_0) ⋅ z_1
    let s00 = s0 - s01; // α ⋅ (1 - z_0) ⋅ (1 - z_1)

    // Group (1, x1) branch using s1 = α ⋅ z_0
    let s11 = s1 * z_1; // α ⋅ z_0 ⋅ z_1
    let s10 = s1 - s11; // α ⋅ z_0 ⋅ (1 - z_1)

    // Third dimension split:
    // For (0,0,x2) branch
    let s001 = s00 * z_2; // α ⋅ (1 - z_0)(1 - z_1) ⋅ z_2
    let s000 = s00 - s001; // α ⋅ (1 - z_0)(1 - z_1) ⋅ (1 - z_2)

    // For (0,1,x2) branch
    let s011 = s01 * z_2; // α ⋅ (1 - z_0) ⋅ z_1 ⋅ z_2
    let s010 = s01 - s011; // α ⋅ (1 - z_0) ⋅ z_1 ⋅ (1 - z_2)

    // For (1,0,x2) branch
    let s101 = s10 * z_2; // α ⋅ z_0 ⋅ (1 - z_1) ⋅ z_2
    let s100 = s10 - s101; // α ⋅ z_0 ⋅ (1 - z_1) ⋅ (1 - z_2)

    // For (1,1,x2) branch
    let s111 = s11 * z_2; // α ⋅ z_0 ⋅ z_1 ⋅ z_2
    let s110 = s11 - s111; // α ⋅ z_0 ⋅ z_1 ⋅ (1 - z_2)

    // Return all 8 evaluations in lexicographic order of x ∈ {0,1}³
    [s000, s001, s010, s011, s100, s101, s110, s111]
}

/// Computes the equality polynomial evaluations via a simple recursive algorithm.
///
/// Given an evaluation point vector `eval`, the function computes
/// the equality polynomial recursively using the formula:
///
/// ```text
/// eq(X) = scalar * ∏ (1 - X_i + 2X_i z_i)
/// ```
///
/// where `z_i` are the constraint points.
///
/// If INITIALIZED is:
/// - false: the result is directly set to the `out` buffer
/// - true: the result is added to the `out` buffer
#[allow(clippy::too_many_lines)]
#[inline]
fn eval_eq_basic<F, IF, EF, const INITIALIZED: bool>(eval: &[IF], out: &mut [EF], scalar: EF)
where
    F: Field,
    IF: Field,
    EF: ExtensionField<F> + Algebra<IF>,
{
    // Ensure that the output buffer size is correct:
    // It should be of size `2^n`, where `n` is the number of variables.
    debug_assert_eq!(out.len(), 1 << eval.len());

    match eval.len() {
        0 => {
            if INITIALIZED {
                out[0] += scalar;
            } else {
                out[0] = scalar;
            }
        }
        1 => {
            // Manually unroll for single variable case
            let eq_evaluations = eval_eq_1(eval, scalar);

            add_or_set::<_, INITIALIZED>(out, &eq_evaluations);
        }
        2 => {
            // Manually unroll for two variable case
            let eq_evaluations = eval_eq_2(eval, scalar);

            add_or_set::<_, INITIALIZED>(out, &eq_evaluations);
        }
        3 => {
            // Manually unroll for three variable case
            let eq_evaluations = eval_eq_3(eval, scalar);

            add_or_set::<_, INITIALIZED>(out, &eq_evaluations);
        }
        _ => {
            let (&x, tail) = eval.split_first().unwrap();

            // Divide the output buffer into two halves: one for `X_i = 0` and one for `X_i = 1`
            let (low, high) = out.split_at_mut(out.len() / 2);

            // Compute weight updates for the two branches:
            // - `s0` corresponds to the case when `X_i = 0`
            // - `s1` corresponds to the case when `X_i = 1`
            //
            // Mathematically, this follows the recurrence:
            // ```text
            // eq_{X1, ..., Xn}(X) = (1 - X_1) * eq_{X2, ..., Xn}(X) + X_1 * eq_{X2, ..., Xn}(X)
            // ```
            let s1 = scalar * x; // Contribution when `X_i = 1`
            let s0 = scalar - s1; // Contribution when `X_i = 0`

            // The recursive approach turns out to be faster than the iterative one here.
            // Probably related to nice cache locality.
            eval_eq_basic::<_, _, _, INITIALIZED>(tail, low, s0);
            eval_eq_basic::<_, _, _, INITIALIZED>(tail, high, s1);
        }
    }
}

/// Computes the equality polynomial evaluations via a simple recursive algorithm.
///
/// Unlike [`eval_eq_basic`], this function makes heavy use of packed values to speed up computations.
/// In particular `scalar` should be passed in as a packed value coming from [`packed_eq_poly`].
///
/// Essentially using packings this functions computes
///
/// ```text
/// eq(X) = scalar[j] * ∏ (1 - X_i + 2X_i z_i)
/// ```
///
/// for a collection of `i` at the same time. Here `scalar[j]` should be thought of as evaluations of an equality
/// polynomial over different variables so `eq(X)` ends up being the evaluation of the equality polynomial over
/// the combined set of variables.
///
/// It then updates the output buffer `out` with the computed values by adding them in.
#[allow(clippy::too_many_lines)]
#[inline]
fn eval_eq_packed<F: Field, EF: ExtensionField<F>, const INITIALIZED: bool>(
    eval: &[EF],
    out: &mut [EF],
    scalar: EF::ExtensionPacking,
) {
    // Ensure that the output buffer size is correct:
    // It should be of size `2^n`, where `n` is the number of variables.
    let width = F::Packing::WIDTH;
    debug_assert_eq!(out.len(), width << eval.len());

    match eval.len() {
        0 => {
            let result: Vec<EF> = EF::ExtensionPacking::to_ext_iter([scalar]).collect();
            add_or_set::<_, INITIALIZED>(out, &result);
        }
        1 => {
            // Manually unroll for single variable case
            let eq_evaluations = eval_eq_1(eval, scalar);

            let result: Vec<EF> = EF::ExtensionPacking::to_ext_iter(eq_evaluations).collect();
            add_or_set::<_, INITIALIZED>(out, &result);
        }
        2 => {
            // Manually unroll for two variables case
            let eq_evaluations = eval_eq_2(eval, scalar);

            let result: Vec<EF> = EF::ExtensionPacking::to_ext_iter(eq_evaluations).collect();
            add_or_set::<_, INITIALIZED>(out, &result);
        }
        3 => {
            const EVAL_LEN: usize = 8;

            // Manually unroll for three variable case
            let eq_evaluations = eval_eq_3(eval, scalar);

            // Unpack the evaluations back into EF elements and add to output.
            // We use `iter_array_chunks_padded` to allow us to use `add_slices` without
            // needing a vector allocation. Note that `eq_evaluations: [EF::ExtensionPacking: 8]`
            // so we know that `out.len() = 8 * F::Packing::WIDTH` meaning we can use `chunks_exact_mut`
            // and `iter_array_chunks_padded` will never actually pad anything.
            // This avoids the allocation used to accumulate `result` in the other branches. We could
            // do a similar strategy in those branches but, those branches should only be hit
            // infrequently in small cases which are already sufficiently fast.
            iter_array_chunks_padded::<_, EVAL_LEN>(
                EF::ExtensionPacking::to_ext_iter(eq_evaluations),
                EF::ZERO,
            )
            .zip(out.chunks_exact_mut(EVAL_LEN))
            .for_each(|(res, out_chunk)| {
                if INITIALIZED {
                    EF::add_slices(out_chunk, &res);
                } else {
                    out_chunk.copy_from_slice(&res);
                }
            });
        }
        _ => {
            let (&x, tail) = eval.split_first().unwrap();

            // Divide the output buffer into two halves: one for `X_i = 0` and one for `X_i = 1`
            let (low, high) = out.split_at_mut(out.len() / 2);

            // Compute weight updates for the two branches:
            // - `s0` corresponds to the case when `X_i = 0`
            // - `s1` corresponds to the case when `X_i = 1`
            //
            // Mathematically, this follows the recurrence:
            // ```text
            // eq_{X1, ..., Xn}(X) = (1 - X_1) * eq_{X2, ..., Xn}(X) + X_1 * eq_{X2, ..., Xn}(X)
            // ```
            let s1 = scalar * x; // Contribution when `X_i = 1`
            let s0 = scalar - s1; // Contribution when `X_i = 0`

            // The recursive approach turns out to be faster than the iterative one here.
            // Probably related to nice cache locality.
            eval_eq_packed::<_, _, INITIALIZED>(tail, low, s0);
            eval_eq_packed::<_, _, INITIALIZED>(tail, high, s1);
        }
    }
}

/// Computes the equality polynomial evaluations via a simple recursive algorithm.
///
/// Unlike [`eval_eq_basic`], this function makes heavy use of packed values to speed up computations.
/// In particular `scalar` should be passed in as a packed value coming from [`packed_eq_poly`].
///
/// Essentially using packings this functions computes
///
/// ```text
/// eq(X) = scalar[j] * ∏ (1 - X_i + 2X_i z_i)
/// ```
///
/// for a collection of `i` at the same time. Here `scalar[j]` should be thought of as evaluations of an equality
/// polynomial over different variables so `eq(X)` ends up being the evaluation of the equality polynomial over
/// the combined set of variables.
///
/// It then updates the output buffer `out` with the computed values by adding them in.
#[allow(clippy::too_many_lines)]
#[inline]
fn base_eval_eq_packed<F, EF, const INITIALIZED: bool>(
    eval_points: &[F],
    out: &mut [EF],
    eq_evals: F::Packing,
    scalar: EF,
) where
    F: Field,
    EF: ExtensionField<F>,
{
    // Ensure that the output buffer size is correct:
    // It should be of size `2^n`, where `n` is the number of variables.
    let width = F::Packing::WIDTH;
    debug_assert_eq!(out.len(), width << eval_points.len());

    match eval_points.len() {
        0 => {
            scale_and_add::<_, _, INITIALIZED>(out, eq_evals.as_slice(), scalar);
        }
        1 => {
            let eq_evaluations = eval_eq_1(eval_points, eq_evals);

            scale_and_add::<_, _, INITIALIZED>(
                out,
                F::Packing::unpack_slice(&eq_evaluations),
                scalar,
            );
        }
        2 => {
            let eq_evaluations = eval_eq_2(eval_points, eq_evals);

            scale_and_add::<_, _, INITIALIZED>(
                out,
                F::Packing::unpack_slice(&eq_evaluations),
                scalar,
            );
        }
        3 => {
            let eq_evaluations = eval_eq_3(eval_points, eq_evals);

            scale_and_add::<_, _, INITIALIZED>(
                out,
                F::Packing::unpack_slice(&eq_evaluations),
                scalar,
            );
        }
        _ => {
            let (&x, tail) = eval_points.split_first().unwrap();

            // Divide the output buffer into two halves: one for `X_i = 0` and one for `X_i = 1`
            let (low, high) = out.split_at_mut(out.len() / 2);

            // Compute weight updates for the two branches:
            // - `s0` corresponds to the case when `X_i = 0`
            // - `s1` corresponds to the case when `X_i = 1`
            //
            // Mathematically, this follows the recurrence:
            // ```text
            // eq_{X1, ..., Xn}(X) = (1 - X_1) * eq_{X2, ..., Xn}(X) + X_1 * eq_{X2, ..., Xn}(X)
            // ```
            let s1 = eq_evals * x; // Contribution when `X_i = 1`
            let s0 = eq_evals - s1; // Contribution when `X_i = 0`

            // The recursive approach turns out to be faster than the iterative one here.
            // Probably related to nice cache locality.
            base_eval_eq_packed::<_, _, INITIALIZED>(tail, low, s0, scalar);
            base_eval_eq_packed::<_, _, INITIALIZED>(tail, high, s1, scalar);
        }
    }
}

/// Adds or sets the equality polynomial evaluations in the output buffer.
///
/// If the output buffer is already initialized, it adds the evaluations otherwise
/// it copies the evaluations into the buffer directly.
#[inline]
fn add_or_set<F: Field, const INITIALIZED: bool>(out: &mut [F], evaluations: &[F]) {
    debug_assert_eq!(out.len(), evaluations.len());
    if INITIALIZED {
        F::add_slices(out, evaluations);
    } else {
        out.copy_from_slice(evaluations);
    }
}

/// Scales the evaluations by scalar and either adds the result to the output buffer or
/// sets the output buffer directly depending on the `INITIALIZED` flag.
///
/// If the output buffer is already initialized, it adds the evaluations otherwise
/// it copies the evaluations into the buffer directly.
#[inline]
fn scale_and_add<F: Field, EF: ExtensionField<F>, const INITIALIZED: bool>(
    out: &mut [EF],
    base_vals: &[F],
    scalar: EF,
) {
    // TODO: We can probably add a custom method to Plonky3 to handle this more efficiently (and use packings).
    // This approach is faster than collecting `scalar * eq_eval` into a vector and using `add_slices`. Presumably
    // this is because we avoid the allocation.
    if INITIALIZED {
        out.iter_mut().zip(base_vals).for_each(|(out, &eq_eval)| {
            *out += scalar * eq_eval;
        });
    } else {
        out.iter_mut().zip(base_vals).for_each(|(out, &eq_eval)| {
            *out = scalar * eq_eval;
        });
    }
}

/// Computes equality polynomial evaluations and packs them into a `PackedFieldExtension`.
///
/// Note that when `F = EF` is a PrimeField, `EF::ExtensionPacking = F::Packing` so this can
/// also be used to compute initial packed evaluations of the equality polynomial over base
/// field elements (instead of extension field elements).
///
/// The length of `eval` must be equal to the `log2` of `F::Packing::WIDTH`.
#[allow(clippy::inline_always)] // Adding inline(always) seems to give a small performance boost.
#[inline(always)]
fn packed_eq_poly<F: Field, EF: ExtensionField<F>>(
    eval: &[EF],
    scalar: EF,
) -> EF::ExtensionPacking {
    // As this function is only available in this file, debug_assert should be fine here.
    // If this function becomes public, this should be changed to an assert.
    debug_assert_eq!(F::Packing::WIDTH, 1 << eval.len());

    // We build up the evaluations of the equality polynomial in buffer.
    let mut buffer = EF::zero_vec(1 << eval.len());
    buffer[0] = scalar;

    fill_buffer(eval.iter().rev(), &mut buffer);

    // Finally we need to do a "transpose" to get a `PackedFieldExtension` element.
    EF::ExtensionPacking::from_ext_slice(&buffer)
}

pub fn parallel_clone<A: Clone + Send + Sync>(src: &[A], dst: &mut [A]) {
    if src.len() < 1 << 15 {
        // sequential copy
        dst.clone_from_slice(src);
    } else {
        assert_eq!(src.len(), dst.len());
        let chunk_size = src.len() / rayon::current_num_threads().max(1);
        dst.par_chunks_mut(chunk_size)
            .zip(src.par_chunks(chunk_size))
            .for_each(|(d, s)| {
                d.clone_from_slice(s);
            });
    }
}

#[must_use]
pub fn parallel_clone_vec<A: Clone + Send + Sync>(vec: &[A]) -> Vec<A> {
    let mut res = unsafe { uninitialized_vec(vec.len()) };
    parallel_clone(vec, &mut res);
    res
}

pub fn parallel_inner_repeat<A: Copy + Send + Sync>(src: &[A], n: usize) -> Vec<A> {
    if src.len() * n <= 1 << 12 {
        // sequential repeat
        src.iter()
            .flat_map(|&v| std::iter::repeat(v).take(n))
            .collect()
    } else {
        let res = unsafe { uninitialized_vec::<A>(src.len() * n) };
        src.par_iter().enumerate().for_each(|(i, &v)| {
            for j in 0..n {
                unsafe {
                    std::ptr::write(res.as_ptr().cast_mut().add(i * n + j), v);
                }
            }
        });
        res
    }
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

/// Recursively computes a chunk of the scaled multilinear equality polynomial over the Boolean hypercube.
///
/// Given an evaluation point $z ∈ 𝔽^n$ and a scalar multiplier $α ∈ 𝔽$, this function computes the values:
///
/// \begin{equation}
/// [α ⋅ eq(x, z)]_{x ∈ \{0,1\}^n}
/// \end{equation}
///
/// for all Boolean inputs $x$ within a **specific chunk** of the full hypercube, defined by `start_index` and `out.len()`.
///
/// The multilinear equality polynomial is defined as:
///
/// \begin{equation}
/// eq(x, z) = \prod_{i=0}^{n-1} \left(x_i z_i + (1 - x_i)(1 - z_i)\right)
/// \end{equation}
///
/// This recursive function updates a sub-slice `out` of a larger buffer with the correct scaled evaluations.
///
/// # Arguments
/// - `eval`: The evaluation point $z = (z_0, z_1, \dots, z_{n-1}) ∈ 𝔽^n$
/// - `out`: The mutable slice of the result buffer, containing a contiguous chunk of the Boolean hypercube
/// - `scalar`: The scalar multiplier $α ∈ 𝔽$
/// - `start_index`: The global starting index of `out` within the full $2^n$-sized hypercube
///
/// # Behavior
/// For a given chunk of the output buffer, this function computes:
///
/// \begin{equation}
/// out[i] += α ⋅ eq(x, z)
/// \end{equation}
///
/// where $x$ is the Boolean vector corresponding to index $i + \text{start_index}$ in lexicographic order.
///
/// # Recursive structure
/// - At each level of recursion, the function considers one variable $z_i$
/// - It determines whether the current chunk lies entirely in the $x_i = 0$ subcube, $x_i = 1$ subcube, or spans both
/// - It updates the `scalar` for each branch accordingly:
///
///   \begin{align}
///   s_1 &= α ⋅ z_i (for x_i = 1) \\
///   s_0 &= α ⋅ (1 - z_i) = α - s_1 (for x_i = 0 )
///   \end{align}
///
/// - It then recurses on the appropriate part(s) of `out`
pub(crate) fn eval_eq_chunked<F>(eval: &[F], out: &mut [F], scalar: F, start_index: usize)
where
    F: Field,
{
    // Early exit: Nothing to process if the output chunk is empty
    if out.is_empty() {
        return;
    }

    // Base case: all variables consumed → we’re at a leaf node of the recursion tree
    // Every point in the current chunk gets incremented by the scalar α
    if eval.is_empty() {
        for v in out.iter_mut() {
            *v += scalar;
        }
        return;
    }

    // --- Recursive step begins here ---

    // Split the input: extract current variable z_0 and the tail z_1..z_{n-1}
    let (&z_i, tail) = eval.split_first().unwrap();

    // The number of remaining variables after removing z_i
    let remaining_vars = tail.len();

    // The midpoint divides the current cube into two equal parts:
    //   - Lower half: x_i = 0 (indices 0..half)
    //   - Upper half: x_i = 1 (indices half..2^remaining_vars)
    let half = 1 << remaining_vars;

    // Compute branch scalars:
    //   - s1: contribution from x_i = 1
    //   - s0: contribution from x_i = 0
    //
    // These correspond to:
    //   s1 = α ⋅ z_i
    //   s0 = α ⋅ (1 - z_i) = α - s1
    let s1 = scalar * z_i;
    let s0 = scalar - s1;

    // Decide whether the current chunk falls entirely in one half or spans both halves

    if start_index + out.len() <= half {
        // Case 1: Entire chunk lies in the lower half (x_i = 0)
        // We recurse only into the s0 (x_i = 0) branch
        eval_eq_chunked(tail, out, s0, start_index);
    } else if start_index >= half {
        // Case 2: Entire chunk lies in the upper half (x_i = 1)
        // We recurse only into the s1 (x_i = 1) branch
        // We subtract `half` to make the index relative to the upper subcube
        eval_eq_chunked(tail, out, s1, start_index - half);
    } else {
        // Case 3: The chunk spans both subcubes
        // We split it at the midpoint and recurse into both branches

        // Number of elements in the lower half of the chunk
        let mid_point = half - start_index;

        // Split `out` into chunks for the x_i = 0 and x_i = 1 subcubes
        let (low_chunk, high_chunk) = out.split_at_mut(mid_point);

        // Optional parallelism for deep recursion trees
        {
            const PARALLEL_THRESHOLD: usize = 10;

            if remaining_vars > PARALLEL_THRESHOLD {
                rayon::join(
                    || eval_eq_chunked(tail, low_chunk, s0, start_index),
                    || eval_eq_chunked(tail, high_chunk, s1, 0),
                );
                return;
            }
        }

        // Sequential fallback: recurse on both branches
        // x_i = 0 part
        eval_eq_chunked(tail, low_chunk, s0, start_index);
        // x_i = 1 part (new subproblem starts at 0)
        eval_eq_chunked(tail, high_chunk, s1, 0);
    }
}

pub fn flatten_scalars_to_base<F: Field, EF: ExtensionField<F>>(scalars: &[EF]) -> Vec<F> {
    scalars
        .iter()
        .flat_map(BasedVectorSpace::as_basis_coefficients_slice)
        .copied()
        .collect()
}

pub fn pack_scalars_to_extension<F: Field, EF: ExtensionField<F>>(scalars: &[F]) -> Vec<EF> {
    let extension_size = <EF as BasedVectorSpace<F>>::DIMENSION;
    assert!(
        scalars.len() % extension_size == 0,
        "Scalars length must be a multiple of the extension size"
    );
    scalars
        .chunks_exact(extension_size)
        .map(|chunk| EF::from_basis_coefficients_slice(chunk).unwrap())
        .collect()
}

/// Returns a vector of uninitialized elements of type `A` with the specified length.
/// # Safety
/// Entries should be overwritten before use.
#[must_use]
pub unsafe fn uninitialized_vec<A>(len: usize) -> Vec<A> {
    #[allow(clippy::uninit_vec)]
    unsafe {
        let mut vec = Vec::with_capacity(len);
        vec.set_len(len);
        vec
    }
}

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

#[cfg(test)]
mod tests {
    use super::*;
    type F = p3_koala_bear::KoalaBear;

    #[test]
    fn test() {
        let eval = vec![
            F::ZERO,
            F::ONE,
            F::ONE,
            F::ZERO,
            F::new(96),
            F::new(85),
            F::new(1),
            F::new(854),
            F::new(2),
            F::ONE,
            F::ZERO,
            F::ONE,
            F::ONE,
            F::ONE,
            F::ZERO,
        ];
        let scalar = F::new(789);
        let mut out_structured = F::zero_vec(1 << eval.len());
        let mut out_unstructured = F::zero_vec(1 << eval.len());
        compute_sparse_eval_eq(&eval, &mut out_structured, scalar);
        compute_eval_eq::<F, F, true>(&eval, &mut out_unstructured, scalar);
        assert_eq!(out_structured, out_unstructured);
    }
}
