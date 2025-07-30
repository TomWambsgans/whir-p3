use p3_field::{
    Algebra, BasedVectorSpace, ExtensionField, Field, PackedFieldExtension, PackedValue,
    PrimeCharacteristicRing,
};
use p3_util::{iter_array_chunks_padded, log2_strict_usize};
#[cfg(feature = "parallel")]
use rayon::prelude::*;

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
pub(crate) fn eval_eq<F: Field, EF: ExtensionField<F>, const INITIALIZED: bool>(
    eval: &[EF],
    out: &mut [EF],
    scalar: EF,
) {
    // Number of threads to spawn.
    // Long term this should be a modifiable parameter.
    // I've chosen 32 here as my machine has 20 logical cores. Plausibly we might want to hard code this
    // to be a little larger than this.
    #[cfg(feature = "parallel")]
    const LOG_NUM_THREADS: usize = 5;
    #[cfg(not(feature = "parallel"))]
    const LOG_NUM_THREADS: usize = 0;

    const NUM_THREADS: usize = 1 << LOG_NUM_THREADS;

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
        eval_eq_basic::<_, _, INITIALIZED>(eval, out, scalar);
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
        for (ind, entry) in eval[..LOG_NUM_THREADS].iter().rev().enumerate() {
            let stride = 1 << ind;

            for index in 0..stride {
                let val = parallel_buffer[index];
                let scaled_val = val * *entry;
                let new_val = val - scaled_val;

                parallel_buffer[index] = new_val;
                parallel_buffer[index + stride] = scaled_val;
            }
        }

        // Finally do all computations involving the middle elements in parallel.
        #[cfg(feature = "parallel")]
        out.par_chunks_exact_mut(out_chunk_size)
            .zip(parallel_buffer.par_iter())
            .for_each(|(out_chunk, buffer_val)| {
                eval_eq_packed::<_, _, INITIALIZED>(
                    &eval[LOG_NUM_THREADS..(eval.len() - log_packing_width)],
                    out_chunk,
                    *buffer_val,
                );
            });

        // Or not in parallel if the feature is unavailable.
        #[cfg(not(feature = "parallel"))]
        out.chunks_exact_mut(out_chunk_size)
            .zip(parallel_buffer.iter())
            .for_each(|(out_chunk, buffer_val)| {
                eval_eq_packed::<_, _, INITIALIZED>(
                    &eval[LOG_NUM_THREADS..(eval.len() - log_packing_width)],
                    out_chunk,
                    *buffer_val,
                );
            });
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
fn eval_eq_1<F: Field, FP: Algebra<F> + Copy>(eval: &[F], scalar: FP) -> [FP; 2] {
    // Extract the evaluation point z_0
    let z_0 = eval[0];

    // Compute α ⋅ z_0 = α ⋅ eq(1, z)
    let s1 = scalar * z_0;

    // Compute α ⋅ (1 - z_0) = α - α ⋅ z_0 = eq(0, z)
    let s0 = scalar - s1;

    // [eq(0, z), eq(1, z)]
    [s0, s1]
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
fn eval_eq_2<F: Field, FP: Algebra<F> + Copy>(eval: &[F], scalar: FP) -> [FP; 4] {
    // First variable z_0
    let z_0 = eval[0];

    // Second variable z_1
    let z_1 = eval[1];

    // Compute s1 = α ⋅ z_0 = α ⋅ eq(x_0 = 1, -)
    let s1 = scalar * z_0;

    // Compute s0 = α - s1 = α ⋅ (1 - z_0) = α ⋅ eq(x_0 = 0, -)
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
fn eval_eq_3<F: Field, FP: Algebra<F> + Copy>(eval: &[F], scalar: FP) -> [FP; 8] {
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
fn eval_eq_basic<F: Field, EF: ExtensionField<F>, const INITIALIZED: bool>(
    eval: &[EF],
    out: &mut [EF],
    scalar: EF,
) {
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

            if INITIALIZED {
                out[0] += eq_evaluations[0];
                out[1] += eq_evaluations[1];
            } else {
                out[0] = eq_evaluations[0];
                out[1] = eq_evaluations[1];
            }
        }
        2 => {
            // Manually unroll for two variable case
            let eq_evaluations = eval_eq_2(eval, scalar);

            out.iter_mut()
                .zip(eq_evaluations.iter())
                .for_each(|(out, eq_eval)| {
                    if INITIALIZED {
                        *out += *eq_eval;
                    } else {
                        *out = *eq_eval;
                    }
                });
        }
        3 => {
            // Manually unroll for three variable case
            let eq_evaluations = eval_eq_3(eval, scalar);

            out.iter_mut()
                .zip(eq_evaluations.iter())
                .for_each(|(out, eq_eval)| {
                    if INITIALIZED {
                        *out += *eq_eval;
                    } else {
                        *out = *eq_eval;
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
            eval_eq_basic::<_, _, INITIALIZED>(tail, low, s0);
            eval_eq_basic::<_, _, INITIALIZED>(tail, high, s1);
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
            if INITIALIZED {
                EF::add_slices(out, &result);
            } else {
                out.copy_from_slice(&result);
            }
        }
        1 => {
            // Manually unroll for single variable case
            let eq_evaluations = eval_eq_1(eval, scalar);

            let result: Vec<EF> = EF::ExtensionPacking::to_ext_iter(eq_evaluations).collect();
            if INITIALIZED {
                EF::add_slices(out, &result);
            } else {
                out.copy_from_slice(&result);
            }
        }
        2 => {
            // Manually unroll for two variables case
            let eq_evaluations = eval_eq_2(eval, scalar);

            let result: Vec<EF> = EF::ExtensionPacking::to_ext_iter(eq_evaluations).collect();
            if INITIALIZED {
                EF::add_slices(out, &result);
            } else {
                out.copy_from_slice(&result);
            }
        }
        3 => {
            const EVAL_LEN: usize = 8;

            // Manually unroll for single variable case
            let eq_evaluations = eval_eq_3(eval, scalar);

            // Unpack the evaluations back into EF elements and add to output.
            // We use `iter_array_chunks_padded` to allow us to use `add_slices` without
            // needing a vector allocation. Note that `eq_evaluations: [EF::ExtensionPacking: 8]`
            // so we know that `out.len() = 8 * F::Packing::WIDTH` meaning we can use `chunks_exact_mut`
            // and `iter_array_chunks_padded` will never actually pad anything.
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

/// Computes equality polynomial evaluations and packs them into a `PackedFieldExtension`.
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

    for (ind, entry) in eval.iter().rev().enumerate() {
        // After round 1 buffer should look like:
        // [(1 - xn)*scalar, xn*scalar, 0, 0, ...]
        // After round 2 buffer should look like:
        // [(1 - xn)*(1 - x_{n-1})*scalar, xn*(1 - x_{n-1})*scalar,
        //      (1 - xn)*x_{n-1}*scalar, xn*x_{n-1}*scalar, 0, 0, ...]
        let stride = 1 << ind;
        for index in 0..stride {
            let val = buffer[index];
            let scaled_val = val * *entry;
            let new_val = val - scaled_val;

            buffer[index] = new_val;
            buffer[index + stride] = scaled_val;
        }
    }

    // Finally we need to do a "transpose" to get a `PackedFieldExtension` element.
    EF::ExtensionPacking::from_ext_slice(&buffer)
}

pub fn parallel_clone<A: Clone + Send + Sync>(src: &[A], dst: &mut [A]) {
    #[cfg(feature = "parallel")]
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

    #[cfg(not(feature = "parallel"))]
    dst.clone_from_slice(src);
}

pub fn parallel_repeat<A: Copy + Send + Sync>(src: &[A], n: usize) -> Vec<A> {
    #[cfg(feature = "parallel")]
    if src.len() * n < 1 << 15 {
        // sequential repeat
        src.repeat(n)
    } else {
        let res = unsafe { uninitialized_vec::<A>(src.len() * n) };
        src.par_iter().enumerate().for_each(|(i, &v)| {
            for j in 0..n {
                unsafe {
                    std::ptr::write(res.as_ptr().cast_mut().add(i + j * src.len()), v);
                }
            }
        });
        res
    }

    #[cfg(not(feature = "parallel"))]
    src.repeat(n)
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
        #[cfg(feature = "parallel")]
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