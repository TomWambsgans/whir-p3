use std::ops::Deref;

use p3_field::{ExtensionField, Field};
use rayon::prelude::*;
use tracing::instrument;

use super::multilinear::MultilinearPoint;
use crate::utils::{eval_eq, parallel_clone, uninitialized_vec};

/// Represents a multilinear polynomial `f` in `num_variables` unknowns, stored via its evaluations
/// over the hypercube `{0,1}^{num_variables}`.
///
/// The vector `evals` contains function evaluations at **lexicographically ordered** points.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct EvaluationsList<F> {
    /// Stores evaluations in **lexicographic order**.
    evals: Vec<F>,
    /// Number of variables in the multilinear polynomial.
    /// Ensures `evals.len() = 2^{num_variables}`.
    num_variables: usize,
}

impl<F> EvaluationsList<F>
where
    F: Field,
{
    /// Constructs an `EvaluationsList` from a given vector of evaluations.
    ///
    /// - The `evals` vector must have a **length that is a power of two** since it represents
    ///   evaluations over an `n`-dimensional binary hypercube.
    /// - The ordering of evaluation points follows **lexicographic order**.
    ///
    /// **Mathematical Constraint:**
    /// If `evals.len() = 2^n`, then `num_variables = n`, ensuring correct indexing.
    ///
    /// **Panics:**
    /// - If `evals.len()` is **not** a power of two.
    #[must_use]
    pub fn new(evals: Vec<F>) -> Self {
        let len = evals.len();
        assert!(
            len.is_power_of_two(),
            "Evaluation list length must be a power of two."
        );

        Self {
            evals,
            num_variables: len.ilog2() as usize,
        }
    }

    /// Given `evals` = (α_1, ..., α_n), returns a multilinear polynomial P in n variables,
    /// defined on the boolean hypercube by: ∀ (x_1, ..., x_n) ∈ {0, 1}^n,
    /// P(x_1, ..., x_n) = Π_{i=1}^{n} (x_i.α_i + (1 - x_i).(1 - α_i))
    /// (often denoted as P(x) = eq(x, evals))
    pub fn eval_eq(eval: &[F]) -> Self {
        // Alloc memory without initializing it to zero.
        // This is safe because we overwrite it inside `eval_eq`.
        let mut out: Vec<F> = Vec::with_capacity(1 << eval.len());
        #[allow(clippy::uninit_vec)]
        unsafe {
            out.set_len(1 << eval.len());
        }
        eval_eq::<_, _, false>(eval, &mut out, F::ONE);
        Self {
            evals: out,
            num_variables: eval.len(),
        }
    }

    /// Returns an immutable reference to the evaluations vector.
    #[must_use]
    pub fn evals(&self) -> &[F] {
        &self.evals
    }

    #[must_use]
    pub fn into_evals(self) -> Vec<F> {
        self.evals
    }

    /// Returns a mutable reference to the evaluations vector.
    pub fn evals_mut(&mut self) -> &mut [F] {
        &mut self.evals
    }

    /// Returns the total number of stored evaluations.
    ///
    /// Mathematical Invariant:
    /// ```ignore
    /// num_evals = 2^{num_variables}
    /// ```
    #[must_use]
    pub fn num_evals(&self) -> usize {
        self.evals.len()
    }

    /// Returns the number of variables in the multilinear polynomial.
    #[must_use]
    pub const fn num_variables(&self) -> usize {
        self.num_variables
    }

    /// Truncates the list of evaluations to a new length.
    ///
    /// This is used in protocols like sumcheck where the number of evaluations is
    /// halved in each round. The new length must be a power of two.
    ///
    /// # Panics
    /// Panics if `new_len` is not a power of two.
    pub fn truncate(&mut self, new_len: usize) {
        assert!(
            new_len.is_power_of_two(),
            "New evaluation list length must be a power of two."
        );
        self.evals.truncate(new_len);
        self.num_variables = if new_len == 0 {
            0
        } else {
            new_len.ilog2() as usize
        };
    }

    /// Evaluates the multilinear polynomial at `point ∈ [0,1]^n`.
    ///
    /// - If `point ∈ {0,1}^n`, returns the precomputed evaluation `f(point)`.
    /// - Otherwise, computes `f(point) = ∑_{x ∈ {0,1}^n} eq(x, point) * f(x)`, where `eq(x, point)
    ///   = ∏_{i=1}^{n} (1 - p_i + 2 p_i x_i)`.
    /// - Uses fast multilinear interpolation for efficiency.
    #[must_use]
    pub fn evaluate<EF>(&self, point: &MultilinearPoint<EF>) -> EF
    where
        EF: ExtensionField<F>,
    {
        if let Some(point) = point.to_hypercube() {
            return self.evals[point.0].into();
        }
        eval_multilinear(&self.evals, point)
    }

    /// Evaluates the polynomial as a constant.
    /// This is only valid for constant polynomials (i.e., when `num_variables` is 0).
    ///
    /// # Panics
    /// Panics if `num_variables` is not 0.
    #[must_use]
    pub fn as_constant(&self) -> F {
        assert_eq!(
            self.num_variables, 0,
            "`as_constant` is only valid for constant polynomials."
        );
        self.evals[0]
    }

    /// Folds a multilinear polynomial stored in evaluation form along the last `k` variables.
    ///
    /// Given evaluations `f: {0,1}^n → F`, this method returns a new evaluation list `g` such that:
    ///
    /// \[
    /// g(x_0, ..., x_{n-k-1}) = f(x_0, ..., x_{n-k-1}, r_0, ..., r_{k-1})
    /// \]
    ///
    /// where `folding_randomness = (r_0, ..., r_{k-1})` is a fixed assignment to the last `k` variables.
    ///
    /// This operation reduces the dimensionality of the polynomial:
    ///
    /// - Input: `f ∈ F^{2^n}`
    /// - Output: `g ∈ EF^{2^{n-k}}`, where `EF` is an extension field of `F`
    ///
    /// # Arguments
    /// - `folding_randomness`: The extension-field values to substitute for the last `k` variables.
    ///
    /// # Returns
    /// - A new `EvaluationsList<EF>` representing the folded function over the remaining `n - k` variables.
    ///
    /// # Panics
    /// - If the evaluation list is not sized `2^n` for some `n`.
    #[instrument(skip_all)]
    #[must_use]
    pub fn fold<EF>(&self, folding_randomness: &MultilinearPoint<EF>) -> EvaluationsList<EF>
    where
        EF: ExtensionField<F>,
    {
        let folding_factor = folding_randomness.num_variables();
        let evals = self
            .evals
            .par_chunks_exact(1 << folding_factor)
            .map(|ev| eval_multilinear(ev, folding_randomness))
            .collect();

        EvaluationsList {
            evals,
            num_variables: self.num_variables() - folding_factor,
        }
    }

    #[must_use]
    #[instrument(skip_all)]
    pub fn parallel_clone(&self) -> Self {
        let mut evals = unsafe { uninitialized_vec(self.evals.len()) };
        parallel_clone(&self.evals, &mut evals);
        Self {
            evals,
            num_variables: self.num_variables,
        }
    }

    /// Multiply the polynomial by a scalar factor.
    #[must_use]
    pub fn scale<EF: ExtensionField<F>>(&self, factor: EF) -> EvaluationsList<EF> {
        let evals = self.evals.par_iter().map(|&e| factor * e).collect();
        EvaluationsList {
            evals,
            num_variables: self.num_variables(),
        }
    }
}

impl<F> Deref for EvaluationsList<F> {
    type Target = [F];

    fn deref(&self) -> &Self::Target {
        &self.evals
    }
}

/// Evaluates a multilinear polynomial at `point ∈ [0,1]^n` using fast interpolation.
///
/// - Given evaluations `evals` over `{0,1}^n`, computes `f(point)` via iterative interpolation.
/// - Uses the recurrence: `f(x_1, ..., x_n) = (1 - x_1) f_0 + x_1 f_1`, reducing dimension at each
///   step.
/// - Ensures `evals.len() = 2^n` to match the number of variables.
fn eval_multilinear<F, EF>(evals: &[F], point: &[EF]) -> EF
where
    F: Field,
    EF: ExtensionField<F>,
{
    debug_assert_eq!(evals.len(), 1 << point.len());
    match point {
        [] => evals[0].into(),
        [x] => *x * (evals[1] - evals[0]) + evals[0],
        [x0, x1] => {
            let a0 = *x1 * (evals[1] - evals[0]) + evals[0];
            let a1 = *x1 * (evals[3] - evals[2]) + evals[2];
            a0 + (a1 - a0) * *x0
        }
        [x0, x1, x2] => {
            let a00 = *x2 * (evals[1] - evals[0]) + evals[0];
            let a01 = *x2 * (evals[3] - evals[2]) + evals[2];
            let a10 = *x2 * (evals[5] - evals[4]) + evals[4];
            let a11 = *x2 * (evals[7] - evals[6]) + evals[6];
            let a0 = a00 + *x1 * (a01 - a00);
            let a1 = a10 + *x1 * (a11 - a10);
            a0 + (a1 - a0) * *x0
        }
        [x0, x1, x2, x3] => {
            let a000 = *x3 * (evals[1] - evals[0]) + evals[0];
            let a001 = *x3 * (evals[3] - evals[2]) + evals[2];
            let a010 = *x3 * (evals[5] - evals[4]) + evals[4];
            let a011 = *x3 * (evals[7] - evals[6]) + evals[6];
            let a100 = *x3 * (evals[9] - evals[8]) + evals[8];
            let a101 = *x3 * (evals[11] - evals[10]) + evals[10];
            let a110 = *x3 * (evals[13] - evals[12]) + evals[12];
            let a111 = *x3 * (evals[15] - evals[14]) + evals[14];
            let a00 = a000 + *x2 * (a001 - a000);
            let a01 = a010 + *x2 * (a011 - a010);
            let a10 = a100 + *x2 * (a101 - a100);
            let a11 = a110 + *x2 * (a111 - a110);
            let a0 = a00 + *x1 * (a01 - a00);
            let a1 = a10 + *x1 * (a11 - a10);
            a0 + (a1 - a0) * *x0
        }
        [x, tail @ ..] => {
            let (f0, f1) = evals.split_at(evals.len() / 2);
            let (f0, f1) = {
                let work_size: usize = (1 << 15) / std::mem::size_of::<F>();
                if evals.len() > work_size {
                    rayon::join(|| eval_multilinear(f0, tail), || eval_multilinear(f1, tail))
                } else {
                    (eval_multilinear(f0, tail), eval_multilinear(f1, tail))
                }
            };
            f0 + (f1 - f0) * *x
        }
    }
}
