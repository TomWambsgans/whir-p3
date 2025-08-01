use std::borrow::Borrow;

use p3_field::{ExtensionField, Field};
use rayon::prelude::*;
use tracing::instrument;

use super::multilinear::MultilinearPoint;
use crate::utils::compute_eval_eq;

/// Given `evals` = (α_1, ..., α_n), returns a multilinear polynomial P in n variables,
/// defined on the boolean hypercube by: ∀ (x_1, ..., x_n) ∈ {0, 1}^n,
/// P(x_1, ..., x_n) = Π_{i=1}^{n} (x_i.α_i + (1 - x_i).(1 - α_i))
/// (often denoted as P(x) = eq(x, evals))
pub fn eval_eq<F: Field>(eval: &[F]) -> Vec<F> {
    // Alloc memory without initializing it to zero.
    // This is safe because we overwrite it inside `eval_eq`.
    let mut out: Vec<F> = Vec::with_capacity(1 << eval.len());
    #[allow(clippy::uninit_vec)]
    unsafe {
        out.set_len(1 << eval.len());
    }
    compute_eval_eq::<_, _, false>(eval, &mut out, F::ONE);
    out
}

pub trait EvaluationsList<F: Field> {
    fn num_variables(&self) -> usize;
    fn num_evals(&self) -> usize;
    fn evaluate<EF: ExtensionField<F>>(&self, point: &MultilinearPoint<EF>) -> EF;
    fn as_constant(&self) -> F;
}

impl<F: Field, EL: Borrow<[F]>> EvaluationsList<F> for EL {
    fn num_variables(&self) -> usize {
        self.borrow().len().ilog2() as usize
    }

    fn num_evals(&self) -> usize {
        self.borrow().len()
    }

    fn evaluate<EF: ExtensionField<F>>(&self, point: &MultilinearPoint<EF>) -> EF {
        eval_multilinear(self.borrow(), point)
    }

    fn as_constant(&self) -> F {
        assert_eq!(self.borrow().len(), 1);
        self.borrow()[0]
    }
}

#[instrument(skip_all)]
#[must_use]
pub fn fold_multilinear<F: Field, EF: ExtensionField<F>>(
    poly: &[F],
    folding_randomness: &MultilinearPoint<EF>,
) -> Vec<EF> {
    let folding_factor = folding_randomness.num_variables();
    poly.par_chunks_exact(1 << folding_factor)
        .map(|ev| eval_multilinear(ev, folding_randomness))
        .collect()
}

/// Multiply the polynomial by a scalar factor.
#[must_use]
pub fn scale_poly<F: Field, EF: ExtensionField<F>>(poly: &[F], factor: EF) -> Vec<EF> {
    poly.par_iter().map(|&e| factor * e).collect()
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
