// use p3_field::{ExtensionField, Field};
// #[cfg(feature = "parallel")]
// use rayon::prelude::*;
// use tracing::instrument;

// use crate::{
//     poly::{evals::EvaluationsList, multilinear::MultilinearPoint},
//     utils::eval_eq,
// };

// /// Represents a weight function used in polynomial evaluations.
// ///
// /// A `Weights<F>` instance allows evaluating or accumulating weighted contributions
// /// to a multilinear polynomial stored in evaluation form. It supports two modes:
// ///
// /// - Evaluation mode: Represents an equality constraint at a specific `MultilinearPoint<F>`.
// /// - Linear mode: Represents a set of per-corner weights stored as `EvaluationsList<F>`.
// #[derive(Clone, Debug, Eq, PartialEq)]
// pub enum Weights<F> {
//     /// Represents a weight function that enforces equality constraints at a specific point.
//     Evaluation { point: MultilinearPoint<F> },
//     /// Represents a weight function defined as a precomputed set of evaluations.
//     Linear { weight: EvaluationsList<F> },
// }

// impl<F: Field> Weights<F> {
//     /// Constructs a weight in evaluation mode, enforcing an equality constraint at `point`.
//     ///
//     /// Given a multilinear polynomial `p(X)`, this weight evaluates:
//     ///
//     /// \begin{equation}
//     /// w(X) = eq_{z}(X)
//     /// \end{equation}
//     ///
//     /// where `eq_z(X)` is the Lagrange interpolation polynomial enforcing `X = z`.
//     #[must_use]
//     pub const fn evaluation(point: MultilinearPoint<F>) -> Self {
//         Self::Evaluation { point }
//     }

//     /// Constructs a weight in linear mode, applying a set of precomputed weights.
//     ///
//     /// This mode allows applying a function `w(X)` stored in `EvaluationsList<F>`:
//     ///
//     /// \begin{equation}
//     /// w(X) = \sum_{i} w_i \cdot X_i
//     /// \end{equation}
//     ///
//     /// where `w_i` are the predefined weight values for each corner of the hypercube.
//     #[must_use]
//     pub const fn linear(weight: EvaluationsList<F>) -> Self {
//         Self::Linear { weight }
//     }

//     /// Returns the number of variables involved in the weight function.
//     #[must_use]
//     pub fn num_variables(&self) -> usize {
//         match self {
//             Self::Evaluation { point } => point.num_variables(),
//             Self::Linear { weight } => weight.num_variables(),
//         }
//     }

//     /// Construct weights for a univariate evaluation
//     pub fn univariate(point: F, size: usize) -> Self {
//         Self::Evaluation {
//             point: MultilinearPoint::expand_from_univariate(point, size),
//         }
//     }

//     /// Computes the weighted sum of a polynomial `p(X)` under the current weight function.
//     ///
//     /// - In linear mode, computes the inner product between the polynomial values and weights:
//     ///
//     /// \begin{equation}
//     /// \sum_{i} p_i \cdot w_i
//     /// \end{equation}
//     ///
//     /// - In evaluation mode, evaluates `p(X)` at the equality constraint point.
//     ///
//     /// **Precondition:**
//     /// If `self` is in linear mode, `poly.num_variables()` must match `weight.num_variables()`.
//     #[must_use]
//     pub fn evaluate_evals<BF>(&self, poly: &EvaluationsList<BF>) -> F
//     where
//         BF: Field,
//         F: ExtensionField<BF>,
//     {
//         match self {
//             Self::Linear { weight } => {
//                 assert_eq!(poly.num_variables(), weight.num_variables());
//                 #[cfg(not(feature = "parallel"))]
//                 {
//                     poly.evals()
//                         .iter()
//                         .zip(weight.evals().iter())
//                         .map(|(p, w)| *w * *p)
//                         .sum()
//                 }
//                 #[cfg(feature = "parallel")]
//                 {
//                     poly.evals()
//                         .par_iter()
//                         .zip(weight.evals().par_iter())
//                         .map(|(p, w)| *w * *p)
//                         .sum()
//                 }
//             }
//             Self::Evaluation { point } => poly.evaluate(point),
//         }
//     }

//     /// Accumulates the contribution of the weight function into `accumulator`, scaled by `factor`.
//     ///
//     /// - In evaluation mode, updates `accumulator` using an equality constraint.
//     /// - In linear mode, scales the weight function by `factor` and accumulates it.
//     ///
//     /// Given a weight function `w(X)` and a factor `λ`, this updates `accumulator` as:
//     ///
//     /// ```math
//     /// a(X) <- a(X) + \lambda \cdot w(X)
//     /// ```
//     ///
//     /// where `a(X)` is the accumulator polynomial.
//     ///
//     /// **Precondition:**
//     /// `accumulator.num_variables()` must match `self.num_variables()`.
//     ///
//     /// **Warning:**
//     /// If INITIALIZED is `false`, the accumulator must be overwritten with the new values.
//     #[instrument(skip_all)]
//     pub fn accumulate<Base, const INITIALIZED: bool>(
//         &self,
//         accumulator: &mut EvaluationsList<F>,
//         factor: F,
//     ) where
//         Base: Field,
//         F: ExtensionField<Base>,
//     {
//         assert_eq!(accumulator.num_variables(), self.num_variables());
//         match self {
//             Self::Evaluation { point } => {
//                 eval_eq::<Base, F, INITIALIZED>(point, accumulator.evals_mut(), factor);
//             }
//             Self::Linear { weight } => {
//                 #[cfg(feature = "parallel")]
//                 let accumulator_iter = accumulator.evals_mut().par_iter_mut();
//                 #[cfg(not(feature = "parallel"))]
//                 let accumulator_iter = accumulator.evals_mut().iter_mut();

//                 accumulator_iter.enumerate().for_each(|(corner, acc)| {
//                     if INITIALIZED {
//                         *acc += factor * weight[corner];
//                     } else {
//                         *acc = factor * weight[corner];
//                     }
//                 });
//             }
//         }
//     }

//     /// Evaluates the weight function at a given folding point.
//     ///
//     /// - In evaluation mode, computes the equality polynomial at the provided point:
//     ///
//     /// \begin{equation}
//     /// w(X) = eq_{\text{point}}(X)
//     /// \end{equation}
//     ///
//     /// This enforces that the polynomial is evaluated exactly at a specific input.
//     ///
//     /// - In linear mode, interprets the stored weight function as a multilinear polynomial
//     ///   and evaluates it at the provided point using multilinear extension.
//     ///
//     /// **Precondition:**
//     /// The input point must have the same number of variables as the weight function.
//     ///
//     /// **Returns:**
//     /// A field element representing the weight function applied to the given point.
//     #[must_use]
//     pub fn compute(&self, folding_randomness: &MultilinearPoint<F>) -> F {
//         match self {
//             Self::Evaluation { point } => point.eq_poly_outside(folding_randomness),
//             Self::Linear { weight } => weight.evaluate(folding_randomness),
//         }
//     }
// }
