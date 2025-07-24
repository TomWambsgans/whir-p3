use p3_dft::{NaiveDft, TwoAdicSubgroupDft};
use p3_field::{ExtensionField, TwoAdicField};
use p3_matrix::{Matrix, dense::RowMajorMatrix};
#[cfg(feature = "parallel")]
use rayon::iter::{IndexedParallelIterator, ParallelIterator};

use super::sumcheck_polynomial::SumcheckPolynomial;
use crate::poly::evals::EvaluationsList;

/// Computes the sumcheck polynomial using the **univariate skip** optimization,
/// which folds the first `k` variables in one step via low-degree extension (LDE).
///
/// The goal is to reduce a multilinear polynomial $f(x_1, \dots, x_n)$
/// and a weight polynomial $w(x_1, \dots, x_n)$ defined over the Boolean hypercube
/// $\{0,1\}^n$ into a univariate polynomial $h(X)$ via partial evaluation and DFT-based extension.
///
/// This function interprets the original evaluations over $\{0,1\}^n$ as a matrix of shape
/// $(2^k \times 2^{n-k})$, where:
/// - Each row corresponds to a distinct assignment to the first $k$ variables (which we skip/fold),
/// - Each column corresponds to a Boolean assignment to the remaining $n - k$ variables.
///
/// It then applies LDE to each row over a multiplicative coset $D$ of size $2^{k+1}$ and computes:
///
/// \begin{equation}
/// h(X) = \sum_{b \in \{0,1\}^{n-k}} f(X, b) \cdot w(X, b)
/// \end{equation}
///
/// where:
/// - $X$ ranges over $D$, a multiplicative coset used to evaluate the first $k$ variables,
/// - $b$ ranges over $\{0,1\}^{n-k}$, the Boolean values of the remaining variables.
///
/// # Arguments
/// - `k`: Number of initial variables to skip and fold into a univariate extension.
/// - `evals`: Evaluations of the multilinear polynomial $f$ over $\{0,1\}^n$, in the base field `F`.
/// - `weights`: Evaluations of the weight polynomial $w$ over $\{0,1\}^n$, in the extension field `EF`.
///
/// # Returns
/// A tuple containing:
/// - `SumcheckPolynomial<EF>`: The resulting univariate polynomial $h(X)$ evaluated over coset $D$.
/// - `RowMajorMatrix<F>`: The original evaluations of $f$, reshaped to $(2^k \times 2^{n-k})$.
/// - `RowMajorMatrix<EF>`: The original evaluations of $w$, reshaped to $(2^k \times 2^{n-k})$.
///
/// # Panics
/// Panics if `k > evals.num_variables()`.
///
/// # Notes
/// - This method assumes that `f` is represented using base field values (`F`)
///   and that `w` is represented using extension field values (`EF`).
/// - The LDE step extends each row from $2^k$ to $2^{k+1}$ using a coset FFT,
///   enabling efficient computation of the univariate sumcheck polynomial.
#[must_use]
pub(crate) fn compute_skipping_sumcheck_polynomial<F: TwoAdicField, EF: ExtensionField<F>>(
    k: usize,
    evals: &EvaluationsList<F>,
    weights: &EvaluationsList<EF>,
) -> (
    SumcheckPolynomial<EF>,
    RowMajorMatrix<F>,
    RowMajorMatrix<EF>,
) {
    // Ensure we have enough variables to skip.
    // We can only skip if the number of variables n ≥ k.
    assert!(
        evals.num_variables() >= k,
        "Need at least k variables to apply univariate skip on k variables"
    );

    // Main logic block that computes the univariate sumcheck polynomial h(X)
    // and returns intermediate matrices of shape (2^k × 2^{n-k}).
    let (out_vec, f, w) = {
        // Number of variables for the multilinear polynomial f(X)
        let n = evals.num_variables();
        // Number of remaining variables after skipping k
        let num_remaining_vars = n - k;
        // Number of columns = 2^{n-k}
        let width = 1 << num_remaining_vars;

        // Reshape the evaluation vector of f (over {0,1}^n) into a matrix:
        // - Each row corresponds to one of the 2^k assignments to the skipped variables.
        // - Each column corresponds to a Boolean assignment to the remaining n−k variables.
        //
        // This aligns with the goal of computing:
        //   h(X) = ∑_{b ∈ {0,1}^{n−k}} f(X, b) · w(X, b)
        let f_mat = RowMajorMatrix::new(evals.evals().to_vec(), width);

        // Do the same for the weight polynomial w(X): shape = (2^k × 2^{n-k})
        let weights_mat = RowMajorMatrix::new(weights.evals().to_vec(), width);

        // Apply a low-degree extension (LDE) to each row of f_mat and weights_mat.
        // The LDE maps each row of length 2^k to 2^{k+1} evaluations over a multiplicative coset.
        //
        // This gives us access to evaluations of f(X, b) and w(X, b)
        // for non-Boolean values of X ∈ D (coset of size 2^{k+1}).
        let dft = NaiveDft;

        // Apply base-field LDE to each row of f_mat: F^2^k → F^2^{k+1}
        let f_on_coset = dft.lde_batch(f_mat.clone(), 1).to_row_major_matrix();

        // Apply extension-field LDE to each row of weights_mat: EF^2^k → EF^2^{k+1}
        let weights_on_coset = dft
            .lde_algebra_batch(weights_mat.clone(), 1)
            .to_row_major_matrix();

        // For each column (i.e., each value X in the coset domain),
        // compute: sum over all b ∈ {0,1}^{n−k} of f(X, b) · w(X, b)
        //
        // This is done by pointwise multiplying the f and w values in each row,
        // then summing across the row. Each output corresponds to one X in the coset.
        let result: Vec<EF> = f_on_coset
            .par_row_slices()
            .zip(weights_on_coset.par_row_slices())
            .map(|(coeffs_row, weights_row)| {
                coeffs_row
                    .iter()
                    .zip(weights_row.iter())
                    .map(|(&c, &w)| w * c)
                    .sum()
            })
            .collect();

        // Return:
        // - result: evaluations of the univariate sumcheck polynomial h(X)
        // - f_mat: original (2^k × 2^{n−k}) matrix of f(X) before LDE
        // - weights_mat: original (2^k × 2^{n−k}) matrix of w(X) before LDE
        (result, f_mat, weights_mat)
    };

    // Return h(X) as a SumcheckPolynomial, along with the raw pre-LDE matrices
    (SumcheckPolynomial::new(out_vec, 1), f, w)
}
