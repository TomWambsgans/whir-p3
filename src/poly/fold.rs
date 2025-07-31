use p3_field::{ExtensionField, Field};

/// Computes the folded value of a function evaluated on a coset.
///
/// This function applies a recursive folding transformation to a given set of function
/// evaluations on a coset, progressively reducing the number of evaluations while incorporating
/// randomness and coset transformations. The folding process is performed `folding_factor` times,
/// halving the number of evaluations at each step.
///
/// Mathematical Formulation:
/// Given an initial evaluation vector:
/// \begin{equation}
/// f(x) = [f_0, f_1, ..., f_{2^m - 1}]
/// \end{equation}
///
/// Each folding step computes:
/// \begin{equation}
/// g_i = \frac{f_i + f_{i + N/2} + r \cdot (f_i - f_{i + N/2}) \cdot (o^{-1} \cdot g^{-i})}{2}
/// \end{equation}
///
/// where:
/// - \( r \) is the folding randomness
/// - \( o^{-1} \) is the inverse coset offset
/// - \( g^{-i} \) is the inverse generator raised to index \( i \)
/// - The function is recursively applied until the vector reduces to size 1.
pub fn compute_fold<F, EF>(
    answers: &[F],
    folding_randomness: &[EF],
    mut coset_offset_inv: EF,
    mut coset_gen_inv: EF,
    folding_factor: usize,
) -> EF
where
    EF: Field + ExtensionField<F>,
    F: Field,
{
    assert_eq!(
        answers.len(),
        1 << folding_factor,
        "Invalid number of answers"
    );

    if folding_factor == 0 {
        return EF::from(*answers.first().unwrap());
    }

    // We do the first folding step separately as in this step answers switches
    // from a base field vector to an extension field vector.
    let r = folding_randomness[folding_randomness.len() - 1];
    let half = answers.len() / 2;
    let (lo, hi) = answers.split_at(half);
    let mut answers: Vec<EF> = lo
        .iter()
        .zip(hi)
        .zip(coset_gen_inv.shifted_powers(r * coset_offset_inv))
        .map(|((a, b), point_inv)| {
            let left = *a + *b;
            let right = *a - *b;
            point_inv * right + left
        })
        .collect();

    coset_offset_inv = coset_offset_inv.square();
    coset_gen_inv = coset_gen_inv.square();

    for r in folding_randomness[..folding_randomness.len() - 1]
        .iter()
        .rev()
    {
        let half = answers.len() / 2;
        let (lo, hi) = answers.split_at_mut(half);
        lo.iter_mut()
            .zip(hi)
            .zip(coset_gen_inv.shifted_powers(*r * coset_offset_inv))
            .for_each(|((a, b), point_inv)| {
                let left = *a + *b;
                let right = *a - *b;
                *a = point_inv * right + left;
            });

        answers.truncate(half);
        coset_offset_inv = coset_offset_inv.square();
        coset_gen_inv = coset_gen_inv.square();
    }

    answers.first().unwrap().div_2exp_u64(folding_factor as u64)
}
