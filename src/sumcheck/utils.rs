use p3_field::{ExtensionField, Field};

/// Computes the partial contributions to the sumcheck polynomial from two evaluations.
///
/// Given two evaluations of a function and two evaluations of a weight:
/// - \( p(0), p(1) \) and \( w(0), w(1) \)
///
/// this function:
/// - Models \( p(x) = p(0) + (p(1) - p(0)) \cdot x \)
/// - Models \( w(x) = w(0) + (w(1) - w(0)) \cdot x \)
/// - Computes the contributions to:
///
/// \[
/// p(x) \cdot w(x) = \text{const term} + \text{linear term} \cdot x + \text{quadratic term} \cdot x^2
/// \]
///
/// Returns:
/// - The **constant coefficient** (\( p(0) \cdot w(0) \))
/// - The **quadratic coefficient** (\( (p(1) - p(0)) \cdot (w(1) - w(0)) \))
///
/// Note: the linear coefficient is reconstructed globally later.
#[inline]
pub(crate) fn sumcheck_quadratic<F, EF>((p, eq): (&[F], &[EF])) -> (EF, EF)
where
    F: Field,
    EF: ExtensionField<F>,
{
    // Compute the constant coefficient:
    // p(0) * w(0)
    let constant = eq[0] * p[0];

    // Compute the quadratic coefficient:
    // (p(1) - p(0)) * (w(1) - w(0))
    let quadratic = (eq[1] - eq[0]) * (p[1] - p[0]);

    (constant, quadratic)
}
