use p3_challenger::{FieldChallenger, GrindingChallenger};
use p3_field::{ExtensionField, TwoAdicField};

use crate::{
    PF,
    fiat_shamir::{
        errors::{ProofError, ProofResult},
        verifier::VerifierState,
    },
    poly::{dense::WhirDensePolynomial, multilinear::MultilinearPoint},
};

/// The full vector of folding randomness values, in reverse round order.
type SumcheckRandomness<F> = MultilinearPoint<F>;

/// Extracts a sequence of `(SumcheckPolynomial, folding_randomness)` pairs from the verifier transcript,
/// and computes the corresponding `MultilinearPoint` folding randomness in reverse order.
///
/// This function reads from the Fiat–Shamir transcript to simulate verifier interaction
/// in the sumcheck protocol. For each round, it recovers:
/// - One univariate polynomial (usually degree ≤ 2) sent by the prover.
/// - One challenge scalar chosen by the verifier (folding randomness).
///
/// ## Modes
///
/// - **Standard mode** (`is_univariate_skip = false`):
///   Each round represents a single variable being folded.
///   The polynomial is evaluated at 3 points, typically `{0, 1, r}` for quadratic reduction.
///
/// - **Univariate skip mode** (`is_univariate_skip = true`):
///   The first `K_SKIP_SUMCHECK` variables are folded simultaneously by evaluating a single univariate polynomial
///   over a coset of size `2^{k+1}`. This yields a larger polynomial but avoids several later rounds.
///
/// # Arguments
///
/// - `verifier_state`: The verifier's Fiat–Shamir transcript state.
/// - `rounds`: Total number of variables being folded.
/// - `pow_bits`: Optional proof-of-work difficulty (0 disables PoW).
/// - `is_univariate_skip`: If true, apply the univariate skip optimization on the first `K_SKIP_SUMCHECK` variables.
///
/// # Returns
///
/// - A `MultilinearPoint` of folding randomness values in reverse order.
pub(crate) fn verify_sumcheck_rounds<EF, F, Challenger>(
    verifier_state: &mut VerifierState<PF<F>, EF, Challenger>,
    claimed_sum: &mut EF,
    rounds: usize,
    pow_bits: usize,
) -> ProofResult<SumcheckRandomness<EF>>
where
    F: TwoAdicField,
    EF: ExtensionField<F> + TwoAdicField + ExtensionField<PF<F>>,
    Challenger: FieldChallenger<PF<F>> + GrindingChallenger<Witness = PF<F>>,
{
    // Preallocate vector to hold the randomness values
    let mut randomness = Vec::with_capacity(rounds);

    for _ in 0..rounds {
        // Extract the 3 evaluations of the quadratic sumcheck polynomial h(X)
        let coeffs: [_; 3] = verifier_state.next_extension_scalars_const()?;

        let poly = WhirDensePolynomial::from_coefficients_vec(coeffs.to_vec());

        // Verify claimed sum is consistent with polynomial
        if poly.evaluate(EF::ZERO) + poly.evaluate(EF::ONE) != *claimed_sum {
            return Err(ProofError::InvalidProof);
        }

        // Sample the next verifier folding randomness rᵢ
        let rand: EF = verifier_state.sample();

        // Update claimed sum using folding randomness
        *claimed_sum = poly.evaluate(rand);

        // Store this round’s randomness
        randomness.push(rand);

        // Optional PoW interaction (grinding resistance)
        verifier_state.check_pow_grinding(pow_bits)?;
    }

    // We should reverse the order of the randomness points:
    // This is because the randomness points are originally reverted at the end of the sumcheck rounds.
    randomness.reverse();

    Ok(MultilinearPoint(randomness))
}
