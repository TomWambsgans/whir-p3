use p3_challenger::{FieldChallenger, GrindingChallenger};
use p3_field::{ExtensionField, PrimeCharacteristicRing, TwoAdicField};
use p3_interpolation::interpolate_subgroup;
use p3_matrix::dense::RowMajorMatrix;

use crate::{
    fiat_shamir::{
        errors::{ProofError, ProofResult},
        verifier::VerifierState,
    },
    poly::multilinear::MultilinearPoint,
    sumcheck::{K_SKIP_SUMCHECK, sumcheck_polynomial::SumcheckPolynomial},
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
    verifier_state: &mut VerifierState<F, EF, Challenger>,
    claimed_sum: &mut EF,
    rounds: usize,
    pow_bits: usize,
    is_univariate_skip: bool,
) -> ProofResult<SumcheckRandomness<EF>>
where
    EF: ExtensionField<F>,
    F: TwoAdicField,
    EF: TwoAdicField,
    Challenger: FieldChallenger<F::PrimeSubfield> + GrindingChallenger<Witness = F::PrimeSubfield>,
    F: ExtensionField<<F as PrimeCharacteristicRing>::PrimeSubfield>,
    EF: ExtensionField<<F as PrimeCharacteristicRing>::PrimeSubfield>,
{
    // Calculate how many `(poly, rand)` pairs to expect based on skip mode
    //
    // If skipping: we do 1 large round for the skip, and the remaining normally
    let effective_rounds = if is_univariate_skip {
        1 + (rounds - K_SKIP_SUMCHECK)
    } else {
        rounds
    };

    // Preallocate vector to hold the randomness values
    let mut randomness = Vec::with_capacity(effective_rounds);

    // Handle the univariate skip case
    if is_univariate_skip {
        // Read `2^{k+1}` evaluations (size of coset domain) for the skipping polynomial
        let evals: [EF; 1 << (K_SKIP_SUMCHECK + 1)] =
            verifier_state.next_extension_scalars_const()?;

        // Interpolate into a univariate polynomial (over the coset domain)
        let poly = SumcheckPolynomial::new(evals.to_vec(), 1);

        // Sample the challenge scalar r₀ ∈ 𝔽 for this round
        let rand = verifier_state.sample();

        // Update the claimed sum using the univariate polynomial and randomness.
        //
        // We interpolate the univariate polynomial at the randomness point.
        *claimed_sum =
            interpolate_subgroup(&RowMajorMatrix::new_col(poly.evaluations().to_vec()), rand)[0];

        // Record this round’s randomness
        randomness.push(rand);

        // Optional: apply proof-of-work query
        verifier_state.check_pow_grinding(pow_bits)?;
    }

    // Continue with the remaining sumcheck rounds (each using 3 evaluations)
    let start_round = if is_univariate_skip {
        K_SKIP_SUMCHECK // skip the first k rounds
    } else {
        0
    };

    for _ in start_round..rounds {
        // Extract the 3 evaluations of the quadratic sumcheck polynomial h(X)
        let evals: [_; 3] = verifier_state.next_extension_scalars_const()?;

        let poly = SumcheckPolynomial::new(evals.to_vec(), 1);

        // Verify claimed sum is consistent with polynomial
        if poly.sum_over_boolean_hypercube() != *claimed_sum {
            return Err(ProofError::InvalidProof);
        }

        // Sample the next verifier folding randomness rᵢ
        let rand: EF = verifier_state.sample();

        // Update claimed sum using folding randomness
        *claimed_sum = poly.evaluate_at_point(&rand.into());

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