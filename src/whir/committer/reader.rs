use std::{fmt::Debug, ops::Deref};

use p3_challenger::{FieldChallenger, GrindingChallenger};
use p3_field::{ExtensionField, Field, PrimeCharacteristicRing, TwoAdicField};
use p3_symmetric::Hash;

use crate::{
    PF,
    fiat_shamir::{errors::ProofResult, verifier::VerifierState},
    whir::{
        parameters::WhirConfig,
        statement::{constraint::Constraint, weights::Weights},
    },
};

/// Represents a parsed commitment from the prover in the WHIR protocol.
///
/// This includes the Merkle root of the committed table and any out-of-domain (OOD)
/// query points and their corresponding answers, which are required for verifier checks.
#[derive(Debug, Clone)]
pub struct ParsedCommitment<F: PrimeCharacteristicRing, EF, const DIGEST_ELEMS: usize> {
    /// Number of variables in the committed polynomial.
    pub num_variables: usize,

    /// Merkle root of the committed evaluation table.
    ///
    /// This hash is used by the verifier to check Merkle proofs of queried evaluations.
    pub root: Hash<PF<F>, PF<F>, DIGEST_ELEMS>,

    /// Points queried by the verifier outside the low-degree evaluation domain.
    ///
    /// These are chosen using Fiat-Shamir and used to test polynomial consistency.
    pub ood_points: Vec<EF>,

    /// Answers (evaluations) of the committed polynomial at the corresponding `ood_points`.
    pub ood_answers: Vec<EF>,
}

impl<F: Field, EF: ExtensionField<F>, const DIGEST_ELEMS: usize>
    ParsedCommitment<F, EF, DIGEST_ELEMS>
{
    /// Parse a commitment from the verifier's transcript state.
    ///
    /// This function extracts a `ParsedCommitment` by reading the Merkle root,
    /// out-of-domain (OOD) challenge points, and corresponding claimed evaluations
    /// from the verifier's Fiat-Shamir transcript.
    ///
    /// # Arguments
    ///
    /// - `verifier_state`: The verifier's Fiat-Shamir state from which data is read.
    /// - `num_variables`: Number of variables in the committed multilinear polynomial.
    /// - `ood_samples`: Number of out-of-domain points the verifier expects to query.
    ///
    /// # Returns
    ///
    /// A [`ParsedCommitment`] containing:
    /// - Number of variables in the committed multilinear polynomial
    /// - The Merkle root of the committed table,
    /// - The OOD challenge points,
    /// - The prover's claimed answers at those points.
    ///
    /// This is used to verify consistency of polynomial commitments in WHIR.
    pub fn parse<Challenger>(
        verifier_state: &mut VerifierState<PF<F>, EF, Challenger>,
        num_variables: usize,
        ood_samples: usize,
    ) -> ProofResult<ParsedCommitment<F, EF, DIGEST_ELEMS>>
    where
        F: TwoAdicField,
        EF: ExtensionField<F> + TwoAdicField,
        Challenger: FieldChallenger<PF<F>> + GrindingChallenger<Witness = PF<F>>,
        EF: ExtensionField<PF<F>>,
    {
        // Read the Merkle root hash committed by the prover.
        let root = verifier_state
            .next_base_scalars_const::<DIGEST_ELEMS>()?
            .into();

        // Allocate space for the OOD challenge points and answers.
        let mut ood_points = EF::zero_vec(ood_samples);

        // If there are any OOD samples expected, read them from the transcript.
        let ood_answers = if ood_samples > 0 {
            // Read challenge points chosen by Fiat-Shamir.
            for ood_point in &mut ood_points {
                *ood_point = verifier_state.sample();
            }

            verifier_state.next_extension_scalars_vec(ood_samples)?
        } else {
            Vec::new()
        };

        // Return a structured representation of the commitment.
        Ok(ParsedCommitment {
            num_variables,
            root,
            ood_points,
            ood_answers,
        })
    }

    /// Construct equality constraints for all out-of-domain (OOD) samples.
    ///
    /// Each constraint enforces that the committed polynomial evaluates to the
    /// claimed `ood_answer` at the corresponding `ood_point`, using a univariate
    /// equality weight over `num_variables` inputs.
    pub fn oods_constraints(&self) -> Vec<Constraint<EF>> {
        self.ood_points
            .iter()
            .zip(&self.ood_answers)
            .map(|(&point, &eval)| Constraint {
                weights: Weights::univariate(point, self.num_variables),
                sum: eval,
            })
            .collect()
    }
}

/// Helper for parsing commitment data during verification.
///
/// The `CommitmentReader` wraps the WHIR configuration and provides a convenient
/// method to extract a `ParsedCommitment` by reading values from the Fiat-Shamir transcript.
#[derive(Debug)]
pub struct CommitmentReader<'a, EF, F, H, C, Challenger>(
    /// Reference to the verifier’s configuration object.
    ///
    /// This contains all parameters needed to parse the commitment,
    /// including how many out-of-domain samples are expected.
    &'a WhirConfig<EF, F, H, C, Challenger>,
)
where
    F: Field,
    EF: ExtensionField<F>;

impl<'a, EF, F, H, C, Challenger> CommitmentReader<'a, EF, F, H, C, Challenger>
where
    F: TwoAdicField,
    EF: ExtensionField<F> + TwoAdicField + ExtensionField<PF<F>>,
    Challenger: FieldChallenger<PF<F>> + GrindingChallenger<Witness = PF<F>>,
{
    /// Create a new commitment reader from a WHIR configuration.
    ///
    /// This allows the verifier to parse a commitment from the Fiat-Shamir transcript.
    pub const fn new(params: &'a WhirConfig<EF, F, H, C, Challenger>) -> Self {
        Self(params)
    }

    /// Parse a commitment from the verifier's transcript state.
    ///
    /// Reads the Merkle root and out-of-domain (OOD) challenge points and answers
    /// expected for verifying the committed polynomial.
    pub fn parse_commitment<const DIGEST_ELEMS: usize>(
        &self,
        verifier_state: &mut VerifierState<PF<F>, EF, Challenger>,
    ) -> ProofResult<ParsedCommitment<F, EF, DIGEST_ELEMS>> {
        ParsedCommitment::<F, EF, DIGEST_ELEMS>::parse(
            verifier_state,
            self.mv_parameters.num_variables,
            self.committment_ood_samples,
        )
    }
}

impl<EF, F, H, C, Challenger> Deref for CommitmentReader<'_, EF, F, H, C, Challenger>
where
    F: Field,
    EF: ExtensionField<F>,
{
    type Target = WhirConfig<EF, F, H, C, Challenger>;

    fn deref(&self) -> &Self::Target {
        self.0
    }
}
