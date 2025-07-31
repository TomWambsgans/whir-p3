use std::ops::Deref;

use p3_challenger::{FieldChallenger, GrindingChallenger};
use p3_commit::{ExtensionMmcs, Mmcs};
use p3_field::{ExtensionField, Field, TwoAdicField};
use p3_matrix::{Matrix, dense::RowMajorMatrix};
use p3_merkle_tree::MerkleTreeMmcs;
use p3_symmetric::{CryptographicHasher, PseudoCompressionFunction};
use serde::{Deserialize, Serialize};
use tracing::{info_span, instrument};

use super::Witness;
use crate::{
    PF, PFPacking,
    dft::EvalsDft,
    fiat_shamir::{errors::ProofResult, prover::ProverState, verifier::ChallengerState},
    poly::evals::EvaluationsList,
    utils::parallel_repeat,
    whir::{config::WhirConfig, utils::sample_ood_points},
};

/// Responsible for committing polynomials using a Merkle-based scheme.
///
/// The `Committer` processes a polynomial, expands and folds its evaluations,
/// and constructs a Merkle tree from the resulting values.
///
/// It provides a commitment that can be used for proof generation and verification.
#[derive(Debug)]
pub struct CommitmentWriter<'a, F, EF, H, C, Challenger, const DIGEST_ELEMS: usize>(
    /// Reference to the WHIR protocol configuration.
    &'a WhirConfig<F, EF, H, C, Challenger, DIGEST_ELEMS>,
)
where
    F: Field,
    EF: ExtensionField<F>;

impl<'a, F, EF, H, C, Challenger, const DIGEST_ELEMS: usize> CommitmentWriter<'a, F, EF, H, C, Challenger, DIGEST_ELEMS>
where
    F: Field,
    EF: ExtensionField<F> + ExtensionField<PF<F>>,
    PF<F>: TwoAdicField,
    F: ExtensionField<PF<F>>,
    Challenger: FieldChallenger<PF<F>> + GrindingChallenger<Witness = PF<F>> + ChallengerState,
{
    /// Create a new writer that borrows the WHIR protocol configuration.
    pub const fn new(params: &'a WhirConfig<F, EF, H, C, Challenger, DIGEST_ELEMS>) -> Self {
        Self(params)
    }

    /// Commits a polynomial using a Merkle-based commitment scheme.
    ///
    /// This function:
    /// - Expands polynomial coefficients to evaluations.
    /// - Applies folding and restructuring optimizations.
    /// - Converts evaluations to an extension field.
    /// - Constructs a Merkle tree from the evaluations.
    /// - Computes out-of-domain (OOD) challenge points and their evaluations.
    /// - Returns a `Witness` containing the commitment data.
    #[instrument(skip_all)]
    pub fn commit(
        &self,
        dft: &EvalsDft<PF<F>>,
        prover_state: &mut ProverState<PF<F>, EF, Challenger>,
        polynomial: &EvaluationsList<F>,
    ) -> ProofResult<Witness<F, EF, DIGEST_ELEMS>>
    where
        H: CryptographicHasher<PF<F>, [PF<F>; DIGEST_ELEMS]>
            + CryptographicHasher<PFPacking<F>, [PFPacking<F>; DIGEST_ELEMS]>
            + Sync,
        C: PseudoCompressionFunction<[PF<F>; DIGEST_ELEMS], 2>
            + PseudoCompressionFunction<[PFPacking<F>; DIGEST_ELEMS], 2>
            + Sync,
        [PF<F>; DIGEST_ELEMS]: Serialize + for<'de> Deserialize<'de>,
    {
        let evals_repeated = info_span!("repeating evals")
            .in_scope(|| parallel_repeat(polynomial.evals(), 1 << self.starting_log_inv_rate));

        // Perform DFT on the padded evaluations matrix
        let width = 1 << self.folding_factor.at_round(0);
        let folded_matrix = info_span!("dft", height = evals_repeated.len() / width, width)
            .in_scope(|| {
                dft.dft_algebra_batch_by_evals(RowMajorMatrix::new(evals_repeated, width))
                    .to_row_major_matrix()
            });

        // Commit to the Merkle tree
        let mmcs = MerkleTreeMmcs::<PFPacking<F>, PFPacking<F>, H, C, DIGEST_ELEMS>::new(
            self.merkle_hash.clone(),
            self.merkle_compress.clone(),
        );
        let extension_mmcs_f = ExtensionMmcs::<PF<F>, F, _>::new(mmcs.clone());
        let (root, prover_data) =
            info_span!("commit matrix").in_scope(|| extension_mmcs_f.commit_matrix(folded_matrix));

        prover_state.add_base_scalars(root.as_ref());

        // Handle OOD (Out-Of-Domain) samples
        let (ood_points, ood_answers) = sample_ood_points::<F, EF, _, _>(
            prover_state,
            self.committment_ood_samples,
            self.num_variables,
            |point| info_span!("ood evaluation").in_scope(|| polynomial.evaluate(point)),
        );

        // Return the witness containing the polynomial, Merkle tree, and OOD results.
        Ok(Witness {
            prover_data,
            ood_points,
            ood_answers,
        })
    }
}

impl<F, EF, H, C, Challenger, const DIGEST_ELEMS: usize> Deref for CommitmentWriter<'_, F, EF, H, C, Challenger, DIGEST_ELEMS>
where
    F: Field,
    EF: ExtensionField<F>,
{
    type Target = WhirConfig<F, EF, H, C, Challenger, DIGEST_ELEMS>;

    fn deref(&self) -> &Self::Target {
        self.0
    }
}
