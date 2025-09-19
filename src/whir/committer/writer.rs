use p3_commit::{ExtensionMmcs, Mmcs};
use p3_field::{ExtensionField, Field, TwoAdicField};
use p3_matrix::{Matrix, dense::RowMajorMatrix};
use p3_merkle_tree::MerkleTreeMmcs;
use p3_symmetric::{CryptographicHasher, PseudoCompressionFunction};
use p3_util::log2_strict_usize;
use serde::{Deserialize, Serialize};

use super::Witness;
use crate::{
    PF, PFPacking,
    dft::EvalsDft,
    fiat_shamir::{FSChallenger, prover::ProverState},
    poly::evals::EvaluationsList,
    utils::parallel_repeat,
    whir::{
        config::{WhirConfig, WhirConfigBuilder},
        utils::sample_ood_points,
    },
};

impl<H, C, const DIGEST_ELEMS: usize> WhirConfigBuilder<H, C, DIGEST_ELEMS> {
    pub fn commit<F, EF>(
        &self,
        dft: &EvalsDft<PF<EF>>,
        prover_state: &mut ProverState<PF<EF>, EF, impl FSChallenger<EF>>,
        polynomial: &[F],
    ) -> Witness<F, EF, DIGEST_ELEMS>
    where
        F: TwoAdicField + ExtensionField<PF<EF>>,
        EF: TwoAdicField + ExtensionField<F> + ExtensionField<PF<EF>>,
        H: CryptographicHasher<PF<EF>, [PF<EF>; DIGEST_ELEMS]>
            + CryptographicHasher<PFPacking<EF>, [PFPacking<EF>; DIGEST_ELEMS]>
            + Sync,
        C: PseudoCompressionFunction<[PF<EF>; DIGEST_ELEMS], 2>
            + PseudoCompressionFunction<[PFPacking<EF>; DIGEST_ELEMS], 2>
            + Sync,
        [PF<EF>; DIGEST_ELEMS]: Serialize + for<'de> Deserialize<'de>,
        PF<EF>: TwoAdicField,
    {
        WhirConfig::new(self.clone(), log2_strict_usize(polynomial.len())).commit(
            dft,
            prover_state,
            polynomial,
        )
    }
}

impl<'a, F, EF, H, C, const DIGEST_ELEMS: usize> WhirConfig<F, EF, H, C, DIGEST_ELEMS>
where
    F: Field,
    EF: ExtensionField<F> + ExtensionField<PF<EF>>,
    PF<EF>: TwoAdicField,
    F: ExtensionField<PF<EF>>,
{
    /// Commits a polynomial using a Merkle-based commitment scheme.
    ///
    /// This function:
    /// - Expands polynomial coefficients to evaluations.
    /// - Applies folding and restructuring optimizations.
    /// - Converts evaluations to an extension field.
    /// - Constructs a Merkle tree from the evaluations.
    /// - Computes out-of-domain (OOD) challenge points and their evaluations.
    /// - Returns a `Witness` containing the commitment data.
    pub fn commit(
        &self,
        dft: &EvalsDft<PF<EF>>,
        prover_state: &mut ProverState<PF<EF>, EF, impl FSChallenger<EF>>,
        polynomial: &[F],
    ) -> Witness<F, EF, DIGEST_ELEMS>
    where
        H: CryptographicHasher<PF<EF>, [PF<EF>; DIGEST_ELEMS]>
            + CryptographicHasher<PFPacking<EF>, [PFPacking<EF>; DIGEST_ELEMS]>
            + Sync,
        C: PseudoCompressionFunction<[PF<EF>; DIGEST_ELEMS], 2>
            + PseudoCompressionFunction<[PFPacking<EF>; DIGEST_ELEMS], 2>
            + Sync,
        [PF<EF>; DIGEST_ELEMS]: Serialize + for<'de> Deserialize<'de>,
    {
        let evals_repeated = parallel_repeat(polynomial, 1 << self.starting_log_inv_rate);

        // Perform DFT on the padded evaluations matrix
        let width = 1 << self.folding_factor.at_round(0);
        let folded_matrix = dft
            .dft_algebra_batch_by_evals(RowMajorMatrix::new(evals_repeated, width))
            .to_row_major_matrix();

        // Commit to the Merkle tree
        let mmcs = MerkleTreeMmcs::<PFPacking<EF>, PFPacking<EF>, H, C, DIGEST_ELEMS>::new(
            self.merkle_hash.clone(),
            self.merkle_compress.clone(),
        );
        let extension_mmcs_f = ExtensionMmcs::<PF<EF>, F, _>::new(mmcs.clone());
        let (root, prover_data) = extension_mmcs_f.commit_matrix(folded_matrix);

        prover_state.add_base_scalars(root.as_ref());

        // Handle OOD (Out-Of-Domain) samples
        let (ood_points, ood_answers) = sample_ood_points::<F, EF, _>(
            prover_state,
            self.committment_ood_samples,
            self.num_variables,
            |point| polynomial.evaluate(point),
        );

        // Return the witness containing the polynomial, Merkle tree, and OOD results.
        Witness {
            prover_data,
            ood_points,
            ood_answers,
        }
    }
}
