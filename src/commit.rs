use std::hash::{Hash, Hasher};

use multilinear_toolkit::prelude::*;
use p3_commit::{ExtensionMmcs, Mmcs};
use p3_field::{ExtensionField, TwoAdicField};
use p3_matrix::{dense::DenseMatrix, extension::FlatMatrixView};
use p3_merkle_tree::{MerkleTree, MerkleTreeMmcs};
use serde::{Deserialize, Serialize};
use tracing::instrument;

use crate::*;

pub type RoundMerkleTree<F, EF> =
    MerkleTree<F, F, FlatMatrixView<F, EF, DenseMatrix<EF>>, DIGEST_ELEMS>;

#[derive(Debug)]
pub enum MerkleData<EF: ExtensionField<PF<EF>>> {
    Base(RoundMerkleTree<PF<EF>, PF<EF>>),
    Extension(RoundMerkleTree<PF<EF>, EF>),
}

impl<EF: ExtensionField<PF<EF>>> MerkleData<EF> {
    #[instrument(skip_all, name = "build merkle tree")]
    pub(crate) fn build<H, C>(
        merkle_hash: H,
        merkle_compress: C,
        matrix: DftOutput<EF>,
    ) -> (Self, [PF<EF>; DIGEST_ELEMS])
    where
        H: MerkleHasher<EF>,
        C: MerkleCompress<EF>,
        [PF<EF>; DIGEST_ELEMS]: Serialize + for<'de> Deserialize<'de>,
    {
        let mmcs = MerkleTreeMmcs::<PFPacking<EF>, PFPacking<EF>, H, C, DIGEST_ELEMS>::new(
            merkle_hash,
            merkle_compress,
        );
        let extension_mmcs_f = ExtensionMmcs::<PF<EF>, PF<EF>, _>::new(mmcs.clone());
        let extension_mmcs_ef = ExtensionMmcs::<PF<EF>, EF, _>::new(mmcs.clone());

        match matrix {
            DftOutput::Base(m) => {
                let (root, prover_data) = extension_mmcs_f.commit_matrix(m);
                (MerkleData::Base(prover_data), root.into())
            }
            DftOutput::Extension(m) => {
                let (root, prover_data) = extension_mmcs_ef.commit_matrix(m);
                (MerkleData::Extension(prover_data), root.into())
            }
        }
    }

    pub(crate) fn open<H, C>(
        &self,
        index: usize,
        merkle_hash: H,
        merkle_compress: C,
    ) -> (MleOwned<EF>, Vec<[PF<EF>; DIGEST_ELEMS]>)
    where
        H: MerkleHasher<EF>,
        C: MerkleCompress<EF>,
        [PF<EF>; DIGEST_ELEMS]: Serialize + for<'de> Deserialize<'de>,
    {
        let mmcs = MerkleTreeMmcs::<PFPacking<EF>, PFPacking<EF>, H, C, DIGEST_ELEMS>::new(
            merkle_hash,
            merkle_compress,
        );
        let extension_mmcs_f = ExtensionMmcs::<PF<EF>, PF<EF>, _>::new(mmcs.clone());
        let extension_mmcs_ef = ExtensionMmcs::<PF<EF>, EF, _>::new(mmcs.clone());

        match self {
            MerkleData::Base(prover_data) => {
                let mut batch_opening = extension_mmcs_f.open_batch(index, prover_data);
                (
                    MleOwned::Base(std::mem::take(&mut batch_opening.opened_values[0])),
                    batch_opening.opening_proof,
                )
            }
            MerkleData::Extension(prover_data) => {
                let mut batch_opening = extension_mmcs_ef.open_batch(index, prover_data);
                (
                    MleOwned::Extension(std::mem::take(&mut batch_opening.opened_values[0])),
                    batch_opening.opening_proof,
                )
            }
        }
    }
}

#[derive(Debug)]
pub struct Witness<EF>
where
    EF: ExtensionField<PF<EF>>,
{
    /// Prover data of the Merkle tree.  
    pub prover_data: MerkleData<EF>,
    /// Out-of-domain challenge points used for polynomial verification.  
    pub ood_points: Vec<EF>,
    /// The corresponding polynomial evaluations at the OOD challenge points.  
    pub ood_answers: Vec<EF>,
}

impl<'a, EF, H, C> WhirConfig<EF, H, C>
where
    EF: ExtensionField<PF<EF>>,
    PF<EF>: TwoAdicField,
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
    #[instrument(skip_all)]
    pub fn commit(
        &self,
        dft: &EvalsDft<PF<EF>>,
        prover_state: &mut ProverState<PF<EF>, EF, impl FSChallenger<EF>>,
        polynomial: &MleOwned<EF>,
    ) -> Witness<EF>
    where
        H: MerkleHasher<EF>,
        C: MerkleCompress<EF>,
        [PF<EF>; DIGEST_ELEMS]: Serialize + for<'de> Deserialize<'de>,
    {
        // Perform DFT on the padded evaluations matrix
        let folded_matrix = reorder_and_dft(
            &polynomial.by_ref(),
            dft,
            self.folding_factor.at_round(0),
            self.starting_log_inv_rate,
        );

        let (prover_data, root) = MerkleData::build(
            self.merkle_hash.clone(),
            self.merkle_compress.clone(),
            folded_matrix,
        );

        prover_state.add_base_scalars(&root);

        // Handle OOD (Out-Of-Domain) samples
        let (ood_points, ood_answers) = sample_ood_points::<EF, _>(
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

fn hash<A: Hash>(a: &A) -> u64 {
    use std::collections::hash_map::DefaultHasher;
    let mut hasher = DefaultHasher::new();
    a.hash(&mut hasher);
    hasher.finish()
}
