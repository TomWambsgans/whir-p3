use std::hash::{Hash, Hasher};

use multilinear_toolkit::prelude::*;
use p3_field::{ExtensionField, TwoAdicField};
use tracing::instrument;

use crate::*;

#[derive(Debug, Clone)]
pub enum MerkleData<EF: ExtensionField<PF<EF>>> {
    Base(RoundMerkleTree<PF<EF>, PF<EF>>),
    Extension(RoundMerkleTree<PF<EF>, EF>),
}

impl<EF: ExtensionField<PF<EF>>> MerkleData<EF> {
    #[instrument(skip_all, name = "build merkle tree")]
    pub(crate) fn build(matrix: DftOutput<EF>) -> (Self, [PF<EF>; DIGEST_ELEMS]) {
        match matrix {
            DftOutput::Base(m) => {
                let (root, prover_data) = merkle_commit::<PF<EF>, PF<EF>>(m);
                (MerkleData::Base(prover_data), root)
            }
            DftOutput::Extension(m) => {
                let (root, prover_data) = merkle_commit::<PF<EF>, EF>(m);
                (MerkleData::Extension(prover_data), root)
            }
        }
    }

    pub(crate) fn open(&self, index: usize) -> (MleOwned<EF>, Vec<[PF<EF>; DIGEST_ELEMS]>) {
        match self {
            MerkleData::Base(prover_data) => {
                let (leaf, proof) = merkle_open::<PF<EF>, PF<EF>>(prover_data, index);
                (MleOwned::Base(leaf), proof)
            }
            MerkleData::Extension(prover_data) => {
                let (leaf, proof) = merkle_open::<PF<EF>, EF>(prover_data, index);
                (MleOwned::Extension(leaf), proof)
            }
        }
    }
}

#[derive(Debug, Clone)]
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

impl<'a, EF> WhirConfig<EF>
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
        prover_state: &mut ProverState<PF<EF>, EF, impl FSChallenger<EF>>,
        polynomial: &MleOwned<EF>,
    ) -> Witness<EF> {
        // Perform DFT on the padded evaluations matrix
        let folded_matrix = reorder_and_dft(
            &polynomial.by_ref(),
            self.folding_factor.at_round(0),
            self.starting_log_inv_rate,
        );

        let (prover_data, root) = MerkleData::build(folded_matrix);

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
