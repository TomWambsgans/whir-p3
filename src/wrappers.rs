use multilinear_toolkit::prelude::*;
use p3_symmetric::{CryptographicHasher, PseudoCompressionFunction};

pub const DIGEST_ELEMS: usize = 8;

pub trait MerkleHasher<EF: Field>:
    CryptographicHasher<PFPacking<EF>, [PFPacking<EF>; DIGEST_ELEMS]>
    + CryptographicHasher<PF<EF>, [PF<EF>; DIGEST_ELEMS]>
    + Sync
{
}

pub trait MerkleCompress<EF: Field>:
    PseudoCompressionFunction<[PFPacking<EF>; DIGEST_ELEMS], 2>
    + PseudoCompressionFunction<[PF<EF>; DIGEST_ELEMS], 2>
    + Sync
{
}

impl<
    EF: Field,
    MH: CryptographicHasher<PFPacking<EF>, [PFPacking<EF>; DIGEST_ELEMS]>
        + CryptographicHasher<PF<EF>, [PF<EF>; DIGEST_ELEMS]>
        + Sync,
> MerkleHasher<EF> for MH
{
}

impl<
    EF: Field,
    MC: PseudoCompressionFunction<[PFPacking<EF>; DIGEST_ELEMS], 2>
        + PseudoCompressionFunction<[PF<EF>; DIGEST_ELEMS], 2>
        + Sync,
> MerkleCompress<EF> for MC
{
}
