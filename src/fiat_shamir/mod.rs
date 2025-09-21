use p3_challenger::{FieldChallenger, GrindingChallenger};
use p3_field::Field;

use crate::{PF, fiat_shamir::verifier::ChallengerState};

pub mod errors;
pub mod prover;
pub mod verifier;

const LEAN_ISA_VECTOR_LEN: usize = 8;

/// A trait for types that can sample challenges in a Fiat-Shamir-based protocol.
///
/// This trait abstracts over objects (such as prover or verifier states) that can
/// deterministically generate random challenges from a transcript using a cryptographic
/// challenger. The challenges are used to drive non-interactive proofs or interactive
/// proof reductions.
pub trait ChallengeSampler<F> {
    fn sample(&mut self) -> F;

    fn sample_vec(&mut self, len: usize) -> Vec<F>;

    fn sample_bits(&mut self, bits: usize) -> usize;
}

pub trait FSChallenger<EF: Field>:
    FieldChallenger<PF<EF>> + GrindingChallenger<Witness = PF<EF>> + ChallengerState
{
}

impl<F: Field, C: FieldChallenger<PF<F>> + GrindingChallenger<Witness = PF<F>> + ChallengerState>
    FSChallenger<F> for C
{
}
