use p3_challenger::{FieldChallenger, GrindingChallenger};
use p3_field::Field;

use crate::PF;

mod errors;
pub use errors::*;

mod prover;
pub use prover::*;

mod verifier;
pub use verifier::*;

const LEAN_ISA_VECTOR_LEN: usize = 8;

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
