use p3_field::{Field, PrimeCharacteristicRing};

pub mod dft;
pub mod domain;
pub mod fiat_shamir;
pub mod parameters;
pub mod poly;
pub mod sumcheck;
pub mod utils;
pub mod whir;

pub(crate) type PF<F> = <F as PrimeCharacteristicRing>::PrimeSubfield;
pub(crate) type PFPacking<F> = <PF<F> as Field>::Packing;
