use p3_field::Field;

pub fn serialize_field<F: Field>(f: &F) -> Vec<u8> {
    let size = std::mem::size_of::<F>();
    let mut bytes = Vec::with_capacity(size);
    unsafe {
        let src_ptr = f as *const F as *const u8;
        bytes.set_len(size);
        std::ptr::copy_nonoverlapping(src_ptr, bytes.as_mut_ptr(), size);
    }
    bytes
}

pub fn deserialize_field<F: Field>(bytes: &[u8]) -> Option<F> {
    // TODO check that the representation is correct
    if bytes.len() != std::mem::size_of::<F>() {
        return None;
    }

    let mut result = std::mem::MaybeUninit::<F>::uninit();

    unsafe {
        std::ptr::copy_nonoverlapping(bytes.as_ptr(), result.as_mut_ptr() as *mut u8, bytes.len());

        Some(result.assume_init())
    }
}

/// Bytes needed in order to obtain a uniformly distributed random element of `modulus_bits`
pub(crate) const fn bytes_uniform_modp(modulus_bits: u32) -> usize {
    (modulus_bits as usize + 128) / 8
}
