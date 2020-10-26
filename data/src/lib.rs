#[macro_use]
extern crate educe;

#[macro_use]
mod macros;

/// A Smallvec instantiation with 4 embeddable values.
///
/// Used about everywhere in tract, for node inputs and outputs, or
/// tensor dimensions.
pub type TVec<T> = smallvec::SmallVec<[T; 4]>;

pub mod prelude {
    pub use crate::datum::{Blob, Datum, DatumType};
    pub use crate::dim::{Symbol, SymbolValues, TDim};
    pub use crate::f16::*;
    pub use crate::tensor::litteral::*;
    pub use crate::tensor::{IntoArcTensor, IntoTensor, Tensor};
    pub use crate::tvec;
    pub use crate::TVec;
    pub use crate::{
        dispatch_copy, dispatch_copy_by_size, dispatch_datum, dispatch_datum_by_size,
        dispatch_floatlike, dispatch_numbers, dispatch_signed,
    };
}

pub mod internal {
    pub use crate::dim::{DimLike, MaybeProduct, TDim, ToDim};
    pub use crate::hash::{dyn_hash, hash_f32, hash_opt_f32, DynHash, SloppyHash};
    pub use crate::impl_dyn_hash;
    pub use crate::prelude::*;
}

pub use anyhow;

mod datum;
mod dim;
mod f16;
mod hash;
mod tensor;
