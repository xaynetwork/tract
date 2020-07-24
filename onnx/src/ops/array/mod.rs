mod compress;
mod pad;
mod slice;

use tract_hir::internal::*;
use tract_hir::ops::array;

use crate::model::{OnnxOpRegister, ParsingContext};
use crate::pb::*;

pub fn register_all_ops(reg: &mut OnnxOpRegister) {
    reg.insert("Compress", compress::compress);
    reg.insert("Concat", concat);
    reg.insert("ConstantOfShape", constant_of_shape);
    reg.insert("Expand", |_, _| Ok((expand(array::MultiBroadcastTo::default()), vec![])));
    reg.insert("EyeLike", eye_like);
    reg.insert("Flatten", flatten);
    reg.insert("Gather", gather);
    reg.insert("Pad", pad::pad);
    reg.insert("Reshape", |_, _| Ok((expand(array::Reshape::default()), vec![])));
    reg.insert("Shape", |_, _| Ok((expand(array::Shape::new(DatumType::I64)), vec![])));
    reg.insert("Size", |_, _| Ok((expand(array::Size::new(DatumType::I64)), vec![])));
    reg.insert("Slice", slice::slice);
    reg.insert("Split", split);
    reg.insert("Squeeze", squeeze);
    reg.insert("Tile", |_, _| Ok((expand(array::Tile::default()), vec![])));
    reg.insert("Transpose", transpose);
    reg.insert("Unsqueeze", unsqueeze);
}

pub fn concat(
    _ctx: &ParsingContext,
    node: &NodeProto,
) -> TractResult<(Box<dyn InferenceOp>, Vec<String>)> {
    let axis = node.get_attr("axis")?;
    Ok((expand(array::Concat::new(axis)), vec![]))
}

pub fn constant_of_shape(
    _ctx: &ParsingContext,
    node: &NodeProto,
) -> TractResult<(Box<dyn InferenceOp>, Vec<String>)> {
    let value = node.get_attr_opt::<Tensor>("value")?.unwrap_or(tensor0(0.0f32));
    Ok((expand(array::ConstantOfShape::new(value.into_arc_tensor())), vec![]))
}

pub fn eye_like(
    _ctx: &ParsingContext,
    node: &NodeProto,
) -> TractResult<(Box<dyn InferenceOp>, Vec<String>)> {
    let dt = node.get_attr_opt("dtype")?;
    let k = node.get_attr_opt("k")?.unwrap_or(0);
    Ok((Box::new(array::EyeLike::new(dt, k)), vec![]))
}

pub fn flatten(
    _ctx: &ParsingContext,
    node: &NodeProto,
) -> TractResult<(Box<dyn InferenceOp>, Vec<String>)> {
    let axis = node.get_attr_opt("axis")?.unwrap_or(1);
    Ok((expand(array::Flatten::new(axis)), vec![]))
}

pub fn gather(
    _ctx: &ParsingContext,
    node: &NodeProto,
) -> TractResult<(Box<dyn InferenceOp>, Vec<String>)> {
    let axis = node.get_attr_opt("axis")?.unwrap_or(0);
    Ok((Box::new(array::Gather::new(axis)), vec![]))
}

pub fn split(
    _ctx: &ParsingContext,
    node: &NodeProto,
) -> TractResult<(Box<dyn InferenceOp>, Vec<String>)> {
    let axis = node.get_attr_opt("axis")?.unwrap_or(0);
    let split = node.get_attr_opt_vec("split")?;
    Ok((expand(array::Split::new(axis, node.output.len(), split)), vec![]))
}

pub fn squeeze(
    _ctx: &ParsingContext,
    node: &NodeProto,
) -> TractResult<(Box<dyn InferenceOp>, Vec<String>)> {
    let axes = node.get_attr_opt_vec("axes")?;
    Ok((expand(array::Squeeze::new(axes)), vec![]))
}

pub fn transpose(
    _ctx: &ParsingContext,
    node: &NodeProto,
) -> TractResult<(Box<dyn InferenceOp>, Vec<String>)> {
    let perm = node.get_attr_opt_vec("perm")?;
    Ok((expand(array::PermuteAxes::new(perm.map(|t| t.into()))), vec![]))
}

pub fn unsqueeze(
    _ctx: &ParsingContext,
    node: &NodeProto,
) -> TractResult<(Box<dyn InferenceOp>, Vec<String>)> {
    let axes = node.get_attr_vec::<i64>("axes")?.into_iter().map(|x| x as isize).collect();
    Ok((expand(array::AddDims::new(axes)), vec![]))
}
