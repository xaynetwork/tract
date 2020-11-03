use crate::internal::*;
use ndarray::prelude::*;
use crate::num_traits::Zero;

#[derive(Debug, Clone, Default, PartialEq, Hash)]
pub struct Slice {
    pub axis: usize,
    pub start: TDim,
    pub end: TDim,
}

impl DynHash for Slice {
    fn dyn_hash(&self, hasher: &mut dyn std::hash::Hasher) {
        dyn_hash(&self, hasher)
    }
}

impl Slice {
    pub fn new(axis: usize, start: impl ToDim, end: impl ToDim) -> Slice {
        Slice { axis, start: start.to_dim(), end: end.to_dim() }
    }

    unsafe fn eval_t<T: Datum>(&self, input: &Tensor) -> TractResult<Tensor> {
        let mut input = input.to_array_view_unchecked::<T>();
        input.slice_axis_inplace(
            Axis(self.axis),
            ndarray::Slice::from((self.start.to_isize()?)..(self.end.to_isize()?)),
        );
        Ok(Tensor::from(input.to_owned()).into())
    }
}

impl Op for Slice {
    fn name(&self) -> Cow<str> {
        "Slice".into()
    }

    fn info(&self) -> TractResult<Vec<String>> {
        Ok(vec![format!("axis: {}, {}..{}", self.axis, self.start, self.end)])
    }

    op_core_lir_mir!();
    op_as_typed_op!();

    fn same_as(&self, other: &dyn Op) -> bool {
        if let Some(other) = other.downcast_ref::<Self>() {
            other == self
        } else {
            false
        }
    }
}

impl EvalOp for Slice {
    fn is_stateless(&self) -> bool {
        true
    }

    fn eval(&self, mut inputs: TVec<Arc<Tensor>>) -> TractResult<TVec<Arc<Tensor>>> {
        let input = args_1!(inputs);
        unsafe {
            let mut tensor =
                dispatch_datum_by_size!(Self::eval_t(input.datum_type())(self, &input))?;
            tensor.set_datum_type(input.datum_type());
            Ok(tvec!(tensor.into_arc_tensor()))
        }
    }
}

impl TypedOp for Slice {
    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        let mut fact = inputs[0].clone();
        fact.shape[self.axis] = (self.end.clone() - &self.start).to_dim();
        Ok(tvec!(fact))
    }

    fn invariants(&self, model: &TypedModel, node: &TypedNode) -> TractResult<Invariants> {
        let fact = model.outlet_fact(node.inputs[0])?;
        let axes = (0..fact.shape.rank())
            .filter(|&ax| self.axis != ax)
            .map(|axis| AxisInfo::simple(axis))
            .collect();
        Ok(axes)
    }

    fn change_axes(
        &self,
        model: &TypedModel,
        node: &TypedNode,
        _io: InOut,
        change: &AxisOp,
    ) -> TractResult<Option<AxisChangeConsequence>> {
        if let Some(axis) = change.transform_axis(self.axis) {
            if axis != self.axis {
                Ok(Some(AxisChangeConsequence::new(
                    model,
                    node,
                    Some(Box::new(Slice { axis, ..self.clone() }) as _),
                    change,
                )))
            } else {
                Ok(Some(AxisChangeConsequence::new(model, node, None, change)))
            }
        } else {
            Ok(None)
        }
    }

    fn declutter(
        &self,
        model: &TypedModel,
        node: &TypedNode,
    ) -> TractResult<Option<TypedModelPatch>> {
        let prec = model.node(node.inputs[0].node);
        if self.start.is_zero() && (self.end == model.outlet_fact(node.inputs[0])?.shape[self.axis])
        {
            return Ok(Some(TypedModelPatch::shunt_one_op(model, node)?.with_context("noop")));
        }
        let (start, end) = if let (Ok(s), Ok(e)) = (self.start.to_usize(), self.end.to_usize()) {
            (s, e)
        } else {
            return Ok(None);
        };
        let mut patch = TypedModelPatch::default();
        if let Some(wire) = prec.op().as_typed().unwrap().slice_output(
            model,
            prec,
            &mut patch,
            node.inputs[0].slot,
            self.axis,
            start,
            end,
        )? {
            patch.shunt_outside(model, OutletId::new(node.id, 0), wire)?;
            if patch.model.nodes.len() == 2 && patch.model.node(1).op().same_as(self) {
                return Ok(None);
            } else if patch.model.nodes.len() == 3 {
                let other = model.node(node.inputs[0].node);
                if other.op_is::<Self>() {
                    patch.dont_apply_twice = Some(format!("Swap {} and {}", node.name, other.name));
                }
            }
            return Ok(Some(patch));
        }
        Ok(None)
    }

    fn slice_output(
        &self,
        model: &TypedModel,
        node: &TypedNode,
        patch: &mut TypedModelPatch,
        _output_slot: usize,
        axis: usize,
        start: usize,
        end: usize,
    ) -> TractResult<Option<OutletId>> {
        let prec = model.node(node.inputs[0].node);
        if axis != self.axis {
            return prec
                .op()
                .as_typed()
                .unwrap()
                .slice_output(model, &prec, patch, node.inputs[0].slot, axis, start, end)?
                .map(|w| Ok(patch.wire_node(&node.name, self.clone(), &[w])?[0]))
                .transpose();
        }
        Ok(None)
    }

    as_op!();
}
