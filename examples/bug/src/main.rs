use tract_onnx::prelude::*;
use env_logger::Builder;
use log::LevelFilter;


fn main() -> TractResult<()> {
    let mut builder = Builder::new();
    builder.filter_level(LevelFilter::Trace).init();

    tract_onnx::onnx()
        .model_for_path("model.onnx")?
        .with_input_fact(0, InferenceFact::dt_shape(i64::datum_type(), tvec!(1, 64)))?
        .with_input_fact(1, InferenceFact::dt_shape(i64::datum_type(), tvec!(1, 64)))?
        .with_input_fact(2, InferenceFact::dt_shape(i64::datum_type(), tvec!(1, 64)))?
        .into_optimized()?
        .into_runnable()?;

    Ok(())
}
