#[derive(PartialEq, Debug)]
pub enum OdeRetCode {
    Continue,
    Success,
    MaxIters,
    DtLessThanMin,
    Stiff,
    SingularMatrix,
    Failure,
}
