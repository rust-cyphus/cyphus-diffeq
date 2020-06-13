#[derive(PartialEq)]
pub enum OdeRetCode {
    Continue,
    Success,
    MaxIters,
    DtLessThanMin,
    Stiff,
    Failure,
}
