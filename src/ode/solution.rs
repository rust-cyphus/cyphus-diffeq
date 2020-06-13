use super::code::OdeRetCode;
use ndarray::prelude::*;

/// Solution object for holding solution to an ODE.
pub struct OdeSolution {
    pub ts: Vec<f64>,
    pub us: Vec<Array1<f64>>,
    pub retcode: OdeRetCode,
}

impl OdeSolution {
    pub fn new() -> OdeSolution {
        OdeSolution {
            ts: vec![],
            us: vec![],
            retcode: OdeRetCode::Continue,
        }
    }
}
