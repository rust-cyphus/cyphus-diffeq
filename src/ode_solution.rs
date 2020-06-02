use crate::ode_algorithm::OdeAlgorithm;
use crate::ode_stats::OdeStats;
use ndarray::prelude::*;

pub(crate) struct OdeSolution {
    pub(crate) ts: Vec<f64>,
    pub(crate) us: Vec<Array1<f64>>,
    pub(crate) stats: OdeStats,
    //pub(crate) cont: Vec<Array1<f64>>,
}

// impl<Alg: OdeAlgorithm> OdeSolution<Alg> {
//     fn u(t: f64) -> Array1<f64> {
//         alg.dense_output(t)
//     }
// }
