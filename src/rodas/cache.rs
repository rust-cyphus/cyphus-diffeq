use ndarray::prelude::*;

/// Cache structure for Rodas algorithm
pub struct RodasCache {
    pub prob_type: usize,
    pub implct: bool,
    pub jband: bool,
    pub caljac: bool,
    pub first: bool,
    pub reject: bool,
    pub fac1: f64,
    pub alphn: f64,
    pub betan: f64,
    pub err: f64,
    pub unew: Array1<f64>,
    pub du: Array1<f64>,
    pub du1: Array1<f64>,
    pub cont: Array1<f64>,
    pub ak1: Array1<f64>,
    pub ak2: Array1<f64>,
    pub ak3: Array1<f64>,
    pub ak4: Array1<f64>,
    pub ak5: Array1<f64>,
    pub ak6: Array1<f64>,
    pub dfdt: Array1<f64>,
    pub ip: Array1<i32>,
    pub e: Array2<f64>,
    pub dfdu: Array2<f64>,
}
