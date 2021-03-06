use ndarray::prelude::*;

pub struct Radau5Cache {
    pub(crate) caljac: bool,
    pub(crate) first: bool,
    pub(crate) last: bool,
    pub(crate) reject: bool,
    pub(crate) decompose: bool,
    pub(crate) fac1: f64,
    pub(crate) alphn: f64,
    pub(crate) betan: f64,
    pub(crate) dtopt: f64,
    pub(crate) faccon: f64,
    pub(crate) dtfac: f64,
    pub(crate) dtacc: f64,
    pub(crate) erracc: f64,
    pub(crate) thqold: f64,
    pub(crate) cfac: f64,
    pub(crate) theta: f64,
    pub(crate) nsing: usize,
    pub(crate) newt: usize,
    pub(crate) err: f64,
    pub(crate) u0: Array1<f64>,
    pub(crate) scal: Array1<f64>,
    pub(crate) cont: Array1<f64>,
    pub(crate) z1: Array1<f64>,
    pub(crate) z2: Array1<f64>,
    pub(crate) z3: Array1<f64>,
    pub(crate) f1: Array1<f64>,
    pub(crate) f2: Array1<f64>,
    pub(crate) f3: Array1<f64>,
    pub(crate) ip1: Array1<i32>,
    pub(crate) ip2: Array1<i32>,
    pub(crate) e1: Array2<f64>,
    pub(crate) e2r: Array2<f64>,
    pub(crate) e2i: Array2<f64>,
    pub(crate) dfdu: Array2<f64>,
}
