use ndarray::prelude::*;

pub struct Radau5Cache {
    pub prob_type: usize,

    pub implct: bool,
    pub jband: bool,

    pub caljac: bool,
    pub calhes: bool,

    pub mljac: usize,
    pub mujac: usize,
    pub mlmas: usize,
    pub mumas: usize,

    pub m1: usize,
    pub m2: usize,
    pub mle: usize,
    pub mue: usize,
    pub mbjac: usize,
    pub mbb: usize,
    pub mdiag: usize,
    pub mdiff: usize,
    pub mbdiag: usize,

    pub ldmas: usize,
    pub ldjac: usize,
    pub lde1: usize,

    pub u0: Array1<f64>,

    pub scal: Array1<f64>,
    /// Coninuous output vectors
    pub cont1: Array1<f64>,
    pub cont2: Array1<f64>,
    pub cont3: Array1<f64>,
    pub cont4: Array1<f64>,

    pub z1: Array1<f64>,
    pub z2: Array1<f64>,
    pub z3: Array1<f64>,

    pub f1: Array1<f64>,
    pub f2: Array1<f64>,
    pub f3: Array1<f64>,

    pub ip1: Array1<i32>,
    pub ip2: Array1<i32>,
    pub iphes: Array1<i32>,

    pub e1: Array2<f64>,
    pub e2r: Array2<f64>,
    pub e2i: Array2<f64>,

    pub dfdu: Array2<f64>,
    pub mass_matrix: Array2<f64>,
}
