use ndarray::prelude::*;

pub struct DormandPrince5Cache {
    pub(crate) n_stiff: usize,
    pub(crate) n_nonstiff: usize,
    pub(crate) reject: bool,
    pub(crate) last: bool,
    pub(crate) facold: f64, // previous ratio of dtnew/dt
    pub(crate) dtlamb: f64,
    pub(crate) k2: Array1<f64>,
    pub(crate) k3: Array1<f64>,
    pub(crate) k4: Array1<f64>,
    pub(crate) k5: Array1<f64>,
    pub(crate) k6: Array1<f64>,
    pub(crate) rcont1: Array1<f64>,
    pub(crate) rcont2: Array1<f64>,
    pub(crate) rcont3: Array1<f64>,
    pub(crate) rcont4: Array1<f64>,
    pub(crate) rcont5: Array1<f64>,
    pub(crate) unew: Array1<f64>,
    pub(crate) du: Array1<f64>,
    pub(crate) dunew: Array1<f64>,
    pub(crate) uerr: Array1<f64>,
    pub(crate) ustiff: Array1<f64>,
}
