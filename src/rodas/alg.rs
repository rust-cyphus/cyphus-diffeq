use crate::ode_algorithm::OdeAlgorithm;

/// The Rodas algorithm.
#[derive(Clone)]
pub struct Rodas {
    /// Relative tolerance
    pub(crate) reltol: f64,
    ///(crate) Absolute tolerance
    pub(crate) abstol: f64,
    ///(crate) Flag specifying if dense output is requested
    pub(crate) dense: bool,
    ///(crate) Initial step value
    pub(crate) dtstart: f64,
    ///(crate) Maximum allowed step size
    pub(crate) dtmax: f64,
    ///(crate) Maximum number of allowed steps
    pub(crate) max_steps: usize,
    ///(crate) If true, use modern predictive controller (Gustafsson).
    pub(crate) modern_pred: bool,
    ///(crate) Safety factor used in adaptive step selection.
    pub(crate) safe: f64,
    ///(crate) Gustafsson step control factor. Restricts dtnew/dtold <= 1/facr.
    pub(crate) facr: f64,
    ///(crate) Gustafsson step control factor. Restricts dtnew/dtold >= 1/facl.
    pub(crate) facl: f64,
    ///(crate) Minimum allowed ratio of dtnew/dtold such that dt will be held constant.
    pub(crate) quot1: f64,
    ///(crate) Maximum allowed ratio of dtnew/dtold such that dt will be held constant.
    pub(crate) quot2: f64,
    /// Parameter to determine if Jacobian needs to be recomputed.
    pub(crate) thet: f64,
    ///(crate) If true, the Jacobian will be converted into Hessenberg form.
    pub(crate) hess: bool,
}

// Declare all the constants needed for Radau5
impl Rodas {
    pub(crate) const C2: f64 = 0.386;
    pub(crate) const C3: f64 = 0.21;
    pub(crate) const C4: f64 = 0.63;
    pub(crate) const BET2P: f64 = 0.0317;
    pub(crate) const BET3P: f64 = 0.0635;
    pub(crate) const BET4P: f64 = 0.3438;
    pub(crate) const D1: f64 = 0.2500000000000000e00;
    pub(crate) const D2: f64 = -0.1043000000000000e00;
    pub(crate) const D3: f64 = 0.1035000000000000e00;
    pub(crate) const D4: f64 = -0.3620000000000023e-01;
    pub(crate) const A21: f64 = 0.1544000000000000e01;
    pub(crate) const A31: f64 = 0.9466785280815826e00;
    pub(crate) const A32: f64 = 0.2557011698983284e00;
    pub(crate) const A41: f64 = 0.3314825187068521e01;
    pub(crate) const A42: f64 = 0.2896124015972201e01;
    pub(crate) const A43: f64 = 0.9986419139977817e00;
    pub(crate) const A51: f64 = 0.1221224509226641e01;
    pub(crate) const A52: f64 = 0.6019134481288629e01;
    pub(crate) const A53: f64 = 0.1253708332932087e02;
    pub(crate) const A54: f64 = -0.6878860361058950e00;
    pub(crate) const C21: f64 = -0.5668800000000000e01;
    pub(crate) const C31: f64 = -0.2430093356833875e01;
    pub(crate) const C32: f64 = -0.2063599157091915e00;
    pub(crate) const C41: f64 = -0.1073529058151375e00;
    pub(crate) const C42: f64 = -0.9594562251023355e01;
    pub(crate) const C43: f64 = -0.2047028614809616e02;
    pub(crate) const C51: f64 = 0.7496443313967647e01;
    pub(crate) const C52: f64 = -0.1024680431464352e02;
    pub(crate) const C53: f64 = -0.3399990352819905e02;
    pub(crate) const C54: f64 = 0.1170890893206160e02;
    pub(crate) const C61: f64 = 0.8083246795921522e01;
    pub(crate) const C62: f64 = -0.7981132988064893e01;
    pub(crate) const C63: f64 = -0.3152159432874371e02;
    pub(crate) const C64: f64 = 0.1631930543123136e02;
    pub(crate) const C65: f64 = -0.6058818238834054e01;
    pub(crate) const GAMMA: f64 = 0.2500000000000000e00;
}

impl OdeAlgorithm for Rodas {
    fn default() -> Self {
        Rodas {
            reltol: 1e-3,
            abstol: 1e-6,
            dense: false,
            dtstart: 1e-6,
            dtmax: 0.0,
            max_steps: 100_000,
            modern_pred: true,
            safe: 0.9,
            facr: 1.0 / 8.0,
            facl: 5.0,
            quot1: 1.0,
            quot2: 1.2,
            thet: 0.001,
            hess: false,
        }
    }
    fn reltol(&mut self, val: f64) {
        self.reltol = val;
    }
    fn abstol(&mut self, val: f64) {
        self.abstol = val
    }
    fn dense(&mut self, val: bool) {
        self.dense = val
    }
    fn dtstart(&mut self, val: f64) {
        self.dtstart = val
    }
    fn dtmax(&mut self, val: f64) {
        self.dtmax = val;
    }
    fn max_steps(&mut self, val: usize) {
        self.max_steps = val;
    }
}

impl Rodas {
    pub fn modern_pred(&mut self, val: bool) {
        self.modern_pred = val;
    }
    pub fn safe(&mut self, val: f64) {
        self.safe = val;
    }
    pub fn facr(&mut self, val: f64) {
        self.facr = val;
    }
    pub fn facl(&mut self, val: f64) {
        self.facl = val;
    }
    pub fn quot1(&mut self, val: f64) {
        self.quot1 = val;
    }
    pub fn quot2(&mut self, val: f64) {
        self.quot2 = val;
    }
    pub fn thet(&mut self, val: f64) {
        self.thet = val;
    }
    pub fn hess(&mut self, val: bool) {
        self.hess = val;
    }
}
