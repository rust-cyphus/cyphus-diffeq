use crate::ode_algorithm::OdeAlgorithm;

/// The Radua5 algorithm.
#[derive(Clone)]
pub struct Radau5 {
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
    ///(crate) Maximum number of allowed newton iterations.
    pub(crate) max_newt: usize,
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
    ///(crate) The amount to decrease the timestep by if the Newton iterations of an
    ///(crate) implicit method fail.
    pub(crate) fnewt: f64,
    ///(crate) If true, the extrapolated collocation solution is taken as the starting
    ///(crate) value of the Newton iteration.
    pub(crate) use_ext_col: bool,
    ///(crate) If true, the Jacobian will be converted into Hessenberg form.
    pub(crate) hess: bool,
}

// Declare all the constants needed for Radau5
impl Radau5 {
    pub(crate) const T11: f64 = 9.1232394870892942792e-02;
    pub(crate) const T12: f64 = -0.14125529502095420843;
    pub(crate) const T13: f64 = -3.0029194105147424492e-02;
    pub(crate) const T21: f64 = 0.24171793270710701896;
    pub(crate) const T22: f64 = 0.20412935229379993199;
    pub(crate) const T23: f64 = 0.38294211275726193779;
    pub(crate) const T31: f64 = 0.96604818261509293619;
    pub(crate) const TI11: f64 = 4.3255798900631553510;
    pub(crate) const TI12: f64 = 0.33919925181580986954;
    pub(crate) const TI13: f64 = 0.54177053993587487119;
    pub(crate) const TI21: f64 = -4.1787185915519047273;
    pub(crate) const TI22: f64 = -0.32768282076106238708;
    pub(crate) const TI23: f64 = 0.47662355450055045196;
    pub(crate) const TI31: f64 = -0.50287263494578687595;
    pub(crate) const TI32: f64 = 2.5719269498556054292;
    pub(crate) const TI33: f64 = -0.59603920482822492497;
}

impl OdeAlgorithm for Radau5 {
    fn default() -> Self {
        Radau5 {
            reltol: 1e-3,
            abstol: 1e-6,
            dense: false,
            dtstart: 1e-6,
            dtmax: 0.0,
            max_steps: 100_000,
            max_newt: 7,
            modern_pred: true,
            safe: 0.9,
            facr: 1.0 / 8.0,
            facl: 5.0,
            quot1: 1.0,
            quot2: 1.2,
            thet: 0.001,
            fnewt: 0.001,
            use_ext_col: true,
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

impl Radau5 {
    pub fn max_newt(&mut self, val: usize) {
        self.max_newt = val;
    }
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
    pub fn fnewt(&mut self, val: f64) {
        self.fnewt = val;
    }
    pub fn use_ext_col(&mut self, val: bool) {
        self.use_ext_col = val;
    }
    pub fn hess(&mut self, val: bool) {
        self.hess = val;
    }
}
