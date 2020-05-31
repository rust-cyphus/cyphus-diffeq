/// Common options for differential equations
#[derive(Clone)]
pub(crate) struct OdeIntegratorOpts {
    /// Relative tolerance
    pub reltol: f64,
    /// Absolute tolerance
    pub abstol: f64,
    /// Flag specifying if dense output is requested
    pub dense: bool,
    /// Initial step value
    pub dtstart: f64,
    /// Maximum allowed step size
    pub dtmax: f64,
    /// Maximum number of allowed steps
    pub max_num_steps: usize,
    /// Maximum number of allowed newton iterations.
    pub max_num_newt_iter: usize,
    /// If true, use modern predictive controller (Gustafsson).
    pub modern_pred: bool,
    /// Safety factor used in adaptive step selection.
    pub safe: f64,
    /// Gustafsson step control factor. Restricts dtnew/dtold <= 1/facr.
    pub facr: f64,
    /// Gustafsson step control factor. Restricts dtnew/dtold >= 1/facl.
    pub facl: f64,
    /// Minimum allowed ratio of dtnew/dtold such that dt will be held constant.
    pub quot1: f64,
    /// Maximum allowed ratio of dtnew/dtold such that dt will be held constant.
    pub quot2: f64,
    /// The amount to decrease the timestep by if the Newton iterations of an
    /// implicit method fail.
    pub fnewt: f64,
    /// If true, the extrapolated collocation solution is taken as the starting
    /// value of the Newton iteration.
    pub use_ext_col: bool,
    /// If true, the Jacobian will be converted into Hessenberg form.
    pub hess: bool,
}

impl OdeIntegratorOpts {
    pub fn default() -> OdeIntegratorOpts {
        OdeIntegratorOpts {
            reltol: 0.0,
            abstol: 0.0,
            dense: false,
            dtstart: 0.0,
            dtmax: 0.0,
            max_num_steps: 0,
            max_num_newt_iter: 0,
            modern_pred: true,
            safe: 0.0,
            facr: 0.0,
            facl: 0.0,
            quot1: 0.0,
            quot2: 0.0,
            fnewt: 0.0,
            use_ext_col: true,
            hess: false,
        }
    }
}
