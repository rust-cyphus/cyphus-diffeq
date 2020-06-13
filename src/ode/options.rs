/// Common options for differential equations
#[derive(Clone)]
pub struct OdeIntegratorOpts {
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
    pub max_steps: usize,
    /// Maximum number of allowed newton iterations.
    pub max_newt_iter: usize,
    /// Maximum number of stiff detections allowed
    pub max_stiff: usize,
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
    /// The "beta" for stabilized step size control (see Sec.(IV.2) of Hairer
    /// and Wanner's book.)
    pub beta: f64,
    /// The amount to decrease the timestep by if the Newton iterations of an
    /// implicit method fail.
    pub fnewt: f64,
    /// If true, the extrapolated collocation solution is taken as the starting
    /// value of the Newton iteration.
    pub use_ext_col: bool,
    /// If true, the Jacobian will be converted into Hessenberg form.
    pub hess: bool,
}
