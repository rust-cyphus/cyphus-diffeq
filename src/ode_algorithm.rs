// use crate::ode_prob::OdeProblem;
// use crate::ode_solution::OdeSolution;

/// The Radua5 algorithm.
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
    pub(crate) max_num_steps: usize,
    ///(crate) Maximum number of allowed newton iterations.
    pub(crate) max_num_newt_iter: usize,
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
    ///(crate) The amount to decrease the timestep by if the Newton iterations of an
    ///(crate) implicit method fail.
    pub(crate) fnewt: f64,
    ///(crate) If true, the extrapolated collocation solution is taken as the starting
    ///(crate) value of the Newton iteration.
    pub(crate) use_ext_col: bool,
    ///(crate) If true, the Jacobian will be converted into Hessenberg form.
    pub(crate) hess: bool,
}
/// The Rodas algorithm.
pub struct Rodas {}

pub(crate) trait OdeAlgorithm {
    /// Create algorithm with default parameters.
    fn default() -> Self;
    /// Set the relative tolerance.
    fn reltol(&mut self, val: f64);
    /// Set the absolute tolerance.
    fn abstol(&mut self, val: f64);
    /// Declare if dense output should be constructed.
    fn dense(&mut self, val: bool);
    /// Set the initial step size.
    fn dtstart(&mut self, val: f64);
    /// Set the maximum step size
    fn dtmax(&mut self, val: f64);
    /// Set the maximum number of allowed steps.
    fn max_steps(&mut self, val: usize);
    // fn integrate(&mut self, prob: OdeProblem) -> OdeSolution<Self>;
}
