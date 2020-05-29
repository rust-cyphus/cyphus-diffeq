use crate::de_stats::DEStatistics;
use ndarray::prelude::*;

/// Common options for differential equations
pub struct DEIntegratorOptions {
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
}

pub struct ODEIntegrator<F, J>
where
    F: Fn(ArrayViewMut1<f64>, ArrayView1<f64>, f64),
    J: Fn(ArrayViewMut2<f64>, ArrayView1<f64>, f64),
{
    /// RHS of the ODE
    pub dudt: F,
    /// Jacobian of the RHS of ODE
    pub dfdu: J,
    /// Mass-matrix
    pub mass_matrix: Array2<f64>,
    /// Current solution vector
    pub u: Array1<f64>,
    /// Current time
    pub t: f64,
    /// Current step size
    pub dt: f64,
    /// Previous solution vector
    pub uprev: Array1<f64>,
    /// Previous time
    pub tprev: f64,
    /// Previous step size
    pub dtprev: f64,
    /// Options of the integrator
    pub opts: DEIntegratorOptions,
    /// Statistics of the DE integration
    pub stats: DEStatistics,
    /// Flag specifying if analytic Jacobian was supplied
    pub(crate) analytical_jac: bool,
}
