use crate::ode_stats::OdeStats;
use ndarray::prelude::Array1;

/// Light-weight struct used to keep track of the state of the ODE throughout
/// integration.
pub struct OdeIntegrator {
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
    /// Statistics of the DE integration
    pub stats: OdeStats,
}
