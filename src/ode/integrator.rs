//! The `OdeIntegrator` is an object which stores all the information needed
//! to integrate an ordinary differential equation. This object is created
//! from the associated `init` method of an `OdeAlgorithm`.
//!
//! # Example
//! Generate an integrator using the DormandPrince5 algorithm.
//! ```
//! use cyphus_diffeq::prelude::*;
//! use ndarray::prelude::*;
//! let dudt = |mut du: ArrayViewMut1<f64>, u: ArrayView1<f64>, _t:f64|{
//!     du[0] = u[1];
//!     du[1] = -u[0];
//! };
//! let uinit = array![0.0, 1.0];
//! let tspan = (0.0, 1.0);
//! let prob = OdeProblemBuilder::default(dudt, uinit, tspan).build().unwrap();
//!
//! let mut integrator = DormandPrince5::init(&prob);
//! // Step the integrator (returns solution if done, else None)
//! integrator.step();
//! // Integrate
//! integrator.integrate();
//! ```

use super::algorithm::OdeAlgorithm;
use super::function::OdeFunction;
use super::options::OdeIntegratorOpts;
use super::solution::OdeSolution;
use super::statistics::OdeStatistics;
use ndarray::prelude::*;

/// Light-weight struct used to keep track of the state of the ODE throughout
/// integration.
pub struct OdeIntegrator<T: OdeFunction, Alg: OdeAlgorithm> {
    /// Function structure representing the RHS of the ODE.
    pub func: T,
    /// Mass matrix for DAE.
    pub mass_matrix: Option<Array2<f64>>,
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
    /// Direction of integration.
    pub tdir: f64,
    /// Final time value
    pub tfinal: f64,
    /// Options
    pub opts: OdeIntegratorOpts,
    /// Statistics
    pub stats: OdeStatistics,
    /// Solution object
    pub sol: OdeSolution,
    /// Cache associated with the algorithm
    pub(crate) cache: Alg::Cache,
}

impl<T: OdeFunction, Alg: OdeAlgorithm> OdeIntegrator<T, Alg> {
    /// Step the ODE to the next state. Returns solution if finished and
    /// None otherwise.
    pub fn step(&mut self) -> Option<usize> {
        if self.sol.retcode == super::code::OdeRetCode::Continue {
            if !Alg::step(self) {
                Some(self.stats.steps)
            } else {
                None
            }
        } else {
            None
        }
    }
    /// Solve the ODE
    pub fn solve(&mut self) {
        while let Some(_i) = self.step() {}
    }
}
