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
//! let mut integrator = OdeProblemBuilder::default(prob, DormandPrince5).build();
//! // Step the integrator (returns solution if done, else None)
//! integrator.step();
//! // Integrate
//! integrator.integrate();
//! ```

use super::algorithm::OdeAlgorithm;
use super::function::OdeFunction;
use super::options::OdeIntegratorOpts;
use super::problem::OdeProblem;
use super::solution::OdeSolution;
use super::statistics::OdeStatistics;
use ndarray::prelude::*;

/// Light-weight struct used to keep track of the state of the ODE throughout
/// integration.
pub struct OdeIntegrator<T: OdeFunction, Alg: OdeAlgorithm> {
    /// Function structure representing the RHS of the ODE.
    pub func: T,
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
        Alg::step(self);
        if self.sol.retcode == super::code::OdeRetCode::Continue {
            Some(self.stats.steps)
        } else {
            None
        }
    }
    /// Solve the ODE
    pub fn solve(&mut self) {
        while let Some(_i) = self.step() {}
    }
}

// Consuming iterator

pub struct OdeIntegratorIterator<T: OdeFunction, Alg: OdeAlgorithm> {
    pub integrator: OdeIntegrator<T, Alg>,
}

impl<T: OdeFunction, Alg: OdeAlgorithm> IntoIterator for OdeIntegrator<T, Alg> {
    type Item = (f64, Array1<f64>);
    type IntoIter = OdeIntegratorIterator<T, Alg>;

    fn into_iter(self) -> Self::IntoIter {
        OdeIntegratorIterator { integrator: self }
    }
}

impl<T: OdeFunction, Alg: OdeAlgorithm> Iterator for OdeIntegratorIterator<T, Alg> {
    type Item = (f64, Array1<f64>);

    fn next(&mut self) -> Option<(f64, Array1<f64>)> {
        let res = self.integrator.step();
        match res {
            Some(_i) => Some((self.integrator.t, self.integrator.u.clone())),
            None => None,
        }
    }
}

pub struct OdeIntegratorMutIterator<'a, T: OdeFunction, Alg: OdeAlgorithm> {
    pub integrator: &'a mut OdeIntegrator<T, Alg>,
}

impl<'a, T: OdeFunction, Alg: OdeAlgorithm> IntoIterator for &'a mut OdeIntegrator<T, Alg> {
    type Item = (f64, Array1<f64>);
    type IntoIter = OdeIntegratorMutIterator<'a, T, Alg>;

    fn into_iter(self) -> Self::IntoIter {
        OdeIntegratorMutIterator { integrator: self }
    }
}

impl<'a, T: OdeFunction, Alg: OdeAlgorithm> Iterator for OdeIntegratorMutIterator<'a, T, Alg> {
    type Item = (f64, Array1<f64>);

    fn next(&mut self) -> Option<(f64, Array1<f64>)> {
        let res = (*self).integrator.step();
        match res {
            Some(_i) => Some((self.integrator.t, self.integrator.u.clone())),
            None => None,
        }
    }
}

// Non-consuming iterator

/// Light-weight struct used to keep track of the state of the ODE throughout
/// integration.
pub struct OdeIntegratorBuilder<T: OdeFunction, Alg: OdeAlgorithm> {
    /// Function structure representing the RHS of the ODE.
    pub func: T,
    /// Current solution vector
    pub u: Array1<f64>,
    /// Current time
    pub t: f64,
    /// Current step size
    pub dt: f64,
    /// Direction of integration.
    pub tdir: f64,
    /// Final time value
    pub tfinal: f64,
    /// Options
    pub opts: OdeIntegratorOpts,
    /// Algorithm
    pub alg: Alg,
}

impl<T: OdeFunction, Alg: OdeAlgorithm> OdeIntegratorBuilder<T, Alg> {
    pub fn default(prob: OdeProblem<T>, alg: Alg) -> OdeIntegratorBuilder<T, Alg> {
        let mut opts = Alg::default_opts();
        opts.dtmax = prob.tspan.1 - prob.tspan.0;

        OdeIntegratorBuilder::<T, Alg> {
            func: prob.func,
            u: prob.uinit.clone(),
            t: prob.tspan.0,
            dt: opts.dtstart,
            tdir: 1f64.copysign(prob.tspan.1 - prob.tspan.0),
            tfinal: prob.tspan.1,
            opts,
            alg,
        }
    }
    pub fn reltol(mut self, val: f64) -> OdeIntegratorBuilder<T, Alg> {
        self.opts.reltol = val;
        self
    }
    pub fn abstol(mut self, val: f64) -> OdeIntegratorBuilder<T, Alg> {
        self.opts.abstol = val;
        self
    }
    pub fn dense(mut self, val: bool) -> OdeIntegratorBuilder<T, Alg> {
        self.opts.dense = val;
        self
    }
    pub fn dtstart(mut self, val: f64) -> OdeIntegratorBuilder<T, Alg> {
        self.opts.dtstart = val;
        self
    }
    pub fn dtmax(mut self, val: f64) -> OdeIntegratorBuilder<T, Alg> {
        self.opts.dtmax = val;
        self
    }
    pub fn max_steps(mut self, val: usize) -> OdeIntegratorBuilder<T, Alg> {
        self.opts.max_steps = val;
        self
    }
    pub fn max_newt_iter(mut self, val: usize) -> OdeIntegratorBuilder<T, Alg> {
        self.opts.max_newt_iter = val;
        self
    }
    pub fn max_stiff(mut self, val: usize) -> OdeIntegratorBuilder<T, Alg> {
        self.opts.max_stiff = val;
        self
    }
    pub fn modern_pred(mut self, val: bool) -> OdeIntegratorBuilder<T, Alg> {
        self.opts.modern_pred = val;
        self
    }
    pub fn safe(mut self, val: f64) -> OdeIntegratorBuilder<T, Alg> {
        self.opts.safe = val;
        self
    }
    pub fn facr(mut self, val: f64) -> OdeIntegratorBuilder<T, Alg> {
        self.opts.facr = val;
        self
    }
    pub fn facl(mut self, val: f64) -> OdeIntegratorBuilder<T, Alg> {
        self.opts.facl = val;
        self
    }
    pub fn quot1(mut self, val: f64) -> OdeIntegratorBuilder<T, Alg> {
        self.opts.quot1 = val;
        self
    }
    pub fn quot2(mut self, val: f64) -> OdeIntegratorBuilder<T, Alg> {
        self.opts.quot2 = val;
        self
    }
    pub fn beta(mut self, val: f64) -> OdeIntegratorBuilder<T, Alg> {
        self.opts.beta = val;
        self
    }
    pub fn fnewt(mut self, val: f64) -> OdeIntegratorBuilder<T, Alg> {
        self.opts.fnewt = val;
        self
    }
    pub fn use_ext_col(mut self, val: bool) -> OdeIntegratorBuilder<T, Alg> {
        self.opts.use_ext_col = val;
        self
    }
    pub fn hess(mut self, val: bool) -> OdeIntegratorBuilder<T, Alg> {
        self.opts.hess = val;
        self
    }
    pub fn build(mut self) -> OdeIntegrator<T, Alg> {
        let cache = Alg::new_cache(&mut self);

        let mut sol = OdeSolution::new();
        sol.ts.push(self.t);
        sol.us.push(self.u.clone());

        OdeIntegrator {
            func: self.func,
            u: self.u.clone(),
            t: self.t,
            dt: self.dt,
            uprev: self.u,
            tprev: self.t,
            dtprev: self.dt,
            tdir: self.tdir,
            tfinal: self.tfinal,
            opts: self.opts,
            stats: OdeStatistics::new(),
            sol,
            cache,
        }
    }
}
