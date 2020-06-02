use crate::ode_options::OdeIntegratorOpts;
use crate::ode_prob::OdeProblem;
use crate::ode_stats::OdeStats;
use ndarray::prelude::*;

pub struct ODEIntegrator<Alg> {
    /// RHS of the ODE
    pub prob: OdeProblem,
    /// Algorithm to use to integrate ordinary differential equation.
    pub alg: Alg,
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
    pub opts: OdeIntegratorOpts,
    /// Statistics of the DE integration
    pub stats: OdeStats,
}

#[derive(Clone)]
pub struct ODEIntegratorBuilder<Alg> {
    /// RHS of the ODE
    pub prob: OdeProblem,
    /// Algorithm to use to integrate ordinary differential equation.
    pub alg: Alg,
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
    pub opts: OdeIntegratorOpts,
    /// Statistics of the DE integration
    pub stats: OdeStats,
}

impl<Alg> ODEIntegratorBuilder<Alg> {
    /// Construct an ODEIntegratorBuilder with default parameters.
    pub fn default(prob: OdeProblem, alg: Alg) -> ODEIntegratorBuilder<Alg> {
        let mut opts = OdeIntegratorOpts::default();
        let u = prob.uinit.clone();
        let t = prob.tspan.0;

        ODEIntegratorBuilder {
            prob,
            alg,
            u: u.clone(),
            t,
            dt: 1e-6,
            uprev: u.clone(),
            tprev: tspan.0,
            dtprev: 1e-6,
            opts,
            stats: OdeStats::new(),
        }
    }
    /// Specify the relative tolerance.
    pub fn reltol(mut self, val: f64) -> ODEIntegratorBuilder<Alg> {
        self.opts.reltol = Some(val);
        self
    }
    /// Specify the absolute tolerance.
    pub fn abstol(mut self, val: f64) -> ODEIntegratorBuilder<Alg> {
        self.opts.abstol = Some(val);
        self
    }
    /// Specify whether dense output should be produced.
    pub fn dense(mut self, val: bool) -> ODEIntegratorBuilder<Alg> {
        self.opts.dense = Some(val);
        self
    }
    /// Set the starting step size.
    pub fn dtstart(mut self, val: f64) -> ODEIntegratorBuilder<Alg> {
        self.opts.dtstart = Some(val);
        self
    }
    /// Set the maximum step size.
    pub fn dtmax(mut self, val: f64) -> ODEIntegratorBuilder<Alg> {
        self.opts.dtmax = Some(val);
        self
    }
    /// Set the maximum number of steps allowed number of steps.
    pub fn max_num_steps(mut self, val: usize) -> ODEIntegratorBuilder<Alg> {
        self.opts.max_num_steps = Some(val);
        self
    }
    /// Set the mass-matrix.
    pub fn mass_matrix(mut self, mass_matrix: ArrayView2<f64>) -> ODEIntegratorBuilder<Alg> {
        self.mass_matrix.unwrap().assign(&mass_matrix);
        self
    }
    /// Finish the building of an integrator and construct integrator.
    pub fn build(self) -> ODEIntegrator<Alg> {
        ODEIntegrator {
            dudt: self.dudt,
            dfdu: self.dfdu,
            mass_matrix: self.mass_matrix.clone(),
            u: self.u.clone(),
            t: self.t,
            dt: self.dt,
            uprev: self.uprev.clone(),
            tprev: self.tprev,
            dtprev: self.dtprev,
            opts: self.opts.clone(),
            stats: self.stats.clone(),
            analytical_jac: self.analytical_jac,
            tfinal: self.tfinal,
        }
    }
}
