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
use super::options::OdeIntegratorOpts;
use super::solution::OdeSolution;
use super::statistics::OdeStatistics;
use ndarray::prelude::*;

/// Structure to hold all information needed to integrate an ODE.
pub struct OdeIntegrator<'a, Params: 'a, Alg: OdeAlgorithm + 'a> {
    /// Function structure representing the RHS of the ODE.
    pub dudt: &'a dyn Fn(ArrayViewMut1<f64>, ArrayView1<f64>, f64, &Params),
    /// Jacobian w.r.t. t of RHS of ODE
    pub(crate) dfdt: Option<&'a dyn Fn(ArrayViewMut1<f64>, ArrayView1<f64>, f64, &Params)>,
    /// Jacobian w.r.t. u of RHS of ODE
    pub(crate) dfdu: Option<&'a dyn Fn(ArrayViewMut2<f64>, ArrayView1<f64>, f64, &Params)>,
    /// Parameters of the ODE
    pub params: Params,
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
    /// The algorithm being used to solve the ODE.
    pub alg: Alg,
    /// Callback function to mutate state of integrator after each step.
    pub callback: Option<&'a dyn Fn(&mut Self)>,
    /// Cache associated with the algorithm
    pub(crate) cache: Alg::Cache,
}

/// Struct for building an OdeIntegrator
pub struct OdeIntegratorBuilder<'a, Params, Alg: OdeAlgorithm> {
    /// Function structure representing the RHS of the ODE.
    pub(crate) dudt: &'a dyn Fn(ArrayViewMut1<f64>, ArrayView1<f64>, f64, &Params),
    /// Jacobian w.r.t. t of RHS of ODE
    pub(crate) dfdt: Option<&'a dyn Fn(ArrayViewMut1<f64>, ArrayView1<f64>, f64, &Params)>,
    /// Jacobian w.r.t. u of RHS of ODE
    pub(crate) dfdu: Option<&'a dyn Fn(ArrayViewMut2<f64>, ArrayView1<f64>, f64, &Params)>,
    /// Parameters of the ODE
    pub params: Params,
    /// Current solution vector
    pub u: Array1<f64>,
    /// Current time
    pub t: f64,
    /// Current step size
    pub dt: f64,
    /// Final time value
    pub tfinal: f64,
    /// Options
    pub opts: OdeIntegratorOpts,
    /// Algorithm
    pub alg: Alg,
    /// Callback function to mutate state of integrator after each step.
    pub callback: Option<&'a dyn Fn(&mut OdeIntegrator<'a, Params, Alg>)>,
}

impl<'a, Params, Alg: OdeAlgorithm> OdeIntegratorBuilder<'a, Params, Alg> {
    pub fn default(
        dudt: &'a dyn Fn(ArrayViewMut1<f64>, ArrayView1<f64>, f64, &Params),
        uinit: Array1<f64>,
        tspan: (f64, f64),
        alg: Alg,
        params: Params,
    ) -> OdeIntegratorBuilder<'a, Params, Alg> {
        OdeIntegratorBuilder {
            dudt,
            dfdt: None,
            dfdu: None,
            params,
            u: uinit.clone(),
            t: tspan.0,
            dt: 1e-6,
            tfinal: tspan.1,
            opts: Alg::default_opts(),
            alg,
            callback: None,
        }
    }
    pub fn dfdu(
        mut self,
        jac: &'a dyn Fn(ArrayViewMut2<f64>, ArrayView1<f64>, f64, &Params),
    ) -> Self {
        self.dfdu = Some(jac);
        self
    }
    pub fn dfdt(
        mut self,
        jac: &'a dyn Fn(ArrayViewMut1<f64>, ArrayView1<f64>, f64, &Params),
    ) -> Self {
        self.dfdt = Some(jac);
        self
    }
    pub fn reltol(mut self, val: f64) -> Self {
        self.opts.reltol = val;
        self
    }
    pub fn abstol(mut self, val: f64) -> Self {
        self.opts.abstol = val;
        self
    }
    pub fn dense(mut self, val: bool) -> Self {
        self.opts.dense = val;
        self
    }
    pub fn dtstart(mut self, val: f64) -> Self {
        self.opts.dtstart = val;
        self
    }
    pub fn dtmax(mut self, val: f64) -> Self {
        self.opts.dtmax = val;
        self
    }
    pub fn max_steps(mut self, val: usize) -> Self {
        self.opts.max_steps = val;
        self
    }
    pub fn max_newt_iter(mut self, val: usize) -> Self {
        self.opts.max_newt_iter = val;
        self
    }
    pub fn max_stiff(mut self, val: usize) -> Self {
        self.opts.max_stiff = val;
        self
    }
    pub fn modern_pred(mut self, val: bool) -> Self {
        self.opts.modern_pred = val;
        self
    }
    pub fn safe(mut self, val: f64) -> Self {
        self.opts.safe = val;
        self
    }
    pub fn facr(mut self, val: f64) -> Self {
        self.opts.facr = val;
        self
    }
    pub fn facl(mut self, val: f64) -> Self {
        self.opts.facl = val;
        self
    }
    pub fn quot1(mut self, val: f64) -> Self {
        self.opts.quot1 = val;
        self
    }
    pub fn quot2(mut self, val: f64) -> Self {
        self.opts.quot2 = val;
        self
    }
    pub fn beta(mut self, val: f64) -> Self {
        self.opts.beta = val;
        self
    }
    pub fn fnewt(mut self, val: f64) -> Self {
        self.opts.fnewt = val;
        self
    }
    pub fn use_ext_col(mut self, val: bool) -> Self {
        self.opts.use_ext_col = val;
        self
    }
    pub fn hess(mut self, val: bool) -> Self {
        self.opts.hess = val;
        self
    }
    pub fn callback(mut self, cb: &'a dyn Fn(&mut OdeIntegrator<'a, Params, Alg>)) -> Self {
        self.callback = Some(cb);
        self
    }
    pub fn build(mut self) -> OdeIntegrator<'a, Params, Alg> {
        let cache = Alg::new_cache(&mut self);

        let mut sol = OdeSolution::new();
        sol.ts.push(self.t);
        sol.us.push(self.u.clone());

        OdeIntegrator {
            dudt: self.dudt,
            dfdt: self.dfdt,
            dfdu: self.dfdu,
            params: self.params,
            u: self.u.clone(),
            t: self.t,
            dt: self.dt,
            uprev: self.u,
            tprev: self.t,
            dtprev: self.dt,
            tdir: 1f64.copysign(self.tfinal - self.t),
            tfinal: self.tfinal,
            opts: self.opts,
            stats: OdeStatistics::new(),
            alg: self.alg,
            callback: self.callback,
            sol,
            cache,
        }
    }
}

impl<'a, Params, Alg: OdeAlgorithm> OdeIntegrator<'a, Params, Alg> {
    /// Step the ODE to the next state. Returns solution if finished and
    /// None otherwise.
    pub fn step(&mut self) -> Option<usize> {
        Alg::step(self);
        match self.callback {
            Some(f) => f(self),
            None => {}
        }
        if self.sol.retcode == super::code::OdeRetCode::Continue {
            Some(self.stats.steps)
        } else {
            None
        }
    }
    /// Solve the ODE
    pub fn integrate(&mut self) {
        while let Some(_i) = self.step() {}
    }
}

/// Consuming iterator for OdeIntegrator.
///
/// # Examples
/// ```
/// struct HO(w: f64);
/// let dudt = |mut du: ArrayViewMut1<f64>, u: ArrayView1<f64>, t: f64, p: &HO|{
///     du[0] = u[1];
///     du[1] = -p.w * u[0];
/// }
/// let uinit = array![0.0, 1.0];
/// let tspan = (0.0, 1.0);
/// let integrator = OdeIntegratorBuilder::default(
///     &dudt,
///     uinit,
///     tspan,
///     DormandPrince5,
///     HO{w:1.0}
/// ).reltol(1e-7)
///     .abstol(1e-7)
///     .build();
/// // Iterate
/// for (t, u) in integrator.into_iter() {
///     assert!((u[0] - t.sin()).abs() < 1e-6);
///     assert!((u[1] - t.cos()).abs() < 1e-6);
/// }
/// ```
pub struct OdeIntegratorIterator<'a, Params, Alg: OdeAlgorithm> {
    pub integrator: OdeIntegrator<'a, Params, Alg>,
}

impl<'a, Params, Alg: OdeAlgorithm> IntoIterator for OdeIntegrator<'a, Params, Alg> {
    type Item = (f64, Array1<f64>);
    type IntoIter = OdeIntegratorIterator<'a, Params, Alg>;

    fn into_iter(self) -> Self::IntoIter {
        OdeIntegratorIterator { integrator: self }
    }
}

impl<'a, Params, Alg: OdeAlgorithm> Iterator for OdeIntegratorIterator<'a, Params, Alg> {
    type Item = (f64, Array1<f64>);

    fn next(&mut self) -> Option<(f64, Array1<f64>)> {
        let res = self.integrator.step();
        match res {
            Some(_i) => Some((self.integrator.t, self.integrator.u.clone())),
            None => None,
        }
    }
}

/// Non-consuming, mutable iterator for OdeIntegrator.
///
/// # Examples
/// ```
/// struct HO(w: f64);
/// let dudt = |mut du: ArrayViewMut1<f64>, u: ArrayView1<f64>, t: f64, p: &HO|{
///     du[0] = u[1];
///     du[1] = -p.w * u[0];
/// }
/// let uinit = array![0.0, 1.0];
/// let tspan = (0.0, 1.0);
/// let mut integrator = OdeIntegratorBuilder::default(
///     &dudt,
///     uinit,
///     tspan,
///     DormandPrince5,
///     HO{w:1.0}
/// ).reltol(1e-7)
///     .abstol(1e-7)
///     .build();
/// // Iterate
/// for (t, u) in (&mut integrator).into_iter() {
///     assert!((u[0] - t.sin()).abs() < 1e-6);
///     assert!((u[1] - t.cos()).abs() < 1e-6);
/// }
/// ```
pub struct OdeIntegratorMutIterator<'a, Params, Alg: OdeAlgorithm> {
    pub integrator: &'a mut OdeIntegrator<'a, Params, Alg>,
}

impl<'a, Params, Alg: OdeAlgorithm> IntoIterator for &'a mut OdeIntegrator<'a, Params, Alg> {
    type Item = (f64, Array1<f64>);
    type IntoIter = OdeIntegratorMutIterator<'a, Params, Alg>;

    fn into_iter(self) -> Self::IntoIter {
        OdeIntegratorMutIterator { integrator: self }
    }
}

impl<'a, Params, Alg: OdeAlgorithm> Iterator for OdeIntegratorMutIterator<'a, Params, Alg> {
    type Item = (f64, Array1<f64>);

    fn next(&mut self) -> Option<(f64, Array1<f64>)> {
        let res = (*self).integrator.step();
        match res {
            Some(_i) => Some((self.integrator.t, self.integrator.u.clone())),
            None => None,
        }
    }
}
