use super::alg::Rodas;
use super::cache::RodasCache;
use crate::ode_integrator::OdeIntegrator;
use crate::ode_prob::OdeProblem;
use crate::ode_solution::OdeSolution;
use crate::ode_stats::OdeStats;
use ndarray::prelude::*;

impl Rodas {
    pub(crate) fn integrate<F, J>(&self, prob: &OdeProblem<F, J>) -> OdeSolution
    where
        F: Fn(ArrayViewMut1<f64>, ArrayView1<f64>, f64),
        J: Fn(ArrayViewMut2<f64>, ArrayView1<f64>, f64),
    {
        let mut alg = (*self).clone();
        alg.dtmax = prob.tspan.1 - prob.tspan.0;

        let mut integrator = OdeIntegrator {
            u: prob.uinit.clone(),
            t: prob.tspan.0,
            dt: alg.dtstart,
            uprev: prob.uinit.clone(),
            tprev: prob.tspan.1,
            dtprev: alg.dtstart,
            stats: OdeStats::new(),
        };

        let mut solution = OdeSolution {
            ts: Vec::<f64>::with_capacity(200),
            us: Vec::<Array1<f64>>::with_capacity(200),
            stats: OdeStats::new(),
        };
        solution.ts.push(prob.tspan.0);
        solution.us.push(prob.uinit.clone());

        let mut cache = RodasCache::new(&alg, &prob).unwrap();

        solution
    }
}
