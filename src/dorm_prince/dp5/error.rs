use super::DormandPrince5;
use crate::ode::{OdeFunction, OdeIntegrator};

impl DormandPrince5 {
    #[allow(dead_code)]
    pub(super) fn error<T: OdeFunction>(integrator: &OdeIntegrator<T, DormandPrince5>) -> f64 {
        let mut err = 0.0;
        let n = integrator.u.len();
        for i in 0..n {
            let sk = integrator.opts.abstol
                + integrator.opts.reltol
                    * (integrator.u[i].abs().max(integrator.cache.unew[i].abs()));
            let sqr = integrator.cache.uerr[i] / sk;
            err += sqr * sqr;
        }
        (err / n as f64).sqrt()
    }
}
