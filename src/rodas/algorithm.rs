use super::cache::RodasCache;
use super::Rodas;
use crate::ode::*;
use ndarray::prelude::*;

impl OdeAlgorithm for Rodas {
    type Cache = RodasCache;
    fn default_opts() -> OdeIntegratorOpts {
        let mut opts = OdeIntegratorOpts::new();
        opts.reltol = 1e-3;
        opts.abstol = 1e-7;
        opts.safe = 0.9;
        opts.facr = 1.0 / 6.0;
        opts.facl = 5.0;
        opts.max_steps = 100000;
        opts.modern_pred = true;
        opts
    }
    fn new_cache<Params>(integrator: &mut OdeIntegratorBuilder<Params, Self>) -> Self::Cache {
        unimplemented!()
    }
    fn step<Params>(integrator: &mut OdeIntegrator<Params, Self>) {
        unimplemented!()
    }
}
