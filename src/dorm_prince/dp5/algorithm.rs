use super::cache::DormandPrince5Cache;
use super::DormandPrince5;
use crate::ode::*;

use ndarray::prelude::*;

impl OdeAlgorithm for DormandPrince5 {
    type Cache = super::cache::DormandPrince5Cache;
    fn default_opts() -> OdeIntegratorOpts {
        let mut opts = OdeIntegratorOpts::new();
        opts.reltol = 1e-3;
        opts.abstol = 1e-7;
        opts.beta = 0.04;
        opts.facr = 10.0;
        opts.facl = 0.2;
        opts.max_stiff = 1000;
        opts.max_steps = 100000;
        opts
    }
    fn new_cache<T: OdeFunction>(integrator: &mut OdeIntegratorBuilder<T, Self>) -> Self::Cache {
        let n = integrator.u.len();
        let mut du = Array1::<f64>::zeros(n);

        integrator
            .func
            .dudt(du.view_mut(), integrator.u.view(), integrator.t);

        DormandPrince5Cache {
            n_stiff: 0,
            n_nonstiff: 0,
            reject: false,
            last: false,
            facold: 0.0, // previous ratio of dtnew/dt
            dtlamb: 0.0,
            k2: Array1::<f64>::zeros(n),
            k3: Array1::<f64>::zeros(n),
            k4: Array1::<f64>::zeros(n),
            k5: Array1::<f64>::zeros(n),
            k6: Array1::<f64>::zeros(n),
            rcont1: Array1::<f64>::zeros(n),
            rcont2: Array1::<f64>::zeros(n),
            rcont3: Array1::<f64>::zeros(n),
            rcont4: Array1::<f64>::zeros(n),
            rcont5: Array1::<f64>::zeros(n),
            unew: Array1::<f64>::zeros(n),
            du,
            dunew: Array1::<f64>::zeros(n),
            uerr: Array1::<f64>::zeros(n),
            ustiff: Array1::<f64>::zeros(n),
        }
    }
    /// Step the integrator using the DormandPrince5 algorithm.
    #[allow(dead_code)]
    fn step<T: OdeFunction>(integrator: &mut OdeIntegrator<T, Self>)
    where
        Self: Sized,
    {
        if (integrator.t + 1.01 * integrator.dt - integrator.tfinal) * integrator.tdir > 0.0 {
            integrator.dt = integrator.tfinal - integrator.t;
            integrator.cache.last = true;
        }
        loop {
            DormandPrince5::compute_stages(integrator);
            let err = DormandPrince5::error(integrator);
            DormandPrince5::prepare_next_step(err, integrator);
            if !integrator.cache.reject {
                break;
            }
            if integrator.dt.abs() <= integrator.t.abs() * f64::EPSILON {
                integrator.sol.retcode = OdeRetCode::DtLessThanMin;
            }
        }

        integrator.stats.steps += 1;
        integrator.sol.ts.push(integrator.t);
        integrator.sol.us.push(integrator.u.clone());

        if integrator.stats.steps > integrator.opts.max_steps {
            integrator.sol.retcode = OdeRetCode::MaxIters;
        }
        if integrator.cache.last {
            integrator.sol.retcode = OdeRetCode::Success;
        }
    }
}
