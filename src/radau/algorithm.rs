use super::cache::Radau5Cache;
use super::Radau5;
use crate::ode::*;
use ndarray::prelude::*;

impl OdeAlgorithm for Radau5 {
    type Cache = Radau5Cache;
    fn default_opts() -> OdeIntegratorOpts {
        let mut opts = OdeIntegratorOpts::new();
        opts.reltol = 1e-3;
        opts.abstol = 1e-7;
        opts.theta = 0.001;
        opts.fnewt = 0.0;
        opts.quot1 = 1.0;
        opts.quot2 = 1.2;
        opts.facr = 1.0 / 8.0;
        opts.facl = 5.0;
        opts.max_steps = 100000;
        opts.max_newt_iter = 7;
        opts.use_ext_col = true;
        opts.modern_pred = true;
        opts
    }
    fn new_cache<T: OdeFunction>(integrator: &mut OdeIntegratorBuilder<T, Self>) -> Self::Cache {
        let quot = integrator.opts.abstol / integrator.opts.reltol;
        integrator.opts.reltol = 0.1 * integrator.opts.reltol.powf(2.0 / 3.0);
        integrator.opts.abstol = integrator.opts.reltol * quot;

        let cfac = integrator.opts.safe * (1 + 2 * integrator.opts.max_newt_iter) as f64;

        let n = integrator.u.shape()[0];
        let mut u0 = Array1::<f64>::zeros(n);
        let mut scal = Array1::<f64>::zeros(n);
        let mut dfdu = Array2::<f64>::zeros((n, n));

        for i in 0..integrator.u.len() {
            scal[i] = integrator.opts.abstol + integrator.opts.reltol * integrator.u[i].abs();
        }

        // Initialize the cache
        integrator
            .func
            .dudt(u0.view_mut(), integrator.u.view(), integrator.t);

        integrator
            .func
            .dfdu(dfdu.view_mut(), integrator.u.view(), integrator.t);

        if integrator.opts.fnewt <= 0.0 {
            integrator.opts.fnewt = (10.0 * f64::EPSILON / integrator.opts.reltol)
                .max(0.03f64.min(integrator.opts.reltol.sqrt()));
        }

        Radau5Cache {
            caljac: false,
            first: true,
            last: false,
            reject: false,
            decompose: true,
            fac1: 0.0,
            alphn: 0.0,
            betan: 0.0,
            err: 0.0,
            dtopt: integrator.dt,
            faccon: 1.0,
            dtfac: integrator.dt,
            dtacc: 0.0,
            erracc: 0.0,
            thqold: 0.0,
            cfac,
            theta: 0.0,
            nsing: 0,
            newt: 0,
            u0,
            scal,
            /// Coninuous output vectors
            cont: Array1::<f64>::zeros(4 * n),
            z1: Array1::<f64>::zeros(n),
            z2: Array1::<f64>::zeros(n),
            z3: Array1::<f64>::zeros(n),
            f1: Array1::<f64>::zeros(n),
            f2: Array1::<f64>::zeros(n),
            f3: Array1::<f64>::zeros(n),
            ip1: Array1::<i32>::zeros(n),
            ip2: Array1::<i32>::zeros(n),
            e1: Array2::<f64>::zeros((n, n)),
            e2r: Array2::<f64>::zeros((n, n)),
            e2i: Array2::<f64>::zeros((n, n)),
            dfdu,
        }
    }
    fn step<T: OdeFunction>(integrator: &mut OdeIntegrator<T, Self>) {
        if integrator.t + integrator.dt * 1.0001 - integrator.tfinal * integrator.tdir >= 0.0 {
            integrator.dt = integrator.tfinal - integrator.t;
            integrator.cache.last = true;
        }
        loop {
            if integrator.cache.decompose {
                // Perform needed decompositions until we succeed.
                loop {
                    let success = Self::perform_decompositions(integrator);
                    if success {
                        break;
                    } else if integrator.sol.retcode != OdeRetCode::Continue {
                        return;
                    }
                }
            }

            integrator.stats.steps += 1;

            // Check that we haven't passed max allowed steps
            if integrator.stats.steps >= integrator.opts.max_steps {
                integrator.sol.retcode = OdeRetCode::MaxIters;
                return;
            }
            // Check that the step size isn't too small
            if 0.1 * (integrator.dt).abs() <= (integrator.t).abs() * f64::EPSILON {
                integrator.sol.retcode = OdeRetCode::DtLessThanMin;
                return;
            }

            // Solve the non-linear systems using a simplified Newton's method.
            if !Self::newton(integrator) {
                continue;
            }

            // error estimation
            Self::error_estimate(integrator);

            //Adjust the step size
            Self::step_size_control(integrator);

            if !integrator.cache.reject {
                break;
            }
        }
    }
}
