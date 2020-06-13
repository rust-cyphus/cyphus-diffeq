use super::DormandPrince5;
use crate::ode::*;

impl OdeAlgorithm for DormandPrince5 {
    type Cache = super::cache::DormandPrince5Cache;
    fn init<T: OdeFunction>(prob: OdeProblem<T>) -> OdeIntegrator<T, Self>
    where
        Self: Sized,
    {
        let mut cache = super::cache::DormandPrince5Cache::new(&prob);
        let opts = OdeIntegratorOpts {
            reltol: 1e-3,
            abstol: 1e-6,
            dense: false,
            dtstart: 1e-6,
            dtmax: prob.tspan.1 - prob.tspan.0,
            max_steps: 100000,
            beta: 0.04,
            max_newt_iter: 0,
            max_stiff: 1000,
            modern_pred: false,
            safe: 0.9,
            facr: 10.0,
            facl: 0.2,
            quot1: 0.0,
            quot2: 0.0,
            fnewt: 0.0,
            use_ext_col: false,
            hess: false,
        };
        let mut func = prob.func;
        // Initialize the solution
        let mut sol = OdeSolution::new();
        sol.ts.push(prob.tspan.0);
        sol.us.push(prob.uinit.clone());
        // Initialize the cache
        func.dudt(cache.du.view_mut(), prob.uinit.view(), prob.tspan.0);

        OdeIntegrator {
            func,
            mass_matrix: prob.mass_matrix,
            u: prob.uinit.clone(),
            t: prob.tspan.0,
            dt: 1e-6,
            uprev: prob.uinit.clone(),
            tprev: prob.tspan.0,
            dtprev: 1e-6,
            tdir: if prob.tspan.1 > prob.tspan.0 {
                1.0
            } else {
                0.0
            },
            tfinal: prob.tspan.1,
            sol,
            stats: OdeStatistics::new(),
            opts,
            cache,
        }
    }
    /// Step the integrator using the DormandPrince5 algorithm.
    #[allow(dead_code)]
    fn step<T: OdeFunction>(integrator: &mut OdeIntegrator<T, Self>) -> bool
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
                return true;
            }
        }

        integrator.stats.steps += 1;
        integrator.sol.ts.push(integrator.t);
        integrator.sol.us.push(integrator.u.clone());

        if integrator.stats.steps > integrator.opts.max_steps {
            integrator.sol.retcode = OdeRetCode::MaxIters;
            return true;
        }
        if integrator.cache.last {
            integrator.sol.retcode = OdeRetCode::Success;
            return true;
        }

        false
    }
}
