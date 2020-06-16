use super::Radau5;
use crate::ode::*;

impl Radau5 {
    pub(super) fn step_size_control<T: OdeFunction>(integrator: &mut OdeIntegrator<T, Self>) {
        let n = integrator.u.len();
        integrator.cache.decompose = true;

        // computation of dtnew -- require 0.2 <= dtnew/integrator.dt <= 8.
        let fac = integrator.opts.safe.min(
            integrator.cache.cfac
                / (integrator.cache.newt + 2 * integrator.opts.max_newt_iter) as f64,
        );
        let mut quot = (integrator.opts.facr).max(
            integrator
                .opts
                .facl
                .min(integrator.cache.err.powf(0.25) / fac),
        );
        let mut dtnew = integrator.dt / quot;

        //  is the error small enough ?
        if integrator.cache.err < 1.0 {
            // step is accepted
            integrator.cache.first = false;
            integrator.stats.accepts += 1;
            if integrator.opts.modern_pred {
                // predictive controller of Gustafsson
                if integrator.stats.accepts > 1 {
                    let mut facgus = (integrator.cache.dtacc / integrator.dt)
                        * (integrator.cache.err.powi(2) / integrator.cache.erracc).powf(0.25)
                        / integrator.opts.safe;
                    facgus = integrator.opts.facr.max(integrator.opts.facl.min(facgus));
                    quot = quot.max(facgus);
                    dtnew = integrator.dt / quot;
                }
                integrator.cache.dtacc = integrator.dt;
                integrator.cache.erracc = 1e-2f64.max(integrator.cache.err);
            }
            integrator.tprev = integrator.t;
            integrator.dtprev = integrator.dt;
            integrator.t += integrator.dt;

            let mut ak: f64;
            let mut acont3: f64;
            for i in 0..n {
                integrator.u[i] += integrator.cache.z3[i];
                integrator.cache.cont[i + n] =
                    (integrator.cache.z2[i] - integrator.cache.z3[i]) / Radau5::C2M1;
                ak = (integrator.cache.z1[i] - integrator.cache.z2[i]) / Radau5::C1MC2;
                acont3 = integrator.cache.z1[i] / Radau5::C1;
                acont3 = (ak - acont3) / Radau5::C2;
                integrator.cache.cont[i + 2 * n] =
                    (ak - integrator.cache.cont[i + n]) / Radau5::C1M1;
                integrator.cache.cont[i + 3 * n] = integrator.cache.cont[i + 2 * n] - acont3;
            }

            for i in 0..n {
                integrator.cache.scal[i] =
                    integrator.opts.abstol + integrator.opts.reltol * integrator.u[i].abs();
            }

            if integrator.opts.dense {
                for i in 0..n {
                    integrator.cache.cont[i] = integrator.u[i];
                }
            }

            integrator.sol.ts.push(integrator.t);
            integrator.sol.us.push(integrator.u.clone());

            integrator.cache.caljac = false;
            if integrator.cache.last {
                integrator.dt = integrator.cache.dtopt;
                integrator.sol.retcode = OdeRetCode::Success;
                return;
            }
            integrator.func.dudt(
                integrator.cache.u0.view_mut(),
                integrator.u.view(),
                integrator.t,
            );
            integrator.stats.function_evals += 1;

            dtnew = integrator.tdir * dtnew.abs().min(integrator.opts.dtmax);
            integrator.cache.dtopt = integrator.dt.min(dtnew);
            if integrator.cache.reject {
                dtnew = integrator.tdir * dtnew.abs().min(integrator.dt.abs());
            }
            integrator.cache.reject = false;
            if (integrator.t + dtnew / integrator.opts.quot1 - integrator.tfinal) * integrator.tdir
                >= 0.0
            {
                integrator.dt = integrator.tfinal - integrator.t;
                integrator.cache.last = true;
            } else {
                let qt = dtnew / integrator.dt;
                integrator.cache.dtfac = integrator.dt;
                if (integrator.cache.theta <= integrator.opts.theta)
                    && (qt >= integrator.opts.quot1)
                    && (qt <= integrator.opts.quot2)
                {
                    integrator.cache.decompose = false;
                    return;
                }
                integrator.dt = dtnew;
            }
            integrator.cache.dtfac = integrator.dt;
            if integrator.cache.theta > integrator.opts.theta {
                integrator.func.dfdu(
                    integrator.cache.dfdu.view_mut(),
                    integrator.u.view(),
                    integrator.t,
                );
            }
        } else {
            // step is rejected
            integrator.cache.reject = true;
            integrator.cache.last = false;
            if integrator.cache.first {
                integrator.dt *= 0.1;
                integrator.cache.dtfac = 0.1;
            } else {
                integrator.cache.dtfac = dtnew / integrator.dt;
                integrator.dt = dtnew;
            }
            if integrator.stats.accepts >= 1 {
                integrator.stats.rejects += 1;
            }
            if !integrator.cache.caljac {
                integrator.func.dfdu(
                    integrator.cache.dfdu.view_mut(),
                    integrator.u.view(),
                    integrator.t,
                );
            }
        }
    }
}
