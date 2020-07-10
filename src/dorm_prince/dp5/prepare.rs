use super::DormandPrince5;
use crate::ode::OdeIntegrator;

impl DormandPrince5 {
    /// Prepare the integrator and integrator.cache for the next DormandPrince5 step.
    #[allow(dead_code)]
    pub(super) fn prepare_next_step<Params>(
        t_err: f64,
        integrator: &mut OdeIntegrator<Params, DormandPrince5>,
    ) {
        let beta = integrator.opts.beta;
        let expo1 = 0.2 - beta.powf(0.75);
        let safe = integrator.opts.safe;
        let fac1 = 1.0 / integrator.opts.facl;
        let fac2 = 1.0 / integrator.opts.facr;

        let n = integrator.u.len();
        // computation of hnew
        let fac11 = t_err.powf(expo1);
        // Lund-stabilization
        let mut fac = fac11 / integrator.cache.facold.powf(beta);
        // we require minNextPrevStepRatio <= hnew/h <= m_maxNextPrevStepRatio
        fac = fac2.max(fac1.min(fac / safe));
        let mut dtnew = integrator.dt / fac;

        if t_err <= 1.0 {
            /* step accepted */

            integrator.cache.facold = t_err.max(1e-4);
            integrator.stats.accepts += 1;

            /* stiffness detection */
            if integrator.stats.accepts % integrator.opts.max_stiff == 0
                || integrator.cache.n_stiff > 0
            {
                let mut stnum = 0.0;
                let mut stden = 0.0;
                for i in 0..n {
                    let mut sqr = integrator.cache.k2[i] - integrator.cache.k6[i];
                    stnum += sqr * sqr;
                    sqr = integrator.cache.unew[i] - integrator.cache.ustiff[i];
                    stden += sqr * sqr;
                }
                if stden > 0.0 {
                    integrator.cache.dtlamb = integrator.dt * (stnum / stden).sqrt();
                }
                if integrator.cache.dtlamb > 3.25 {
                    integrator.cache.n_nonstiff = 0;
                    integrator.cache.n_stiff += 1;
                    if integrator.cache.n_stiff == 15 {
                        //throw DormandPrince5Stiff(integrator.t);
                    }
                } else {
                    integrator.cache.n_nonstiff += 1;
                    if integrator.cache.n_nonstiff == 6 {
                        integrator.cache.n_stiff = 0;
                    }
                }
            }
            if integrator.opts.dense {
                DormandPrince5::prepare_dense(integrator);
            }

            integrator.cache.du.assign(&integrator.cache.dunew);
            integrator.uprev.assign(&integrator.u);
            integrator.u.assign(&integrator.cache.unew);
            integrator.tprev = integrator.t;
            integrator.t += integrator.dt;

            if dtnew.abs() > integrator.opts.dtmax {
                dtnew = integrator.tdir * integrator.opts.dtmax;
            }
            if integrator.cache.reject {
                dtnew = integrator.tdir * dtnew.abs().min(integrator.dt.abs());
            }

            integrator.cache.reject = false;
        } else {
            /* step rejected */
            dtnew = integrator.dt / fac1.min(fac11 / safe);
            integrator.cache.reject = true;
            if integrator.stats.accepts >= 1 {
                integrator.stats.rejects += 1;
            }
            integrator.cache.last = false;
        }

        integrator.dtprev = integrator.dt;
        integrator.dt = dtnew;
    }
}
