use super::Radau5;
use crate::ode::*;

impl Radau5 {
    pub(super) fn prepare_newton<Params>(integrator: &mut OdeIntegrator<Params, Self>) {
        let n = integrator.u.len();
        //  starting values for Newton iteration
        if integrator.cache.first || !integrator.opts.use_ext_col {
            for i in 0..n {
                integrator.cache.z1[i] = 0.0;
                integrator.cache.z2[i] = 0.0;
                integrator.cache.z3[i] = 0.0;
                integrator.cache.f1[i] = 0.0;
                integrator.cache.f2[i] = 0.0;
                integrator.cache.f3[i] = 0.0;
            }
        } else {
            let c3q = integrator.dt / integrator.dtprev;
            let c1q = Radau5::C1 * c3q;
            let c2q = Radau5::C2 * c3q;
            for i in 0..n {
                let ak1 = integrator.cache.cont[i + n];
                let ak2 = integrator.cache.cont[i + 2 * n];
                let ak3 = integrator.cache.cont[i + 3 * n];

                integrator.cache.z1[i] = ak3
                    .mul_add(c1q - Radau5::C1M1, ak2)
                    .mul_add(c1q - Radau5::C2M1, ak1)
                    * c1q;
                integrator.cache.z2[i] = ak3
                    .mul_add(c2q - Radau5::C1M1, ak2)
                    .mul_add(c2q - Radau5::C2M1, ak1)
                    * c2q;
                integrator.cache.z3[i] = ak3
                    .mul_add(c3q - Radau5::C1M1, ak2)
                    .mul_add(c3q - Radau5::C2M1, ak1)
                    * c3q;

                integrator.cache.f1[i] = Radau5::TI11 * integrator.cache.z1[i]
                    + Radau5::TI12 * integrator.cache.z2[i]
                    + Radau5::TI13 * integrator.cache.z3[i];
                integrator.cache.f2[i] = Radau5::TI21 * integrator.cache.z1[i]
                    + Radau5::TI22 * integrator.cache.z2[i]
                    + Radau5::TI23 * integrator.cache.z3[i];
                integrator.cache.f3[i] = Radau5::TI31 * integrator.cache.z1[i]
                    + Radau5::TI32 * integrator.cache.z2[i]
                    + Radau5::TI33 * integrator.cache.z3[i];
            }
        }
    }
    /// Solve the non-linear systems using a simplified newton iteration.
    /// Returns true if successful and false otherwise.
    pub(super) fn newton<Params>(integrator: &mut OdeIntegrator<Params, Self>) -> bool {
        let n = integrator.u.len();

        Self::prepare_newton(integrator);

        //  loop for the simplified Newton iteration
        integrator.cache.newt = 0;
        integrator.cache.faccon = integrator.cache.faccon.max(f64::EPSILON).powf(0.8);
        integrator.cache.theta = integrator.opts.theta.abs();
        let mut dyno: f64;
        let mut dynold: f64 = 0.0;
        loop {
            if integrator.cache.newt >= integrator.opts.max_newt_iter {
                integrator.dt *= 0.5;
                integrator.cache.dtfac = 0.5;
                integrator.cache.reject = true;
                integrator.cache.last = false;
                if !integrator.cache.caljac {
                    jacobian_u(
                        integrator.dudt,
                        integrator.dfdu,
                        integrator.cache.dfdu.view_mut(),
                        integrator.u.view(),
                        integrator.t,
                        &integrator.params,
                    );
                }
                return false;
            }
            // compute the right-hand side
            for i in 0..n {
                integrator.cache.cont[i] = integrator.u[i] + integrator.cache.z1[i];
            }
            (integrator.dudt)(
                integrator.cache.z1.view_mut(),
                integrator.cache.cont.view(),
                integrator.t + Radau5::C1 * integrator.dt,
                &integrator.params,
            );

            for i in 0..n {
                integrator.cache.cont[i] = integrator.u[i] + integrator.cache.z2[i];
            }
            (integrator.dudt)(
                integrator.cache.z2.view_mut(),
                integrator.cache.cont.view(),
                integrator.t + Radau5::C2 * integrator.dt,
                &integrator.params,
            );

            for i in 0..n {
                integrator.cache.cont[i] = integrator.u[i] + integrator.cache.z3[i];
            }
            (integrator.dudt)(
                integrator.cache.z3.view_mut(),
                integrator.cache.cont.view(),
                integrator.t + integrator.dt,
                &integrator.params,
            );

            integrator.stats.function_evals += 3;

            // solve the linear systems
            for i in 0..n {
                let a1 = integrator.cache.z1[i];
                let a2 = integrator.cache.z2[i];
                let a3 = integrator.cache.z3[i];
                integrator.cache.z1[i] = Radau5::TI11 * a1 + Radau5::TI12 * a2 + Radau5::TI13 * a3;
                integrator.cache.z2[i] = Radau5::TI21 * a1 + Radau5::TI22 * a2 + Radau5::TI23 * a3;
                integrator.cache.z3[i] = Radau5::TI31 * a1 + Radau5::TI32 * a2 + Radau5::TI33 * a3;
            }
            Self::linear_solve(integrator);
            integrator.stats.linear_solves += 1;
            integrator.cache.newt += 1;
            dyno = 0.0;
            let mut denom: f64;
            for i in 0..n {
                denom = integrator.cache.scal[i];
                dyno += (integrator.cache.z1[i] / denom).powi(2)
                    + (integrator.cache.z2[i] / denom).powi(2)
                    + (integrator.cache.z3[i] / denom).powi(2);
            }
            dyno = (dyno / ((3 * n) as f64)).sqrt();
            // bad convergence or number of iterations to large
            if (integrator.cache.newt > 1)
                && (integrator.cache.newt < integrator.opts.max_newt_iter)
            {
                let thq = dyno / dynold;
                integrator.cache.theta = if integrator.cache.newt == 2 {
                    thq
                } else {
                    (thq * integrator.cache.thqold).sqrt()
                };
                integrator.cache.thqold = thq;
                if integrator.cache.theta < 0.99 {
                    integrator.cache.faccon =
                        integrator.cache.theta / (1.0 - integrator.cache.theta);
                    let dyth = integrator.cache.faccon
                        * dyno
                        * integrator.cache.theta.powi(
                            (integrator.opts.max_newt_iter - 1 - integrator.cache.newt) as i32,
                        )
                        / integrator.opts.fnewt;
                    if dyth >= 1.0 {
                        let qnewt: f64 = 1e-4f64.max(20f64.min(dyth));
                        integrator.cache.dtfac = 0.8
                            * qnewt.powf(
                                -1f64
                                    / (4 + integrator.opts.max_newt_iter
                                        - 1
                                        - integrator.cache.newt)
                                        as f64,
                            );
                        integrator.dt *= integrator.cache.dtfac;
                        integrator.cache.reject = true;
                        integrator.cache.last = false;
                        if integrator.cache.caljac {
                            jacobian_u(
                                integrator.dudt,
                                integrator.dfdu,
                                integrator.cache.dfdu.view_mut(),
                                integrator.u.view(),
                                integrator.t,
                                &integrator.params,
                            );
                        }
                        return false;
                    }
                } else {
                    integrator.dt *= 0.5;
                    integrator.cache.dtfac = 0.5;
                    integrator.cache.reject = true;
                    integrator.cache.last = false;
                    if !integrator.cache.caljac {
                        jacobian_u(
                            integrator.dudt,
                            integrator.dfdu,
                            integrator.cache.dfdu.view_mut(),
                            integrator.u.view(),
                            integrator.t,
                            &integrator.params,
                        );
                    }
                    return false;
                }
            }
            dynold = dyno.max(f64::EPSILON);
            for i in 0..n {
                integrator.cache.f1[i] += integrator.cache.z1[i];
                integrator.cache.f2[i] += integrator.cache.z2[i];
                integrator.cache.f3[i] += integrator.cache.z3[i];
                integrator.cache.z1[i] = Radau5::T11 * integrator.cache.f1[i]
                    + Radau5::T12 * integrator.cache.f2[i]
                    + Radau5::T13 * integrator.cache.f3[i];
                integrator.cache.z2[i] = Radau5::T21 * integrator.cache.f1[i]
                    + Radau5::T22 * integrator.cache.f2[i]
                    + Radau5::T23 * integrator.cache.f3[i];
                integrator.cache.z3[i] =
                    Radau5::T31 * integrator.cache.f1[i] + integrator.cache.f2[i];
            }
            if integrator.cache.faccon * dyno <= integrator.opts.fnewt {
                return true;
            }
        }
    }
}
