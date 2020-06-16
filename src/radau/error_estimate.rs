use super::Radau5;
use crate::linalg::sol::*;
use crate::ode::{OdeFunction, OdeIntegrator};

impl Radau5 {
    /// Compute the error estimate for the Radau5 algorithm.
    pub(crate) fn error_estimate<T: OdeFunction>(integrator: &mut OdeIntegrator<T, Radau5>) {
        integrator.cache.err = 0.0;

        let n = integrator.cache.e1.nrows();
        let hee1 = -(13.0 + 7.0 * 6f64.sqrt()) / (3.0 * integrator.dt);
        let hee2 = (-13.0 + 7.0 * 6f64.sqrt()) / (3.0 * integrator.dt);
        let hee3 = -1.0 / (3.0 * integrator.dt);

        for i in 0..n {
            integrator.cache.f2[i] = hee1 * integrator.cache.z1[i]
                + hee2 * integrator.cache.z2[i]
                + hee3 * integrator.cache.z3[i];
            integrator.cache.cont[i] = integrator.cache.f2[i] + integrator.cache.u0[i];
        }
        sol(
            n,
            integrator.cache.e1.view(),
            integrator.cache.cont.view_mut(),
            integrator.cache.ip1.view(),
        );

        integrator.cache.err = 0.0;
        for i in 0..n {
            integrator.cache.err += (integrator.cache.cont[i] / integrator.cache.scal[i]).powi(2);
        }
        integrator.cache.err = (integrator.cache.err / n as f64).sqrt().max(1.0e-10);

        if integrator.cache.err < 1.0 {
            return;
        }

        if integrator.cache.first || integrator.cache.reject {
            for i in 0..n {
                integrator.cache.cont[i] = integrator.u[i] + integrator.cache.cont[i];
            }
            integrator.func.dudt(
                integrator.cache.f1.view_mut(),
                integrator.cache.cont.view(),
                integrator.t,
            );
            integrator.stats.function_evals += 1;
            for i in 0..n {
                integrator.cache.cont[i] = integrator.cache.f1[i] + integrator.cache.f2[i];
            }
            sol(
                n,
                integrator.cache.e1.view(),
                integrator.cache.cont.view_mut(),
                integrator.cache.ip1.view(),
            );

            integrator.cache.err = 0.0;
            for i in 0..n {
                integrator.cache.err +=
                    (integrator.cache.cont[i] / integrator.cache.scal[i]).powi(2);
            }
            integrator.cache.err = (integrator.cache.err / n as f64).sqrt().max(1.0e-10);
        }
    }
}
