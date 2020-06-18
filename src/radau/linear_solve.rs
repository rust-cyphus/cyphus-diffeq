use super::Radau5;
use crate::linalg::sol::*;
use crate::ode::OdeIntegrator;

impl Radau5 {
    /// Solve the linear systems for the Radau5 algorithm
    pub(crate) fn linear_solve<Params>(integrator: &mut OdeIntegrator<Params, Self>) {
        let n = integrator.cache.e1.nrows();
        for i in 0..n {
            let s2 = -integrator.cache.f2[i];
            let s3 = -integrator.cache.f3[i];
            integrator.cache.z1[i] -= integrator.cache.f1[i] * integrator.cache.fac1;
            integrator.cache.z2[i] =
                integrator.cache.z2[i] + s2 * integrator.cache.alphn - s3 * integrator.cache.betan;
            integrator.cache.z3[i] =
                integrator.cache.z3[i] + s3 * integrator.cache.alphn + s2 * integrator.cache.betan;
        }
        sol(
            n,
            integrator.cache.e1.view(),
            integrator.cache.z1.view_mut(),
            integrator.cache.ip1.view(),
        );
        solc(
            n,
            integrator.cache.e2r.view(),
            integrator.cache.e2i.view(),
            integrator.cache.z2.view_mut(),
            integrator.cache.z3.view_mut(),
            integrator.cache.ip2.view(),
        );
    }
}
