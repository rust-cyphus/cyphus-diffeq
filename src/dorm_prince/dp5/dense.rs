use super::DormandPrince5;
use crate::ode::OdeIntegrator;
use ndarray::prelude::*;

impl DormandPrince5 {
    /// Prepare the continuous output vector for dense output.
    #[allow(dead_code)]
    pub(super) fn prepare_dense<Params>(integrator: &mut OdeIntegrator<Params, DormandPrince5>) {
        integrator.cache.rcont1.assign(&integrator.u);
        integrator
            .cache
            .rcont2
            .assign(&(&integrator.cache.unew - &integrator.u));
        integrator
            .cache
            .rcont3
            .assign(&(integrator.dt * &integrator.cache.du - &integrator.cache.rcont2));
        integrator.cache.rcont4.assign(
            &(&integrator.cache.rcont2
                - &(integrator.dt * &integrator.cache.dunew)
                - &integrator.cache.rcont3),
        );
        integrator.cache.rcont5.assign(
            &(integrator.dt
                * &(DormandPrince5::D1 * &integrator.cache.du
                    + DormandPrince5::D3 * &integrator.cache.k3
                    + DormandPrince5::D4 * &integrator.cache.k4
                    + DormandPrince5::D5 * &integrator.cache.k5
                    + DormandPrince5::D6 * &integrator.cache.k6
                    + DormandPrince5::D7 * &integrator.cache.dunew)),
        );
    }
    /// Compute the dense output for the solution vector.
    #[allow(dead_code)]
    pub(super) fn dense_output<Params>(
        integrator: &mut OdeIntegrator<Params, DormandPrince5>,
        t: f64,
        dt: f64,
    ) -> Array1<f64> {
        let s = (t - integrator.tprev) / dt;
        let s1 = 1.0 - s;

        &integrator.cache.rcont1
            + &(s
                * (&integrator.cache.rcont2
                    + &(s1
                        * (&integrator.cache.rcont3
                            + &(s
                                * (&integrator.cache.rcont4 + &(s1 * &integrator.cache.rcont5)))))))
    }
}
