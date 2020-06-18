use super::DormandPrince5;
use crate::ode::OdeIntegrator;

impl DormandPrince5 {
    /// Compute the 6 stages of the Dormand-Prince DormandPrince8::rithm.
    #[allow(dead_code)]
    pub(super) fn compute_stages<Params>(integrator: &mut OdeIntegrator<Params, DormandPrince5>) {
        integrator.cache.unew.assign(
            &(&integrator.u + &(integrator.dt * DormandPrince5::A21 * &integrator.cache.du)),
        );
        (integrator.dudt)(
            integrator.cache.k2.view_mut(),
            integrator.cache.unew.view(),
            integrator.t + DormandPrince5::C2 * integrator.dt,
            &mut integrator.params,
        );

        integrator.cache.unew.assign(
            &(&integrator.u
                + &(integrator.dt
                    * (DormandPrince5::A31 * &integrator.cache.du
                        + DormandPrince5::A32 * &integrator.cache.k2))),
        );
        (integrator.dudt)(
            integrator.cache.k3.view_mut(),
            integrator.cache.unew.view(),
            integrator.t + DormandPrince5::C3 * integrator.dt,
            &mut integrator.params,
        );

        integrator.cache.unew.assign(
            &(&integrator.u
                + &(integrator.dt
                    * (DormandPrince5::A41 * &integrator.cache.du
                        + DormandPrince5::A42 * &integrator.cache.k2
                        + DormandPrince5::A43 * &integrator.cache.k3))),
        );
        (integrator.dudt)(
            integrator.cache.k4.view_mut(),
            integrator.cache.unew.view(),
            integrator.t + DormandPrince5::C4 * integrator.dt,
            &mut integrator.params,
        );

        integrator.cache.unew.assign(
            &(&integrator.u
                + &(integrator.dt
                    * (DormandPrince5::A51 * &integrator.cache.du
                        + DormandPrince5::A52 * &integrator.cache.k2
                        + DormandPrince5::A53 * &integrator.cache.k3
                        + DormandPrince5::A54 * &integrator.cache.k4))),
        );
        (integrator.dudt)(
            integrator.cache.k5.view_mut(),
            integrator.cache.unew.view(),
            integrator.t + DormandPrince5::C5 * integrator.dt,
            &mut integrator.params,
        );

        integrator.cache.ustiff.assign(
            &(&integrator.u
                + &(integrator.dt
                    * (DormandPrince5::A61 * &integrator.cache.du
                        + DormandPrince5::A62 * &integrator.cache.k2
                        + DormandPrince5::A63 * &integrator.cache.k3
                        + DormandPrince5::A64 * &integrator.cache.k4
                        + DormandPrince5::A65 * &integrator.cache.k5))),
        );
        // The RK step
        let tph = integrator.t + integrator.dt;
        (integrator.dudt)(
            integrator.cache.k6.view_mut(),
            integrator.cache.ustiff.view(),
            tph,
            &mut integrator.params,
        );

        integrator.cache.unew.assign(
            &(&integrator.u
                + &(integrator.dt
                    * (DormandPrince5::A71 * &integrator.cache.du
                        + DormandPrince5::A73 * &integrator.cache.k3
                        + DormandPrince5::A74 * &integrator.cache.k4
                        + DormandPrince5::A75 * &integrator.cache.k5
                        + DormandPrince5::A76 * &integrator.cache.k6))),
        );
        // Error estimate using embedded solution
        (integrator.dudt)(
            integrator.cache.dunew.view_mut(),
            integrator.cache.unew.view(),
            tph,
            &mut integrator.params,
        );

        integrator.cache.uerr.assign(
            &(integrator.dt
                * (DormandPrince5::E1 * &integrator.cache.du
                    + DormandPrince5::E3 * &integrator.cache.k3
                    + DormandPrince5::E4 * &integrator.cache.k4
                    + DormandPrince5::E5 * &integrator.cache.k5
                    + DormandPrince5::E6 * &integrator.cache.k6
                    + DormandPrince5::E7 * &integrator.cache.dunew)),
        );
        // Update counting variables
        integrator.stats.function_evals += 6;
    }
}
