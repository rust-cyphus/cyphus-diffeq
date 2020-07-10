use super::DormandPrince8;
use super::DormandPrince8Cache;
use crate::ode_integrator::OdeIntegrator;
use crate::ode_prob::OdeProblem;

use ndarray::prelude::*;

impl DormandPrince8 {
    /// Compute the 6 stages of the Dormand-Prince DormandPrince8::rithm.
    #[allow(dead_code)]
    fn compute_stages<F, J>(
        &self,
        integrator: &mut OdeIntegrator,
        prob: &mut OdeProblem<F, J>,
        cache: &mut DormandPrince8Cache,
    ) where
        F: Fn(ArrayViewMut1<f64>, ArrayView1<f64>, f64),
        J: Fn(ArrayViewMut2<f64>, ArrayView1<f64>, f64),
    {
        cache
            .unew
            .assign(&(&integrator.u + &(integrator.dt * DormandPrince8::a21 * &cache.du)));
        (prob.dudt)(
            cache.k2.view_mut(),
            cache.unew.view(),
            integrator.t + DormandPrince8::c2 * integrator.dt,
        );

        cache.unew.assign(
            &(&integrator.u
                + &(integrator.dt
                    * (DormandPrince8::a31 * &cache.du + DormandPrince8::a32 * &cache.k2))),
        );
        (prob.dudt)(
            cache.k3.view_mut(),
            cache.unew.view(),
            integrator.t + DormandPrince8::c3 * integrator.dt,
        );

        cache.unew.assign(
            &(&integrator.u
                + &(integrator.dt
                    * (DormandPrince8::a41 * &cache.du + DormandPrince8::a43 * &cache.k3))),
        );
        (prob.dudt)(
            cache.k4.view_mut(),
            cache.unew.view(),
            integrator.t + DormandPrince8::c4 * integrator.dt,
        );

        cache.unew.assign(
            &(&integrator.u
                + &(integrator.dt
                    * (DormandPrince8::a51 * &cache.du
                        + DormandPrince8::a53 * &cache.k3
                        + DormandPrince8::a54 * &cache.k4))),
        );
        (prob.dudt)(
            cache.k5.view_mut(),
            cache.unew.view(),
            integrator.t + DormandPrince8::c5 * integrator.dt,
        );

        cache.unew.assign(
            &(&integrator.u
                + &(integrator.dt
                    * (DormandPrince8::a61 * &cache.du
                        + DormandPrince8::a64 * &cache.k4
                        + DormandPrince8::a65 * &cache.k5))),
        );
        (prob.dudt)(
            cache.k6.view_mut(),
            cache.unew.view(),
            integrator.t + DormandPrince8::c6 * integrator.dt,
        );

        cache.unew.assign(
            &(&integrator.u
                + &(integrator.dt
                    * (DormandPrince8::a71 * &cache.du
                        + DormandPrince8::a74 * &cache.k4
                        + DormandPrince8::a75 * &cache.k5
                        + DormandPrince8::a76 * &cache.k6))),
        );
        (prob.dudt)(
            cache.k7.view_mut(),
            cache.unew.view(),
            integrator.t + DormandPrince8::c7 * integrator.dt,
        );

        cache.unew.assign(
            &(&integrator.u
                + &(integrator.dt
                    * (DormandPrince8::a81 * &cache.du
                        + DormandPrince8::a84 * &cache.k4
                        + DormandPrince8::a85 * &cache.k5
                        + DormandPrince8::a86 * &cache.k6
                        + DormandPrince8::a87 * &cache.k7))),
        );
        (prob.dudt)(
            cache.k8.view_mut(),
            cache.unew.view(),
            integrator.t + DormandPrince8::c8 * integrator.dt,
        );

        cache.unew.assign(
            &(&integrator.u
                + &(integrator.dt
                    * (DormandPrince8::a91 * &cache.du
                        + DormandPrince8::a94 * &cache.k4
                        + DormandPrince8::a95 * &cache.k5
                        + DormandPrince8::a96 * &cache.k6
                        + DormandPrince8::a97 * &cache.k7
                        + DormandPrince8::a98 * &cache.k8))),
        );
        (prob.dudt)(
            cache.k9.view_mut(),
            cache.unew.view(),
            integrator.t + DormandPrince8::c9 * integrator.dt,
        );

        cache.unew.assign(
            &(&integrator.u
                + &(integrator.dt
                    * (DormandPrince8::a101 * &cache.du
                        + DormandPrince8::a104 * &cache.k4
                        + DormandPrince8::a105 * &cache.k5
                        + DormandPrince8::a106 * &cache.k6
                        + DormandPrince8::a107 * &cache.k7
                        + DormandPrince8::a108 * &cache.k8
                        + DormandPrince8::a109 * &cache.k9))),
        );
        (prob.dudt)(
            cache.k10.view_mut(),
            cache.unew.view(),
            integrator.t + DormandPrince8::c10 * integrator.dt,
        );

        cache.unew.assign(
            &(&integrator.u
                + &(integrator.dt
                    * (DormandPrince8::a111 * &cache.du
                        + DormandPrince8::a114 * &cache.k4
                        + DormandPrince8::a115 * &cache.k5
                        + DormandPrince8::a116 * &cache.k6
                        + DormandPrince8::a117 * &cache.k7
                        + DormandPrince8::a118 * &cache.k8
                        + DormandPrince8::a119 * &cache.k9
                        + DormandPrince8::a1110 * &cache.k10))),
        );
        (prob.dudt)(
            cache.k2.view_mut(),
            cache.unew.view(),
            integrator.t + DormandPrince8::c11 * integrator.dt,
        );

        cache.unew.assign(
            &(&integrator.u
                + &(integrator.dt
                    * (DormandPrince8::a121 * &cache.du
                        + DormandPrince8::a124 * &cache.k4
                        + DormandPrince8::a125 * &cache.k5
                        + DormandPrince8::a126 * &cache.k6
                        + DormandPrince8::a127 * &cache.k7
                        + DormandPrince8::a128 * &cache.k8
                        + DormandPrince8::a129 * &cache.k9
                        + DormandPrince8::a1210 * &cache.k10
                        + DormandPrince8::a1211 * &cache.k2))),
        );
        (prob.dudt)(
            cache.k3.view_mut(),
            cache.unew.view(),
            integrator.t + integrator.dt,
        );
        integrator.stats.function_evals += 11;

        cache.k4.assign(
            &(DormandPrince8::b1 * &cache.du
                + DormandPrince8::b6 * &cache.k6
                + DormandPrince8::b7 * &cache.k7
                + DormandPrince8::b8 * &cache.k8
                + DormandPrince8::b9 * &cache.k9
                + DormandPrince8::b10 * &cache.k10
                + DormandPrince8::b11 * &cache.k2
                + DormandPrince8::b12 * &cache.k3),
        );
        cache
            .k5
            .assign(&(&integrator.u + &(integrator.dt * &cache.k4)));
    }
}
