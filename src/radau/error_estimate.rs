use super::alg::Radau5;
use super::cache::Radau5Cache;
use crate::linalg::sol::*;
use crate::ode_integrator::OdeIntegrator;
use crate::ode_prob::OdeProblem;
use ndarray::prelude::*;

impl Radau5 {
    /// Compute the error estimate for the Radau5 algorithm.
    pub(crate) fn error_estimate<F, J>(
        &self,
        integrator: &mut OdeIntegrator,
        prob: &OdeProblem<F, J>,
        cache: &mut Radau5Cache,
    ) -> usize
    where
        F: Fn(ArrayViewMut1<f64>, ArrayView1<f64>, f64),
        J: Fn(ArrayViewMut2<f64>, ArrayView1<f64>, f64),
    {
        let n = cache.e1.nrows();
        let nm1 = n - cache.m1;
        let hee1 = -(13.0 + 7.0 * 6f64.sqrt()) / (3.0 * integrator.dt);
        let hee2 = (-13.0 + 7.0 * 6f64.sqrt()) / (3.0 * integrator.dt);
        let hee3 = -1.0 / (3.0 * integrator.dt);
        let mut ier = 0;

        match cache.prob_type {
            1 => {
                for i in 0..n {
                    cache.f2[i] = hee1 * cache.z1[i] + hee2 * cache.z2[i] + hee3 * cache.z3[i];
                    cache.cont[i] = cache.f2[i] + cache.u0[i];
                }
                sol(n, cache.e1.view(), cache.cont.view_mut(), cache.ip1.view());
            }
            2 => {
                // mass = identity, Jacobian a banded matrix
                for i in 0..n {
                    cache.f2[i] = hee1 * cache.z1[i] + hee2 * cache.z2[i] + hee3 * cache.z3[i];
                    cache.cont[i] = cache.f2[i] + cache.u0[i];
                }
                solb(
                    n,
                    cache.e1.view(),
                    cache.mle,
                    cache.mue,
                    cache.cont.view_mut(),
                    cache.ip1.view(),
                );
            }
            3 => {
                // mass is a banded matrix, Jacobian a full matrix
                for i in 0..n {
                    cache.f1[i] = hee1 * cache.z1[i] + hee2 * cache.z2[i] + hee3 * cache.z3[i];
                }
                for i in 0..n {
                    let mut sum = 0.0;
                    for j in 0.max(i - cache.mlmas)..n.min(i + cache.mumas + 1) {
                        sum += cache.mass_matrix[[i - j + cache.mbdiag - 1, j]] * cache.f1[j];
                    }
                    cache.f2[i] = sum;
                    cache.cont[i] = sum + cache.u0[i];
                }
                sol(n, cache.e1.view(), cache.cont.view_mut(), cache.ip1.view());
            }
            4 => {
                // mass is a banded matrix, Jacobian a banded matrix
                for i in 0..n {
                    cache.f1[i] = hee1 * cache.z1[i] + hee2 * cache.z2[i] + hee3 * cache.z3[i];
                }
                for i in 0..n {
                    let mut sum = 0.0;
                    for j in 0.max(i - cache.mlmas)..n.min(i + cache.mumas + 1) {
                        sum = sum + cache.mass_matrix[[i - j + cache.mbdiag - 1, j]] * cache.f1[j];
                    }
                    cache.f2[i] = sum;
                    cache.cont[i] = sum + cache.u0[i];
                }
                solb(
                    n,
                    cache.e1.view(),
                    cache.mle,
                    cache.mue,
                    cache.cont.view_mut(),
                    cache.ip1.view(),
                );
            }
            5 => {
                // mass is a full matrix, Jacobian a full matrix
                for i in 0..n {
                    cache.f1[i] = hee1 * cache.z1[i] + hee2 * cache.z2[i] + hee3 * cache.z3[i];
                }
                for i in 0..n {
                    let mut sum = 0.0;
                    for j in 0..n {
                        sum += cache.mass_matrix[[j, i]] * cache.f1[j];
                    }
                    cache.f2[i] = sum;
                    cache.cont[i] = sum + cache.u0[i];
                }
                sol(n, cache.e1.view(), cache.cont.view_mut(), cache.ip1.view());
            }
            7 => {
                // mass = identity, Jacobian a full matrix, Hessenberg-option
                for i in 0..n {
                    cache.f2[i] = hee1 * cache.z1[i] + hee2 * cache.z2[i] + hee3 * cache.z3[i];
                    cache.cont[i] = cache.f2[i] + cache.u0[i];
                }

                for mm1 in (0..(n - 2)).rev() {
                    let mp = n - mm1 - 2;
                    let ii = cache.iphes[mp] as usize;
                    if ii != mp {
                        let zsafe = cache.cont[mp];
                        cache.cont[mp] = cache.cont[ii];
                        cache.cont[ii] = zsafe;
                    }
                    for i in mp..n {
                        cache.cont[i] -= cache.dfdu[[i, mp - 1]] * cache.cont[mp];
                    }
                }
                solh(
                    n,
                    cache.e1.view(),
                    1,
                    cache.cont.view_mut(),
                    cache.ip1.view(),
                );

                for mm1 in 0..(n - 2) {
                    let mp = n - mm1 - 2;
                    for i in mp..n {
                        cache.cont[i] += cache.dfdu[[i, mp - 1]] * cache.cont[mp];
                    }
                    let ii = cache.iphes[mp] as usize;
                    if ii != mp {
                        let zsafe = cache.cont[mp];
                        cache.cont[mp] = cache.cont[ii];
                        cache.cont[ii] = zsafe;
                    }
                }
            }
            11 => {
                // mass = identity, Jacobian a full matrix, second order
                for i in 0..n {
                    cache.f2[i] = hee1 * cache.z1[i] + hee2 * cache.z2[i] + hee3 * cache.z3[i];
                    cache.cont[i] = cache.f2[i] + cache.u0[i];
                }
            }
            12 => {
                // mass = identity, Jacobian a banded matrix, second order
                for i in 0..n {
                    cache.f2[i] = hee1 * cache.z1[i] + hee2 * cache.z2[i] + hee3 * cache.z3[i];
                    cache.cont[i] = cache.f2[i] + cache.u0[i];
                }
            }
            13 => {
                // mass is a banded matrix, Jacobian a full matrix, second order
                for i in 0..cache.m1 {
                    cache.f1[i] = hee1 * cache.z1[i] + hee2 * cache.z2[i] + hee3 * cache.z3[i];
                    cache.cont[i] = cache.f2[i] + cache.u0[i];
                }
                for i in cache.m1..n {
                    cache.f1[i] = hee1 * cache.z1[i] + hee2 * cache.z2[i] + hee3 * cache.z3[i];
                }
                for i in 0..nm1 {
                    let mut sum = 0.0;
                    for j in 0.max(i - cache.mlmas)..nm1.min(i + cache.mumas + 1) {
                        sum += cache.mass_matrix[[i - j + cache.mbdiag - 1, j]]
                            * cache.f1[j + cache.m1];
                    }
                    cache.f2[i + cache.m1] = sum;
                    cache.cont[i + cache.m1] = sum + cache.u0[i + cache.m1];
                }
            }
            14 => {
                // mass is a banded matrix, Jacobian a banded matrix, second order
                for i in 0..cache.m1 {
                    cache.f2[i] = hee1 * cache.z1[i] + hee2 * cache.z2[i] + hee3 * cache.z3[i];
                    cache.cont[i] = cache.f2[i] + cache.u0[i];
                }
                for i in cache.m1..n {
                    cache.f1[i] = hee1 * cache.z1[i] + hee2 * cache.z2[i] + hee3 * cache.z3[i];
                }
                for i in 0..nm1 {
                    let mut sum = 0.0;
                    for j in 0.max(i - cache.mlmas)..nm1.min(i + cache.mumas + 1) {
                        sum += cache.mass_matrix[[i - j + cache.mbdiag - 1, j]]
                            * cache.f1[j + cache.m1];
                    }
                    cache.f2[i + cache.m1] = sum;
                    cache.cont[i + cache.m1] = sum + cache.u0[i + cache.m1];
                }
            }
            15 => {
                // mass is a banded matrix, Jacobian a full matrix, second order
                for i in 0..cache.m1 {
                    cache.f2[i] = hee1 * cache.z1[i] + hee2 * cache.z2[i] + hee3 * cache.z3[i];
                    cache.cont[i] = cache.f2[i] + cache.u0[i];
                }
                for i in cache.m1..n {
                    cache.f1[i] = hee1 * cache.z1[i] + hee2 * cache.z2[i] + hee3 * cache.z3[i];
                }
                for i in 0..nm1 {
                    let mut sum = 0.0;
                    for j in 0..nm1 {
                        sum += cache.mass_matrix[[j, i]] * cache.f1[j + cache.m1];
                    }
                    cache.f2[i + cache.m1] = sum;
                    cache.cont[i + cache.m1] = sum + cache.u0[i + cache.m1];
                }
            }
            _ => {}
        }

        match cache.prob_type {
            11 | 13 | 15 => {
                let mm = cache.m1 / cache.m2;
                for j in 0..cache.m2 {
                    let mut sum = 0.0;
                    for k in (0..mm).rev() {
                        sum = (cache.cont[j + k * cache.m2] + sum) / cache.fac1;
                        for i in 0..nm1 {
                            cache.cont[i + cache.m1] += cache.dfdu[[i, j + k * cache.m2]] * sum;
                        }
                    }
                }
                sol(
                    nm1,
                    cache.e1.view(),
                    cache.cont.slice_mut(s![cache.m1..]),
                    cache.ip1.view(),
                );
                for i in (0..cache.m1).rev() {
                    cache.cont[i] = (cache.cont[i] + cache.cont[cache.m2 + i]) / cache.fac1;
                }
            }
            12 | 14 => {
                let mm = cache.m1 / cache.m2;
                for j in 0..cache.m2 {
                    let mut sum = 0.0;
                    for k in (0..mm).rev() {
                        sum = (cache.cont[j + k * cache.m2] + sum) / cache.fac1;
                        for i in 0.max(j - cache.mujac)..nm1.min(j + cache.mljac) {
                            cache.cont[i + cache.m1] +=
                                cache.dfdu[[i + cache.mujac - j, j + k * cache.m2]] * sum;
                        }
                    }
                }
                solb(
                    nm1,
                    cache.e1.view(),
                    cache.mle,
                    cache.mue,
                    cache.cont.slice_mut(s![cache.m1..]),
                    cache.ip1.view(),
                );
                for i in (0..cache.m1).rev() {
                    cache.cont[i] = (cache.cont[i] + cache.cont[cache.m2 + i]) / cache.fac1;
                }
            }
            _ => {}
        }

        cache.err = 0.0;
        for i in 0..n {
            cache.err += (cache.cont[i] / cache.scal[i]).powi(2);
        }
        cache.err = (cache.err / n as f64).sqrt().max(1.0e-10);

        if cache.err < 1.0 {
            return ier;
        }

        if cache.first || cache.reject {
            for i in 0..n {
                cache.cont[i] = integrator.u[i] + cache.cont[i];
            }
            (prob.dudt)(cache.f1.view_mut(), cache.cont.view(), integrator.t);
            integrator.stats.function_evals += 1;
            for i in 0..n {
                cache.cont[i] = cache.f1[i] + cache.f2[i];
            }

            match cache.prob_type {
                1 | 3 | 5 => {
                    sol(n, cache.e1.view(), cache.cont.view_mut(), cache.ip1.view());
                }
                2 | 4 => {
                    solb(
                        n,
                        cache.e1.view(),
                        cache.mle,
                        cache.mue,
                        cache.cont.view_mut(),
                        cache.ip1.view(),
                    );
                }
                7 => {
                    // Hessenberg matrix option
                    // mass = identity, Jacobian a full matrix, Hessenberg-option
                    for mm1 in (0..(n - 2)).rev() {
                        let mp = n - mm1 - 2;
                        let ii = cache.iphes[mp] as usize;
                        if ii != mp {
                            let zsafe = cache.cont[mp];
                            cache.cont[mp] = cache.cont[ii];
                            cache.cont[ii] = zsafe;
                        }
                        for i in mp..n {
                            cache.cont[i] -= cache.dfdu[[i, mp - 1]] * cache.cont[mp];
                        }
                    }
                    solh(
                        n,
                        cache.e1.view(),
                        1,
                        cache.cont.view_mut(),
                        cache.ip1.view(),
                    );
                    for mm1 in 0..(n - 2) {
                        let mp = n - mm1 - 2;
                        for i in mp..n {
                            cache.cont[i] += cache.dfdu[[i, mp - 1]] * cache.cont[mp];
                        }
                        let ii = cache.iphes[mp] as usize;
                        if ii != mp {
                            let zsafe = cache.cont[mp];
                            cache.cont[mp] = cache.cont[ii];
                            cache.cont[ii] = zsafe;
                        }
                    }
                }
                11 | 13 | 15 => {
                    // Full matrix option, second order
                    for j in 0..cache.m2 {
                        let mut sum = 0.0;
                        let mm = cache.m1 / cache.m2;
                        for k in (0..mm).rev() {
                            sum = (cache.cont[j + k * cache.m2] + sum) / cache.fac1;
                            for i in 0..nm1 {
                                cache.cont[i + cache.m1] += cache.dfdu[[i, j + k * cache.m2]] * sum;
                            }
                        }
                    }

                    sol(
                        nm1,
                        cache.e1.view(),
                        cache.cont.slice_mut(s![cache.m1..]),
                        cache.ip1.view(),
                    );
                    for i in (0..cache.m1).rev() {
                        cache.cont[i] = (cache.cont[i] + cache.cont[cache.m2 + i]) / cache.fac1;
                    }
                }
                12 | 14 => {
                    // Banded matrix option, second order
                    for j in 0..cache.m2 {
                        let mut sum = 0.0;
                        let mm = cache.m1 / cache.m2;
                        for k in (0..mm).rev() {
                            sum = (cache.cont[j + k * cache.m2] + sum) / cache.fac1;
                            for i in 0.max(j - cache.mujac)..nm1.min(j + cache.mljac) {
                                cache.cont[i + cache.m1] +=
                                    cache.dfdu[[i + cache.mujac - j, j + k * cache.m2]] * sum;
                            }
                        }
                    }
                    solb(
                        nm1,
                        cache.e1.view(),
                        cache.mle,
                        cache.mue,
                        cache.cont.slice_mut(s![cache.m1..]),
                        cache.ip1.view(),
                    );
                    for i in (0..cache.m1).rev() {
                        cache.cont[i] = (cache.cont[i] + cache.cont[cache.m2 + i]) / cache.fac1;
                    }
                }
                _ => {}
            }

            cache.err = 0.0;
            for i in 0..n {
                cache.err += (cache.cont[i] / cache.scal[i]).powi(2);
            }
            cache.err = (cache.err / n as f64).sqrt().max(1.0e-10);
        }

        return ier;
    }
}
