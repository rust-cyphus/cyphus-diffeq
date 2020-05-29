use super::cache::Radau5Cache;
use crate::de_integrator::ODEIntegrator;
use crate::linalg::sol::*;
use ndarray::prelude::*;

impl Radau5Cache {
    /// Compute the error estimate for the Radau5 algorithm.
    pub(crate) fn error_estimate<F, J>(&mut self, integrator: &mut ODEIntegrator<F, J>) -> usize
    where
        F: Fn(ArrayViewMut1<f64>, ArrayView1<f64>, f64),
        J: Fn(ArrayViewMut2<f64>, ArrayView1<f64>, f64),
    {
        let n = self.e1.nrows();
        let nm1 = n - self.m1;
        let hee1 = -(13.0 + 7.0 * 6f64.sqrt()) / (3.0 * integrator.dt);
        let hee2 = (-13.0 + 7.0 * 6f64.sqrt()) / (3.0 * integrator.dt);
        let hee3 = -1.0 / (3.0 * integrator.dt);
        let mut ier = 0;

        match self.prob_type {
            1 => {
                for i in 0..n {
                    self.f2[i] = hee1 * self.z1[i] + hee2 * self.z2[i] + hee3 * self.z3[i];
                    self.cont[i] = self.f2[i] + self.u0[i];
                }
                sol(n, self.e1.view(), self.cont.view_mut(), self.ip1.view());
            }
            2 => {
                // mass = identity, Jacobian a banded matrix
                for i in 0..n {
                    self.f2[i] = hee1 * self.z1[i] + hee2 * self.z2[i] + hee3 * self.z3[i];
                    self.cont[i] = self.f2[i] + self.u0[i];
                }
                solb(
                    n,
                    self.e1.view(),
                    self.mle,
                    self.mue,
                    self.cont.view_mut(),
                    self.ip1.view(),
                );
            }
            3 => {
                // mass is a banded matrix, Jacobian a full matrix
                for i in 0..n {
                    self.f1[i] = hee1 * self.z1[i] + hee2 * self.z2[i] + hee3 * self.z3[i];
                }
                for i in 0..n {
                    let mut sum = 0.0;
                    for j in 0.max(i - self.mlmas)..n.min(i + self.mumas + 1) {
                        sum += self.mass_matrix[[i - j + self.mbdiag - 1, j]] * self.f1[j];
                    }
                    self.f2[i] = sum;
                    self.cont[i] = sum + self.u0[i];
                }
                sol(n, self.e1.view(), self.cont.view_mut(), self.ip1.view());
            }
            4 => {
                // mass is a banded matrix, Jacobian a banded matrix
                for i in 0..n {
                    self.f1[i] = hee1 * self.z1[i] + hee2 * self.z2[i] + hee3 * self.z3[i];
                }
                for i in 0..n {
                    let mut sum = 0.0;
                    for j in 0.max(i - self.mlmas)..n.min(i + self.mumas + 1) {
                        sum = sum + self.mass_matrix[[i - j + self.mbdiag - 1, j]] * self.f1[j];
                    }
                    self.f2[i] = sum;
                    self.cont[i] = sum + self.u0[i];
                }
                solb(
                    n,
                    self.e1.view(),
                    self.mle,
                    self.mue,
                    self.cont.view_mut(),
                    self.ip1.view(),
                );
            }
            5 => {
                // mass is a full matrix, Jacobian a full matrix
                for i in 0..n {
                    self.f1[i] = hee1 * self.z1[i] + hee2 * self.z2[i] + hee3 * self.z3[i];
                }
                for i in 0..n {
                    let mut sum = 0.0;
                    for j in 0..n {
                        sum += self.mass_matrix[[j, i]] * self.f1[j];
                    }
                    self.f2[i] = sum;
                    self.cont[i] = sum + self.u0[i];
                }
                sol(n, self.e1.view(), self.cont.view_mut(), self.ip1.view());
            }
            7 => {
                // mass = identity, Jacobian a full matrix, Hessenberg-option
                for i in 0..n {
                    self.f2[i] = hee1 * self.z1[i] + hee2 * self.z2[i] + hee3 * self.z3[i];
                    self.cont[i] = self.f2[i] + self.u0[i];
                }

                for mm1 in (0..(n - 2)).rev() {
                    let mp = n - mm1 - 2;
                    let ii = self.iphes[mp] as usize;
                    if ii != mp {
                        let zsafe = self.cont[mp];
                        self.cont[mp] = self.cont[ii];
                        self.cont[ii] = zsafe;
                    }
                    for i in mp..n {
                        self.cont[i] -= self.dfdu[[i, mp - 1]] * self.cont[mp];
                    }
                }
                solh(n, self.e1.view(), 1, self.cont.view_mut(), self.ip1.view());

                for mm1 in 0..(n - 2) {
                    let mp = n - mm1 - 2;
                    for i in mp..n {
                        self.cont[i] += self.dfdu[[i, mp - 1]] * self.cont[mp];
                    }
                    let ii = self.iphes[mp] as usize;
                    if ii != mp {
                        let zsafe = self.cont[mp];
                        self.cont[mp] = self.cont[ii];
                        self.cont[ii] = zsafe;
                    }
                }
            }
            11 => {
                // mass = identity, Jacobian a full matrix, second order
                for i in 0..n {
                    self.f2[i] = hee1 * self.z1[i] + hee2 * self.z2[i] + hee3 * self.z3[i];
                    self.cont[i] = self.f2[i] + self.u0[i];
                }
            }
            12 => {
                // mass = identity, Jacobian a banded matrix, second order
                for i in 0..n {
                    self.f2[i] = hee1 * self.z1[i] + hee2 * self.z2[i] + hee3 * self.z3[i];
                    self.cont[i] = self.f2[i] + self.u0[i];
                }
            }
            13 => {
                // mass is a banded matrix, Jacobian a full matrix, second order
                for i in 0..self.m1 {
                    self.f1[i] = hee1 * self.z1[i] + hee2 * self.z2[i] + hee3 * self.z3[i];
                    self.cont[i] = self.f2[i] + self.u0[i];
                }
                for i in self.m1..n {
                    self.f1[i] = hee1 * self.z1[i] + hee2 * self.z2[i] + hee3 * self.z3[i];
                }
                for i in 0..nm1 {
                    let mut sum = 0.0;
                    for j in 0.max(i - self.mlmas)..nm1.min(i + self.mumas + 1) {
                        sum +=
                            self.mass_matrix[[i - j + self.mbdiag - 1, j]] * self.f1[j + self.m1];
                    }
                    self.f2[i + self.m1] = sum;
                    self.cont[i + self.m1] = sum + self.u0[i + self.m1];
                }
            }
            14 => {
                // mass is a banded matrix, Jacobian a banded matrix, second order
                for i in 0..self.m1 {
                    self.f2[i] = hee1 * self.z1[i] + hee2 * self.z2[i] + hee3 * self.z3[i];
                    self.cont[i] = self.f2[i] + self.u0[i];
                }
                for i in self.m1..n {
                    self.f1[i] = hee1 * self.z1[i] + hee2 * self.z2[i] + hee3 * self.z3[i];
                }
                for i in 0..nm1 {
                    let mut sum = 0.0;
                    for j in 0.max(i - self.mlmas)..nm1.min(i + self.mumas + 1) {
                        sum +=
                            self.mass_matrix[[i - j + self.mbdiag - 1, j]] * self.f1[j + self.m1];
                    }
                    self.f2[i + self.m1] = sum;
                    self.cont[i + self.m1] = sum + self.u0[i + self.m1];
                }
            }
            15 => {
                // mass is a banded matrix, Jacobian a full matrix, second order
                for i in 0..self.m1 {
                    self.f2[i] = hee1 * self.z1[i] + hee2 * self.z2[i] + hee3 * self.z3[i];
                    self.cont[i] = self.f2[i] + self.u0[i];
                }
                for i in self.m1..n {
                    self.f1[i] = hee1 * self.z1[i] + hee2 * self.z2[i] + hee3 * self.z3[i];
                }
                for i in 0..nm1 {
                    let mut sum = 0.0;
                    for j in 0..nm1 {
                        sum += self.mass_matrix[[j, i]] * self.f1[j + self.m1];
                    }
                    self.f2[i + self.m1] = sum;
                    self.cont[i + self.m1] = sum + self.u0[i + self.m1];
                }
            }
            _ => {}
        }

        match self.prob_type {
            11 | 13 | 15 => {
                let mm = self.m1 / self.m2;
                for j in 0..self.m2 {
                    let mut sum = 0.0;
                    for k in (0..mm).rev() {
                        sum = (self.cont[j + k * self.m2] + sum) / self.fac1;
                        for i in 0..nm1 {
                            self.cont[i + self.m1] += self.dfdu[[i, j + k * self.m2]] * sum;
                        }
                    }
                }
                sol(
                    nm1,
                    self.e1.view(),
                    self.cont.slice_mut(s![self.m1..]),
                    self.ip1.view(),
                );
                for i in (0..self.m1).rev() {
                    self.cont[i] = (self.cont[i] + self.cont[self.m2 + i]) / self.fac1;
                }
            }
            12 | 14 => {
                let mm = self.m1 / self.m2;
                for j in 0..self.m2 {
                    let mut sum = 0.0;
                    for k in (0..mm).rev() {
                        sum = (self.cont[j + k * self.m2] + sum) / self.fac1;
                        for i in 0.max(j - self.mujac)..nm1.min(j + self.mljac) {
                            self.cont[i + self.m1] +=
                                self.dfdu[[i + self.mujac - j, j + k * self.m2]] * sum;
                        }
                    }
                }
                solb(
                    nm1,
                    self.e1.view(),
                    self.mle,
                    self.mue,
                    self.cont.slice_mut(s![self.m1..]),
                    self.ip1.view(),
                );
                for i in (0..self.m1).rev() {
                    self.cont[i] = (self.cont[i] + self.cont[self.m2 + i]) / self.fac1;
                }
            }
            _ => {}
        }

        let mut err = 0.0;
        for i in 0..n {
            err += (self.cont[i] / self.scal[i]).powi(2);
        }
        err = (err / n as f64).sqrt().max(1.0e-10);

        if err < 1.0 {
            return ier;
        }

        if self.first || self.reject {
            for i in 0..n {
                self.cont[i] = integrator.u[i] + self.cont[i];
            }
            (integrator.dudt)(self.f1.view_mut(), self.cont.view(), integrator.t);
            integrator.stats.function_evals += 1;
            for i in 0..n {
                self.cont[i] = self.f1[i] + self.f2[i];
            }

            match self.prob_type {
                1 | 3 | 5 => {
                    sol(n, self.e1.view(), self.cont.view_mut(), self.ip1.view());
                }
                2 | 4 => {
                    solb(
                        n,
                        self.e1.view(),
                        self.mle,
                        self.mue,
                        self.cont.view_mut(),
                        self.ip1.view(),
                    );
                }
                7 => {
                    // Hessenberg matrix option
                    // mass = identity, Jacobian a full matrix, Hessenberg-option
                    for mm1 in (0..(n - 2)).rev() {
                        let mp = n - mm1 - 2;
                        let ii = self.iphes[mp] as usize;
                        if ii != mp {
                            let zsafe = self.cont[mp];
                            self.cont[mp] = self.cont[ii];
                            self.cont[ii] = zsafe;
                        }
                        for i in mp..n {
                            self.cont[i] -= self.dfdu[[i, mp - 1]] * self.cont[mp];
                        }
                    }
                    solh(n, self.e1.view(), 1, self.cont.view_mut(), self.ip1.view());
                    for mm1 in 0..(n - 2) {
                        let mp = n - mm1 - 2;
                        for i in mp..n {
                            self.cont[i] += self.dfdu[[i, mp - 1]] * self.cont[mp];
                        }
                        let ii = self.iphes[mp] as usize;
                        if ii != mp {
                            let zsafe = self.cont[mp];
                            self.cont[mp] = self.cont[ii];
                            self.cont[ii] = zsafe;
                        }
                    }
                }
                11 | 13 | 15 => {
                    // Full matrix option, second order
                    for j in 0..self.m2 {
                        let mut sum = 0.0;
                        let mm = self.m1 / self.m2;
                        for k in (0..mm).rev() {
                            sum = (self.cont[j + k * self.m2] + sum) / self.fac1;
                            for i in 0..nm1 {
                                self.cont[i + self.m1] += self.dfdu[[i, j + k * self.m2]] * sum;
                            }
                        }
                    }

                    sol(
                        nm1,
                        self.e1.view(),
                        self.cont.slice_mut(s![self.m1..]),
                        self.ip1.view(),
                    );
                    for i in (0..self.m1).rev() {
                        self.cont[i] = (self.cont[i] + self.cont[self.m2 + i]) / self.fac1;
                    }
                }
                12 | 14 => {
                    // Banded matrix option, second order
                    for j in 0..self.m2 {
                        let mut sum = 0.0;
                        let mm = self.m1 / self.m2;
                        for k in (0..mm).rev() {
                            sum = (self.cont[j + k * self.m2] + sum) / self.fac1;
                            for i in 0.max(j - self.mujac)..nm1.min(j + self.mljac) {
                                self.cont[i + self.m1] +=
                                    self.dfdu[[i + self.mujac - j, j + k * self.m2]] * sum;
                            }
                        }
                    }
                    solb(
                        nm1,
                        self.e1.view(),
                        self.mle,
                        self.mue,
                        self.cont.slice_mut(s![self.m1..]),
                        self.ip1.view(),
                    );
                    for i in (0..self.m1).rev() {
                        self.cont[i] = (self.cont[i] + self.cont[self.m2 + i]) / self.fac1;
                    }
                }
                _ => {}
            }

            err = 0.0;
            for i in 0..n {
                err += (self.cont[i] / self.scal[i]).powi(2);
            }
            err = (err / n as f64).sqrt().max(1.0e-10);
        }

        return ier;
    }
}
