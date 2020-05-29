use super::cache::Radau5Cache;
use crate::linalg::sol::*;
use crate::radau::alg::Radau5;
use ndarray::prelude::*;

impl Radau5Cache {
    /// Solve the linear systems for the Radau5 algorithm
    pub(crate) fn linear_solve(&mut self) {
        let n = self.e1.nrows();
        let nm1 = n - self.m1;

        match self.prob_type {
            1 => {
                // mass = identity, Jacobian a full matrix
                for i in 0..n {
                    let s2 = -self.f2[i];
                    let s3 = -self.f3[i];
                    self.z1[i] -= self.f1[i] * self.fac1;
                    self.z2[i] = self.z2[i] + s2 * self.alphn - s3 * self.betan;
                    self.z3[i] = self.z3[i] + s3 * self.alphn + s2 * self.betan;
                }
                sol(n, self.e1.view(), self.z1.view_mut(), self.ip1.view());
                solc(
                    n,
                    self.e2r.view(),
                    self.e2i.view(),
                    self.z2.view_mut(),
                    self.z3.view_mut(),
                    self.ip2.view(),
                );
            }
            2 => {
                // mass = identity, Jacobian a banded matrix
                for i in 0..n {
                    let s2 = -self.f2[i];
                    let s3 = -self.f3[i];
                    self.z1[i] -= self.f1[i] * self.fac1;
                    self.z2[i] = self.z2[i] + s2 * self.alphn - s3 * self.betan;
                    self.z3[i] = self.z3[i] + s3 * self.alphn + s2 * self.betan;
                }
                solb(
                    n,
                    self.e1.view(),
                    self.mle,
                    self.mue,
                    self.z1.view_mut(),
                    self.ip1.view(),
                );
                solbc(
                    n,
                    self.e2r.view(),
                    self.e2i.view(),
                    self.mle,
                    self.mue,
                    self.z2.view_mut(),
                    self.z3.view_mut(),
                    self.ip2.view(),
                );
            }
            3 => {
                // mass is a banded matrix, Jacobian a full matrix
                for i in 0..n {
                    let mut s1 = 0.0;
                    let mut s2 = 0.0;
                    let mut s3 = 0.0;
                    for j in 0.max(i - self.mlmas)..n.min(i + self.mumas + 1) {
                        let bb = self.mass_matrix[[i - j + self.mbdiag - 1, j]];
                        s1 -= bb * self.f1[j];
                        s2 -= bb * self.f2[j];
                        s3 -= bb * self.f3[j];
                    }
                    self.z1[i] += s1 * self.fac1;
                    self.z2[i] = self.z2[i] + s2 * self.alphn - s3 * self.betan;
                    self.z3[i] = self.z3[i] + s3 * self.alphn + s2 * self.betan;
                }
                sol(n, self.e1.view(), self.z1.view_mut(), self.ip1.view());
                solc(
                    n,
                    self.e2r.view(),
                    self.e2i.view(),
                    self.z2.view_mut(),
                    self.z3.view_mut(),
                    self.ip2.view(),
                );
            }
            4 => {
                // mass is a banded matrix, Jacobian a banded matrix
                for i in 0..n {
                    let mut s1 = 0.0;
                    let mut s2 = 0.0;
                    let mut s3 = 0.0;
                    for j in 0.max(i - self.mlmas)..n.min(i + self.mumas + 1) {
                        let bb = self.mass_matrix[[i - j + self.mbdiag - 1, j]];
                        s1 -= bb * self.f1[j];
                        s2 -= bb * self.f2[j];
                        s3 -= bb * self.f3[j];
                    }
                    self.z1[i] += s1 * self.fac1;
                    self.z2[i] = self.z2[i] + s2 * self.alphn - s3 * self.betan;
                    self.z3[i] = self.z3[i] + s3 * self.alphn + s2 * self.betan;
                }
                solb(
                    n,
                    self.e1.view(),
                    self.mle,
                    self.mue,
                    self.z1.view_mut(),
                    self.ip1.view(),
                );
                solbc(
                    n,
                    self.e2r.view(),
                    self.e2i.view(),
                    self.mle,
                    self.mue,
                    self.z2.view_mut(),
                    self.z3.view_mut(),
                    self.ip2.view(),
                );
            }
            5 => {
                // mass is a full matrix, Jacobian a full matrix
                for i in 0..n {
                    let mut s1 = 0.0;
                    let mut s2 = 0.0;
                    let mut s3 = 0.0;
                    for j in 0..n {
                        let bb = self.mass_matrix[[i, j]];
                        s1 -= bb * self.f1[j];
                        s2 -= bb * self.f2[j];
                        s3 -= bb * self.f3[j];
                    }
                    self.z1[i] += s1 * self.fac1;
                    self.z2[i] = self.z2[i] + s2 * self.alphn - s3 * self.betan;
                    self.z3[i] = self.z3[i] + s3 * self.alphn + s2 * self.betan;
                }
                sol(n, self.e1.view(), self.z1.view_mut(), self.ip1.view());
                solc(
                    n,
                    self.e2r.view(),
                    self.e2i.view(),
                    self.z2.view_mut(),
                    self.z3.view_mut(),
                    self.ip2.view(),
                );
            }
            7 => {
                // mass = identity, Jacobian a full matrix, Hessenberg-option
                for i in 0..n {
                    let s2 = -self.f2[i];
                    let s3 = -self.f3[i];
                    self.z1[i] -= self.f1[i] * self.fac1;
                    self.z2[i] = self.z2[i] + s2 * self.alphn - s3 * self.betan;
                    self.z3[i] = self.z3[i] + s3 * self.alphn + s2 * self.betan;
                }
                for mm1 in (0..(n - 2)).rev() {
                    let mp = n - mm1 - 2;
                    let mp1 = mp - 1;
                    let ii = self.iphes[mp] as usize;
                    if ii != mp {
                        let mut zsafe = self.z1[mp];
                        self.z1[mp] = self.z1[ii];
                        self.z1[ii] = zsafe;
                        zsafe = self.z2[mp];
                        self.z2[mp] = self.z2[ii];
                        self.z2[ii] = zsafe;
                        zsafe = self.z3[mp];
                        self.z3[mp] = self.z3[ii];
                        self.z3[ii] = zsafe;
                    }
                    for i in (mp + 1)..n {
                        let e1imp = self.dfdu[[i, mp1]];
                        self.z1[i] -= e1imp * self.z1[mp];
                        self.z2[i] -= e1imp * self.z2[mp];
                        self.z3[i] -= e1imp * self.z3[mp];
                    }
                }
                solh(n, self.e1.view(), 1, self.z1.view_mut(), self.ip1.view());
                solhc(
                    n,
                    self.e2r.view(),
                    self.e2i.view(),
                    1,
                    self.z2.view_mut(),
                    self.z3.view_mut(),
                    self.ip2.view(),
                );
                for mm1 in 0..(n - 2) {
                    let mp = n - mm1 - 2;
                    let mp1 = mp - 1;
                    for i in mp..n {
                        let e1imp = self.dfdu[[i, mp1]];
                        self.z1[i] += e1imp * self.z1[mp];
                        self.z2[i] += e1imp * self.z2[mp];
                        self.z3[i] += e1imp * self.z3[mp];
                    }
                    let ii = self.iphes[mp] as usize;
                    if ii != mp {
                        let mut zsafe = self.z1[mp];
                        self.z1[mp] = self.z1[ii];
                        self.z1[ii] = zsafe;
                        zsafe = self.z2[mp];
                        self.z2[mp] = self.z2[ii];
                        self.z2[ii] = zsafe;
                        zsafe = self.z3[mp];
                        self.z3[mp] = self.z3[ii];
                        self.z3[ii] = zsafe;
                    }
                }
            }
            11 | 12 => {
                for i in 0..n {
                    let s2 = -self.f2[i];
                    let s3 = -self.f3[i];
                    self.z1[i] -= self.f1[i] * self.fac1;
                    self.z2[i] = self.z2[i] + s2 * self.alphn - s3 * self.betan;
                    self.z3[i] = self.z3[i] + s3 * self.alphn + s2 * self.betan;
                }
            }
            13 | 14 => {
                for i in 0..self.m1 {
                    let s2 = -self.f2[i];
                    let s3 = -self.f3[i];
                    self.z1[i] -= self.f1[i] * self.fac1;
                    self.z2[i] = self.z2[i] + s2 * self.alphn - s3 * self.betan;
                    self.z3[i] = self.z3[i] + s3 * self.alphn + s2 * self.betan;
                }
                for i in 0..nm1 {
                    let mut s1 = 0.0;
                    let mut s2 = 0.0;
                    let mut s3 = 0.0;
                    for j in 0.max(i - self.mlmas)..nm1.min(i + self.mumas + 1) {
                        let bb = self.mass_matrix[[i - j + self.mbdiag - 1, j]];
                        s1 -= bb * self.f1[j + self.m1];
                        s2 -= bb * self.f2[j + self.m1];
                        s3 -= bb * self.f3[j + self.m1];
                    }
                    self.z1[i + self.m1] += s1 * self.fac1;
                    self.z2[i + self.m1] = self.z2[i + self.m1] + s2 * self.alphn - s3 * self.betan;
                    self.z3[i + self.m1] = self.z3[i + self.m1] + s3 * self.alphn + s2 * self.betan;
                }
            }
            15 => {
                for i in 0..self.m1 {
                    let s2 = -self.f2[i];
                    let s3 = -self.f3[i];
                    self.z1[i] -= self.f1[i] * self.fac1;
                    self.z2[i] = self.z2[i] + s2 * self.alphn - s3 * self.betan;
                    self.z3[i] = self.z3[i] + s3 * self.alphn + s2 * self.betan;
                }
                for i in 0..nm1 {
                    let mut s1 = 0.0;
                    let mut s2 = 0.0;
                    let mut s3 = 0.0;
                    for j in 0..nm1 {
                        let bb = self.mass_matrix[[i, j]];
                        s1 -= bb * self.f1[j + self.m1];
                        s2 -= bb * self.f2[j + self.m1];
                        s3 -= bb * self.f3[j + self.m1];
                    }
                    self.z1[i + self.m1] += s1 * self.fac1;
                    self.z2[i + self.m1] = self.z2[i + self.m1] + s2 * self.alphn - s3 * self.betan;
                    self.z3[i + self.m1] = self.z3[i + self.m1] + s3 * self.alphn + s2 * self.betan;
                }
            }
            _ => {}
        }

        match self.prob_type {
            11 | 13 | 15 => {
                let abno = self.alphn * self.alphn + self.betan * self.betan;
                let mm = self.m1 / self.m2;
                for j in 0..self.m2 {
                    let mut sum1 = 0.0;
                    let mut sum2 = 0.0;
                    let mut sum3 = 0.0;
                    for k in (0..mm).rev() {
                        let jkm = j + k * self.m2;
                        sum1 = (self.z1[jkm] + sum1) / self.fac1;
                        let sumh = (self.z2[jkm] + sum2) / abno;
                        sum3 = (self.z3[jkm] + sum3) / abno;
                        sum2 = sumh * self.alphn + sum3 * self.betan;
                        sum3 = sum3 * self.alphn - sumh * self.betan;
                        for i in 0..nm1 {
                            self.z1[i + self.m1] += self.dfdu[[i, jkm]] * sum1;
                            self.z2[i + self.m1] += self.dfdu[[i, jkm]] * sum2;
                            self.z3[i + self.m1] += self.dfdu[[i, jkm]] * sum3;
                        }
                    }
                }

                sol(
                    nm1,
                    self.e1.view(),
                    self.z1.slice_mut(s![self.m1..]),
                    self.ip1.view(),
                );
                solc(
                    nm1,
                    self.e2r.view(),
                    self.e2i.view(),
                    self.z2.slice_mut(s![self.m1..]),
                    self.z3.slice_mut(s![self.m1..]),
                    self.ip2.view(),
                );
            }
            12 | 14 => {
                let abno = self.alphn * self.alphn + self.betan * self.betan;
                let mm = self.m1 / self.m2;
                for j in 0..self.m2 {
                    let mut sum1 = 0.0;
                    let mut sum2 = 0.0;
                    let mut sum3 = 0.0;
                    for k in 0..mm {
                        let jkm = j + k * self.m2;
                        sum1 = (self.z1[jkm] + sum1) / self.fac1;
                        let sumh = (self.z2[jkm] + sum2) / abno;
                        sum3 = (self.z3[jkm] + sum3) / abno;
                        sum2 = sumh * self.alphn + sum3 * self.betan;
                        sum3 = sum3 * self.alphn - sumh * self.betan;
                        for i in 0.max(j - self.mujac)..nm1.min(j + self.mljac + 1) {
                            let ffja = self.dfdu[[i + self.mujac - j, jkm]];
                            self.z1[i + self.m1] += ffja * sum1;
                            self.z2[i + self.m1] += ffja * sum2;
                            self.z3[i + self.m1] += ffja * sum3;
                        }
                    }
                }
                solb(
                    nm1,
                    self.e1.view(),
                    self.mle,
                    self.mue,
                    self.z1.slice_mut(s![self.m1..]),
                    self.ip1.view(),
                );
                solbc(
                    nm1,
                    self.e2r.view(),
                    self.e2i.view(),
                    self.mle,
                    self.mue,
                    self.z2.slice_mut(s![self.m1..]),
                    self.z3.slice_mut(s![self.m1..]),
                    self.ip2.view(),
                );
            }
            _ => {}
        }
    }
}
