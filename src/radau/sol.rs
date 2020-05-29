use super::cache::Radau5Cache;
use crate::linalg::sol::*;
use ndarray::prelude::*;

/// Solve the linear systems for the Radau5 algorithm
pub(crate) fn linear_solve(cache: &mut Radau5Cache, alphn: f64, betan: f64, fac1: f64) {
    let n = cache.e1.nrows();
    let nm1 = n - cache.m1;

    match cache.prob_type {
        1 => {
            // mass = identity, Jacobian a full matrix
            for i in 0..n {
                let s2 = -cache.f2[i];
                let s3 = -cache.f3[i];
                cache.z1[i] -= cache.f1[i] * fac1;
                cache.z2[i] = cache.z2[i] + s2 * alphn - s3 * betan;
                cache.z3[i] = cache.z3[i] + s3 * alphn + s2 * betan;
            }
            sol(n, cache.e1.view(), cache.z1.view_mut(), cache.ip1.view());
            solc(
                n,
                cache.e2r.view(),
                cache.e2i.view(),
                cache.z2.view_mut(),
                cache.z3.view_mut(),
                cache.ip2.view(),
            );
        }
        2 => {
            // mass = identity, Jacobian a banded matrix
            for i in 0..n {
                let s2 = -cache.f2[i];
                let s3 = -cache.f3[i];
                cache.z1[i] -= cache.f1[i] * fac1;
                cache.z2[i] = cache.z2[i] + s2 * alphn - s3 * betan;
                cache.z3[i] = cache.z3[i] + s3 * alphn + s2 * betan;
            }
            solb(
                n,
                cache.e1.view(),
                cache.mle,
                cache.mue,
                cache.z1.view_mut(),
                cache.ip1.view(),
            );
            solbc(
                n,
                cache.e2r.view(),
                cache.e2i.view(),
                cache.mle,
                cache.mue,
                cache.z2.view_mut(),
                cache.z3.view_mut(),
                cache.ip2.view(),
            );
        }
        3 => {
            // mass is a banded matrix, Jacobian a full matrix
            for i in 0..n {
                let mut s1 = 0.0;
                let mut s2 = 0.0;
                let mut s3 = 0.0;
                for j in 0.max(i - cache.mlmas)..n.min(i + cache.mumas + 1) {
                    let bb = cache.mass_matrix[[i - j + cache.mbdiag - 1, j]];
                    s1 -= bb * cache.f1[j];
                    s2 -= bb * cache.f2[j];
                    s3 -= bb * cache.f3[j];
                }
                cache.z1[i] += s1 * fac1;
                cache.z2[i] = cache.z2[i] + s2 * alphn - s3 * betan;
                cache.z3[i] = cache.z3[i] + s3 * alphn + s2 * betan;
            }
            sol(n, cache.e1.view(), cache.z1.view_mut(), cache.ip1.view());
            solc(
                n,
                cache.e2r.view(),
                cache.e2i.view(),
                cache.z2.view_mut(),
                cache.z3.view_mut(),
                cache.ip2.view(),
            );
        }
        4 => {
            // mass is a banded matrix, Jacobian a banded matrix
            for i in 0..n {
                let mut s1 = 0.0;
                let mut s2 = 0.0;
                let mut s3 = 0.0;
                for j in 0.max(i - cache.mlmas)..n.min(i + cache.mumas + 1) {
                    let bb = cache.mass_matrix[[i - j + cache.mbdiag - 1, j]];
                    s1 -= bb * cache.f1[j];
                    s2 -= bb * cache.f2[j];
                    s3 -= bb * cache.f3[j];
                }
                cache.z1[i] += s1 * fac1;
                cache.z2[i] = cache.z2[i] + s2 * alphn - s3 * betan;
                cache.z3[i] = cache.z3[i] + s3 * alphn + s2 * betan;
            }
            solb(
                n,
                cache.e1.view(),
                cache.mle,
                cache.mue,
                cache.z1.view_mut(),
                cache.ip1.view(),
            );
            solbc(
                n,
                cache.e2r.view(),
                cache.e2i.view(),
                cache.mle,
                cache.mue,
                cache.z2.view_mut(),
                cache.z3.view_mut(),
                cache.ip2.view(),
            );
        }
        5 => {
            // mass is a full matrix, Jacobian a full matrix
            for i in 0..n {
                let mut s1 = 0.0;
                let mut s2 = 0.0;
                let mut s3 = 0.0;
                for j in 0..n {
                    let bb = cache.mass_matrix[[i, j]];
                    s1 -= bb * cache.f1[j];
                    s2 -= bb * cache.f2[j];
                    s3 -= bb * cache.f3[j];
                }
                cache.z1[i] += s1 * fac1;
                cache.z2[i] = cache.z2[i] + s2 * alphn - s3 * betan;
                cache.z3[i] = cache.z3[i] + s3 * alphn + s2 * betan;
            }
            sol(n, cache.e1.view(), cache.z1.view_mut(), cache.ip1.view());
            solc(
                n,
                cache.e2r.view(),
                cache.e2i.view(),
                cache.z2.view_mut(),
                cache.z3.view_mut(),
                cache.ip2.view(),
            );
        }
        7 => {
            // mass = identity, Jacobian a full matrix, Hessenberg-option
            for i in 0..n {
                let s2 = -cache.f2[i];
                let s3 = -cache.f3[i];
                cache.z1[i] -= cache.f1[i] * fac1;
                cache.z2[i] = cache.z2[i] + s2 * alphn - s3 * betan;
                cache.z3[i] = cache.z3[i] + s3 * alphn + s2 * betan;
            }
            for mm1 in 0..(n - 2) {
                let mp = n - mm1 - 2;
                let mp1 = mp - 1;
                let ii = cache.iphes[mp] as usize;
                if ii != mp {
                    let mut zsafe = cache.z1[mp];
                    cache.z1[mp] = cache.z1[ii];
                    cache.z1[ii] = zsafe;
                    zsafe = cache.z2[mp];
                    cache.z2[mp] = cache.z2[ii];
                    cache.z2[ii] = zsafe;
                    zsafe = cache.z3[mp];
                    cache.z3[mp] = cache.z3[ii];
                    cache.z3[ii] = zsafe;
                }
                for i in (mp + 1)..n {
                    let e1imp = cache.dfdu[[i, mp1]];
                    cache.z1[i] -= e1imp * cache.z1[mp];
                    cache.z2[i] -= e1imp * cache.z2[mp];
                    cache.z3[i] -= e1imp * cache.z3[mp];
                }
            }
            solh(n, cache.e1.view(), 1, cache.z1.view_mut(), cache.ip1.view());
            solhc(
                n,
                cache.e2r.view(),
                cache.e2i.view(),
                1,
                cache.z2.view_mut(),
                cache.z3.view_mut(),
                cache.ip2.view(),
            );
            for mm1 in 0..(n - 2) {
                let mp = n - mm1 - 2;
                let mp1 = mp - 1;
                for i in mp..n {
                    let e1imp = cache.dfdu[[i, mp1]];
                    cache.z1[i] += e1imp * cache.z1[mp];
                    cache.z2[i] += e1imp * cache.z2[mp];
                    cache.z3[i] += e1imp * cache.z3[mp];
                }
                let ii = cache.iphes[mp] as usize;
                if ii != mp {
                    let mut zsafe = cache.z1[mp];
                    cache.z1[mp] = cache.z1[ii];
                    cache.z1[ii] = zsafe;
                    zsafe = cache.z2[mp];
                    cache.z2[mp] = cache.z2[ii];
                    cache.z2[ii] = zsafe;
                    zsafe = cache.z3[mp];
                    cache.z3[mp] = cache.z3[ii];
                    cache.z3[ii] = zsafe;
                }
            }
        }
        11 | 12 => {
            for i in 0..n {
                let s2 = -cache.f2[i];
                let s3 = -cache.f3[i];
                cache.z1[i] -= cache.f1[i] * fac1;
                cache.z2[i] = cache.z2[i] + s2 * alphn - s3 * betan;
                cache.z3[i] = cache.z3[i] + s3 * alphn + s2 * betan;
            }
        }
        13 | 14 => {
            for i in 0..cache.m1 {
                let s2 = -cache.f2[i];
                let s3 = -cache.f3[i];
                cache.z1[i] -= cache.f1[i] * fac1;
                cache.z2[i] = cache.z2[i] + s2 * alphn - s3 * betan;
                cache.z3[i] = cache.z3[i] + s3 * alphn + s2 * betan;
            }
            for i in 0..nm1 {
                let mut s1 = 0.0;
                let mut s2 = 0.0;
                let mut s3 = 0.0;
                for j in 0.max(i - cache.mlmas)..nm1.min(i + cache.mumas + 1) {
                    let bb = cache.mass_matrix[[i - j + cache.mbdiag - 1, j]];
                    s1 -= bb * cache.f1[j + cache.m1];
                    s2 -= bb * cache.f2[j + cache.m1];
                    s3 -= bb * cache.f3[j + cache.m1];
                }
                cache.z1[i + cache.m1] += s1 * fac1;
                cache.z2[i + cache.m1] = cache.z2[i + cache.m1] + s2 * alphn - s3 * betan;
                cache.z3[i + cache.m1] = cache.z3[i + cache.m1] + s3 * alphn + s2 * betan;
            }
        }
        15 => {
            for i in 0..cache.m1 {
                let s2 = -cache.f2[i];
                let s3 = -cache.f3[i];
                cache.z1[i] -= cache.f1[i] * fac1;
                cache.z2[i] = cache.z2[i] + s2 * alphn - s3 * betan;
                cache.z3[i] = cache.z3[i] + s3 * alphn + s2 * betan;
            }
            for i in 0..nm1 {
                let mut s1 = 0.0;
                let mut s2 = 0.0;
                let mut s3 = 0.0;
                for j in 0..nm1 {
                    let bb = cache.mass_matrix[[i, j]];
                    s1 -= bb * cache.f1[j + cache.m1];
                    s2 -= bb * cache.f2[j + cache.m1];
                    s3 -= bb * cache.f3[j + cache.m1];
                }
                cache.z1[i + cache.m1] += s1 * fac1;
                cache.z2[i + cache.m1] = cache.z2[i + cache.m1] + s2 * alphn - s3 * betan;
                cache.z3[i + cache.m1] = cache.z3[i + cache.m1] + s3 * alphn + s2 * betan;
            }
        }
        _ => {}
    }

    match cache.prob_type {
        11 | 13 | 15 => {
            let abno = alphn * alphn + betan * betan;
            let mm = cache.m1 / cache.m2;
            for j in 0..cache.m2 {
                let mut sum1 = 0.0;
                let mut sum2 = 0.0;
                let mut sum3 = 0.0;
                for k in 0..mm {
                    let jkm = j + k * cache.m2;
                    sum1 = (cache.z1[jkm] + sum1) / fac1;
                    let sumh = (cache.z2[jkm] + sum2) / abno;
                    sum3 = (cache.z3[jkm] + sum3) / abno;
                    sum2 = sumh * alphn + sum3 * betan;
                    sum3 = sum3 * alphn - sumh * betan;
                    for i in 0..nm1 {
                        cache.z1[i + cache.m1] += cache.dfdu[[i, jkm]] * sum1;
                        cache.z2[i + cache.m1] += cache.dfdu[[i, jkm]] * sum2;
                        cache.z3[i + cache.m1] += cache.dfdu[[i, jkm]] * sum3;
                    }
                }
            }

            sol(
                nm1,
                cache.e1.view(),
                cache.z1.slice_mut(s![cache.m1..]),
                cache.ip1.view(),
            );
            solc(
                nm1,
                cache.e2r.view(),
                cache.e2i.view(),
                cache.z2.slice_mut(s![cache.m1..]),
                cache.z3.slice_mut(s![cache.m1..]),
                cache.ip2.view(),
            );
        }
        12 | 14 => {
            let abno = alphn * alphn + betan * betan;
            let mm = cache.m1 / cache.m2;
            for j in 0..cache.m2 {
                let mut sum1 = 0.0;
                let mut sum2 = 0.0;
                let mut sum3 = 0.0;
                for k in 0..mm {
                    let jkm = j + k * cache.m2;
                    sum1 = (cache.z1[jkm] + sum1) / fac1;
                    let sumh = (cache.z2[jkm] + sum2) / abno;
                    sum3 = (cache.z3[jkm] + sum3) / abno;
                    sum2 = sumh * alphn + sum3 * betan;
                    sum3 = sum3 * alphn - sumh * betan;
                    for i in 0.max(j - cache.mujac)..nm1.min(j + cache.mljac + 1) {
                        let ffja = cache.dfdu[[i + cache.mujac - j, jkm]];
                        cache.z1[i + cache.m1] += ffja * sum1;
                        cache.z2[i + cache.m1] += ffja * sum2;
                        cache.z3[i + cache.m1] += ffja * sum3;
                    }
                }
            }
            solb(
                nm1,
                cache.e1.view(),
                cache.mle,
                cache.mue,
                cache.z1.slice_mut(s![cache.m1..]),
                cache.ip1.view(),
            );
            solbc(
                nm1,
                cache.e2r.view(),
                cache.e2i.view(),
                cache.mle,
                cache.mue,
                cache.z2.slice_mut(s![cache.m1..]),
                cache.z3.slice_mut(s![cache.m1..]),
                cache.ip2.view(),
            );
        }
        _ => {}
    }
}
