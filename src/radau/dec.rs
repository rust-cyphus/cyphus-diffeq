use super::alg::Radau5;
use super::cache::Radau5Cache;
use crate::linalg::dec::*;

impl Radau5 {
    /// Perform a decomposition on the real matrices used for solving ODE using
    /// Radau5
    pub(crate) fn decomp_real(&self, cache: &mut Radau5Cache) -> usize {
        let n = cache.e1.nrows();
        let nm1 = n - cache.m1;
        let mut ier = 0;

        match cache.prob_type {
            1 => {
                for j in 0..n {
                    for i in 0..n {
                        cache.e1[[i, j]] = -cache.dfdu[[i, j]];
                    }
                    cache.e1[[j, j]] += cache.fac1;
                }
                ier = dec(n, cache.e1.view_mut(), cache.ip1.view_mut());
            }
            2 => {
                for j in 0..n {
                    for i in 0..cache.mbjac {
                        cache.e1[[i + cache.mle, j]] = -cache.dfdu[[i, j]];
                    }
                    cache.e1[[cache.mdiag, j]] += cache.fac1;
                }
                ier = decb(
                    n,
                    cache.e1.view_mut(),
                    cache.mle,
                    cache.mue,
                    cache.ip1.view_mut(),
                );
            }
            3 => {
                for j in 0..n {
                    for i in 0..n {
                        cache.e1[[i, j]] = -cache.dfdu[[i, j]];
                    }
                    for i in 0.max(j - cache.mumas)..n.min(j + cache.mlmas + 1) {
                        cache.e1[[i, j]] +=
                            cache.fac1 * cache.mass_matrix[[i - j + cache.mbdiag - 1, j]];
                    }
                }
                ier = dec(n, cache.e1.view_mut(), cache.ip1.view_mut());
            }
            4 => {
                for j in 0..n {
                    for i in 0..cache.mbjac {
                        cache.e1[[i + cache.mle, j]] = -cache.dfdu[[i, j]];
                    }
                    for i in 0..cache.mbb {
                        cache.e1[[i + cache.mdiff, j]] += cache.fac1 * cache.mass_matrix[[i, j]];
                    }
                }
                ier = decb(
                    n,
                    cache.e1.view_mut(),
                    cache.mle,
                    cache.mue,
                    cache.ip1.view_mut(),
                );
            }
            5 => {
                // mass is a full matrix, Jacobian a full matrix
                for j in 0..n {
                    for i in 0..n {
                        cache.e1[[i, j]] =
                            cache.mass_matrix[[i, j]] * cache.fac1 - cache.dfdu[[i, j]];
                    }
                }
                ier = dec(n, cache.e1.view_mut(), cache.ip1.view_mut());
            }
            7 => {
                // mass = identity, Jacobian a full matrix, Hessenberg-option
                if cache.calhes {
                    elmhes(n, 0, n, cache.dfdu.view_mut(), cache.iphes.view_mut());
                }
                cache.calhes = false;
                for j in 0..(n - 1) {
                    cache.e1[[j + 1, j]] = -cache.dfdu[[j + 1, j]];
                }
                for j in 0..n {
                    for i in 0..(j + 1) {
                        cache.e1[[i, j]] = -cache.dfdu[[i, j]];
                    }
                    cache.e1[[j, j]] += cache.fac1;
                }
                ier = dech(n, cache.e1.view_mut(), 1, cache.ip1.view_mut());
            }
            11 => {
                for j in 0..nm1 {
                    for i in 0..nm1 {
                        cache.e1[[i, j]] = -cache.dfdu[[i, j + cache.m1]];
                    }
                    cache.e1[[j, j]] += cache.fac1;
                }
            }
            12 => {
                for j in 0..nm1 {
                    for i in 0..cache.mbjac {
                        cache.e1[[i + cache.mle, j]] = -cache.dfdu[[i, j + cache.m1]];
                    }
                    cache.e1[[cache.mdiag, j]] += cache.fac1;
                }
            }
            13 => {
                for j in 0..nm1 {
                    for i in 0..nm1 {
                        cache.e1[[i, j]] = -cache.dfdu[[i, j + cache.m1]];
                    }
                    for i in 0.max(j - cache.mumas)..n.min(j + cache.mlmas + 1) {
                        cache.e1[[i, j]] +=
                            cache.fac1 * cache.mass_matrix[[i - j + cache.mbdiag - 1, j]];
                    }
                }
            }
            14 => {
                for j in 0..nm1 {
                    for i in 0..cache.mbjac {
                        cache.e1[[i + cache.mle, j]] = -cache.dfdu[[i, j + cache.m1]];
                    }
                    for i in 0..cache.mbb {
                        cache.e1[[i + cache.mdiff, j]] += cache.fac1 * cache.mass_matrix[[i, j]];
                    }
                }
            }
            15 => {
                for j in 0..nm1 {
                    for i in 0..nm1 {
                        cache.e1[[i, j]] =
                            cache.mass_matrix[[i, j]] * cache.fac1 - cache.dfdu[[i, j + cache.m1]];
                    }
                }
            }
            _ => {}
        }

        match cache.prob_type {
            11 | 13 | 15 => {
                let mm = cache.m1 / cache.m2;
                for j in 0..cache.m2 {
                    for i in 0..nm1 {
                        let mut sum = 0.0;
                        for k in 0..mm {
                            sum = (sum + cache.dfdu[[i, j + k * cache.m2]]) / cache.fac1;
                        }
                        cache.e1[[i, j]] -= sum;
                    }
                }
                ier = dec(nm1, cache.e1.view_mut(), cache.ip1.view_mut());
            }
            12 | 14 => {
                let mm = cache.m1 / cache.m2;
                for j in 0..cache.m2 {
                    for i in 0..cache.mbjac {
                        let mut sum = 0.0;
                        for k in 0..mm {
                            sum = (sum + cache.dfdu[[i, j + k * cache.m2]]) / cache.fac1;
                        }
                        cache.e1[[i + cache.mle, j]] -= sum;
                    }
                }
                ier = decb(
                    nm1,
                    cache.e1.view_mut(),
                    cache.mle,
                    cache.mue,
                    cache.ip1.view_mut(),
                );
            }
            _ => {}
        }

        ier
    }

    /// Perform a decomposition on the complex matrices used for solving ODE using
    /// Radau5
    pub(crate) fn decomp_complex(&self, cache: &mut Radau5Cache) -> usize {
        let n = cache.e1.nrows();
        let nm1 = n - cache.m1;
        let mut ier = 0;

        match cache.prob_type {
            1 => {
                // mass = identity, Jacobian a full matrix
                for j in 0..n {
                    for i in 0..n {
                        cache.e2r[[i, j]] = -cache.dfdu[[i, j]];
                        cache.e2i[[i, j]] = 0.0;
                    }
                    cache.e2r[[j, j]] += cache.alphn;
                    cache.e2i[[j, j]] = cache.betan;
                }
                ier = decc(
                    n,
                    cache.e2r.view_mut(),
                    cache.e2i.view_mut(),
                    cache.ip2.view_mut(),
                );
            }
            2 => {
                // mass = identiy, Jacobian a banded matrix
                for j in 0..n {
                    for i in 0..cache.mbjac {
                        cache.e2r[[i + cache.mle, j]] = -cache.dfdu[[i, j]];
                        cache.e2i[[i + cache.mle, j]] = 0.0;
                    }
                    cache.e2r[[cache.mdiag, j]] += cache.alphn;
                    cache.e2i[[cache.mdiag, j]] = cache.betan;
                }
                ier = decbc(
                    n,
                    cache.e2r.view_mut(),
                    cache.e2i.view_mut(),
                    cache.mle,
                    cache.mue,
                    cache.ip2.view_mut(),
                );
            }
            3 => {
                // mass is a banded matrix, Jacobian a full matrix
                for j in 0..n {
                    for i in 0..n {
                        cache.e2r[[i, j]] = -cache.dfdu[[i, j]];
                        cache.e2i[[i, j]] = 0.0;
                    }
                }
                for j in 0..n {
                    for i in 0.max(j - cache.mumas)..n.min(j + cache.mlmas + 1) {
                        let bb = cache.mass_matrix[[i - j + cache.mbdiag - 1, j]];
                        cache.e2r[[i, j]] += cache.alphn * bb;
                        cache.e2i[[i, j]] = cache.betan * bb;
                    }
                }
                ier = decc(
                    n,
                    cache.e2r.view_mut(),
                    cache.e2i.view_mut(),
                    cache.ip2.view_mut(),
                );
            }
            4 => {
                // mass is a banded matrix, Jacobian a banded matrix
                for j in 0..n {
                    for i in 0..cache.mbjac {
                        cache.e2r[[i + cache.mle, j]] = -cache.dfdu[[i, j]];
                        cache.e2i[[i + cache.mle, j]] = 0.0;
                    }
                    for i in 0.max(cache.mumas - j)..cache.mbb.min(cache.mumas - j + n) {
                        let bb = cache.mass_matrix[[i, j]];
                        cache.e2r[[i + cache.mdiff, j]] += cache.alphn * bb;
                        cache.e2i[[i + cache.mdiff, j]] = cache.betan * bb;
                    }
                }
                ier = decbc(
                    n,
                    cache.e2r.view_mut(),
                    cache.e2i.view_mut(),
                    cache.mle,
                    cache.mue,
                    cache.ip2.view_mut(),
                );
            }
            5 => {
                // mass is a full matrix, Jacobian a full matrix
                for j in 0..n {
                    for i in 0..n {
                        let bb = cache.mass_matrix[[i, j]];
                        cache.e2r[[i, j]] = cache.alphn * bb - cache.dfdu[[i, j]];
                        cache.e2i[[i, j]] = cache.betan * bb;
                    }
                }
                ier = decc(
                    n,
                    cache.e2r.view_mut(),
                    cache.e2i.view_mut(),
                    cache.ip2.view_mut(),
                );
            }
            7 => {
                // mass = identity, Jacobian a full matrix, Hessenberg-option
                for j in 0..(n - 1) {
                    cache.e2r[[j + 1, j]] = -cache.dfdu[[j + 1, j]];
                    cache.e2i[[j + 1, j]] = 0.0;
                }
                for j in 0..n {
                    for i in 0..(j + 1) {
                        cache.e2i[[i, j]] = 0.0;
                        cache.e2r[[i, j]] = -cache.dfdu[[i, j]];
                    }
                    cache.e2r[[j, j]] += cache.alphn;
                    cache.e2i[[j, j]] = cache.betan;
                }
                ier = dechc(
                    n,
                    cache.e2r.view_mut(),
                    cache.e2i.view_mut(),
                    1,
                    cache.ip2.view_mut(),
                );
            }
            11 => {
                // mass = identity, Jacobian a full matrix, second order
                for j in 0..nm1 {
                    for i in 0..nm1 {
                        cache.e2r[[i, j]] = -cache.dfdu[[i, j + cache.m1]];
                        cache.e2i[[i, j]] = 0.0;
                    }
                    cache.e2r[[j, j]] += cache.alphn;
                    cache.e2i[[j, j]] = cache.betan;
                }
            }
            12 => {
                // mass = identity, Jacobian a banded matrix, second order
                for j in 0..nm1 {
                    for i in 0..cache.mbjac {
                        cache.e2r[[i + cache.mle, j]] = -cache.dfdu[[i, j + cache.m1]];
                        cache.e2i[[i + cache.mle, j]] = 0.0;
                    }
                    cache.e2r[[cache.mdiag, j]] += cache.alphn;
                    cache.e2i[[cache.mdiag, j]] += cache.betan;
                }
            }
            13 => {
                // mass is a banded matrix, Jacobian a full matrix, second order
                for j in 0..nm1 {
                    for i in 0..nm1 {
                        cache.e2r[[i, j]] = -cache.dfdu[[i, j + cache.m1]];
                        cache.e2i[[i, j]] = 0.0;
                    }
                    for i in 0.max(j - cache.mumas)..nm1.min(j + cache.mlmas + 1) {
                        let ffma = cache.mass_matrix[[i - j + cache.mbdiag - 1, j]];
                        cache.e2r[[j, j]] += cache.alphn * ffma;
                        cache.e2i[[j, j]] += cache.betan * ffma;
                    }
                }
            }
            14 => {
                // mass is a banded matrix, Jacobian a banded matrix, second order
                for j in 0..nm1 {
                    for i in 0..cache.mbjac {
                        cache.e2r[[i + cache.mle, j]] = -cache.dfdu[[i, j + cache.m1]];
                        cache.e2i[[i + cache.mle, j]] = 0.0;
                    }
                    for i in 0..cache.mbb {
                        let ffma = cache.mass_matrix[[i, j]];
                        cache.e2r[[i + cache.mdiff, j]] += cache.alphn * ffma;
                        cache.e2i[[i + cache.mdiff, j]] += cache.betan * ffma;
                    }
                }
            }
            15 => {
                // mass is a full matrix, Jacobian a full matrix, second order
                for j in 0..nm1 {
                    for i in 0..nm1 {
                        cache.e2r[[i, j]] =
                            cache.alphn * cache.mass_matrix[[i, j]] - cache.dfdu[[i, j + cache.m1]];
                        cache.e2i[[i, j]] = cache.betan * cache.mass_matrix[[i, j]];
                    }
                }
            }
            _ => {}
        }

        match cache.prob_type {
            11 | 13 | 15 => {
                let mm = cache.m1 / cache.m2;
                let abno = cache.alphn * cache.alphn + cache.betan * cache.betan;
                let alp = cache.alphn / abno;
                let bet = cache.betan / abno;
                for j in 0..cache.m2 {
                    for i in 0..nm1 {
                        let mut sumr = 0.0;
                        let mut sumi = 0.0;
                        for k in 0..mm {
                            let sums = sumr + cache.dfdu[[i, j + k * cache.m2]];
                            sumr = sums * alp + sumi * bet;
                            sumi = sumi * alp - sums * bet;
                        }
                        cache.e2r[[i, j]] -= sumr;
                        cache.e2i[[i, j]] -= sumi;
                    }
                }
                ier = decc(
                    nm1,
                    cache.e2r.view_mut(),
                    cache.e2i.view_mut(),
                    cache.ip2.view_mut(),
                );
            }
            12 | 14 => {
                let mm = cache.m1 / cache.m2;
                let abno = cache.alphn * cache.alphn + cache.betan * cache.betan;
                let alp = cache.alphn / abno;
                let bet = cache.betan / abno;
                for j in 0..cache.m2 {
                    for i in 0..cache.mbjac {
                        let mut sumr = 0.0;
                        let mut sumi = 0.0;
                        for k in 0..mm {
                            let sums = sumr + cache.dfdu[[i, j + k * cache.m2]];
                            sumr = sums * alp + sumi * bet;
                            sumi = sumi * alp - sums * bet;
                        }
                        cache.e2r[[i + cache.mle, j]] -= sumr;
                        cache.e2i[[i + cache.mle, j]] -= sumi;
                    }
                }
                ier = decbc(
                    nm1,
                    cache.e2r.view_mut(),
                    cache.e2i.view_mut(),
                    cache.mle,
                    cache.mue,
                    cache.ip2.view_mut(),
                );
            }
            _ => {}
        }

        ier
    }
}
