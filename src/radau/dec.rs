use super::cache::Radau5Cache;
use crate::linalg::dec::*;
use ndarray::prelude::*;

impl Radau5Cache {
    /// Perform a decomposition on the real matrices used for solving ODE using
    /// Radau5
    pub(crate) fn decomp_real(&mut self) -> usize {
        let n = self.e1.nrows();
        let nm1 = n - self.m1;
        let mut ier = 0;

        match self.prob_type {
            1 => {
                for j in 0..n {
                    for i in 0..n {
                        self.e1[[i, j]] = -self.dfdu[[i, j]];
                    }
                    self.e1[[j, j]] += self.fac1;
                }
                ier = dec(n, self.e1.view_mut(), self.ip1.view_mut());
            }
            2 => {
                for j in 0..n {
                    for i in 0..self.mbjac {
                        self.e1[[i + self.mle, j]] = -self.dfdu[[i, j]];
                    }
                    self.e1[[self.mdiag, j]] += self.fac1;
                }
                ier = decb(
                    n,
                    self.e1.view_mut(),
                    self.mle,
                    self.mue,
                    self.ip1.view_mut(),
                );
            }
            3 => {
                for j in 0..n {
                    for i in 0..n {
                        self.e1[[i, j]] = -self.dfdu[[i, j]];
                    }
                    for i in 0.max(j - self.mumas)..n.min(j + self.mlmas + 1) {
                        self.e1[[i, j]] +=
                            self.fac1 * self.mass_matrix[[i - j + self.mbdiag - 1, j]];
                    }
                }
                ier = dec(n, self.e1.view_mut(), self.ip1.view_mut());
            }
            4 => {
                for j in 0..n {
                    for i in 0..self.mbjac {
                        self.e1[[i + self.mle, j]] = -self.dfdu[[i, j]];
                    }
                    for i in 0..self.mbb {
                        self.e1[[i + self.mdiff, j]] += self.fac1 * self.mass_matrix[[i, j]];
                    }
                }
                ier = decb(
                    n,
                    self.e1.view_mut(),
                    self.mle,
                    self.mue,
                    self.ip1.view_mut(),
                );
            }
            5 => {
                // mass is a full matrix, Jacobian a full matrix
                for j in 0..n {
                    for i in 0..n {
                        self.e1[[i, j]] = self.mass_matrix[[i, j]] * self.fac1 - self.dfdu[[i, j]];
                    }
                }
                ier = dec(n, self.e1.view_mut(), self.ip1.view_mut());
            }
            7 => {
                // mass = identity, Jacobian a full matrix, Hessenberg-option
                if self.calhes {
                    elmhes(n, 0, n, self.dfdu.view_mut(), self.iphes.view_mut());
                }
                self.calhes = false;
                for j in 0..(n - 1) {
                    self.e1[[j + 1, j]] = -self.dfdu[[j + 1, j]];
                }
                for j in 0..n {
                    for i in 0..(j + 1) {
                        self.e1[[i, j]] = -self.dfdu[[i, j]];
                    }
                    self.e1[[j, j]] += self.fac1;
                }
                ier = dech(n, self.e1.view_mut(), 1, self.ip1.view_mut());
            }
            11 => {
                for j in 0..nm1 {
                    for i in 0..nm1 {
                        self.e1[[i, j]] = -self.dfdu[[i, j + self.m1]];
                    }
                    self.e1[[j, j]] += self.fac1;
                }
            }
            12 => {
                for j in 0..nm1 {
                    for i in 0..self.mbjac {
                        self.e1[[i + self.mle, j]] = -self.dfdu[[i, j + self.m1]];
                    }
                    self.e1[[self.mdiag, j]] += self.fac1;
                }
            }
            13 => {
                for j in 0..nm1 {
                    for i in 0..nm1 {
                        self.e1[[i, j]] = -self.dfdu[[i, j + self.m1]];
                    }
                    for i in 0.max(j - self.mumas)..n.min(j + self.mlmas + 1) {
                        self.e1[[i, j]] +=
                            self.fac1 * self.mass_matrix[[i - j + self.mbdiag - 1, j]];
                    }
                }
            }
            14 => {
                for j in 0..nm1 {
                    for i in 0..self.mbjac {
                        self.e1[[i + self.mle, j]] = -self.dfdu[[i, j + self.m1]];
                    }
                    for i in 0..self.mbb {
                        self.e1[[i + self.mdiff, j]] += self.fac1 * self.mass_matrix[[i, j]];
                    }
                }
            }
            15 => {
                for j in 0..nm1 {
                    for i in 0..nm1 {
                        self.e1[[i, j]] =
                            self.mass_matrix[[i, j]] * self.fac1 - self.dfdu[[i, j + self.m1]];
                    }
                }
            }
            _ => {}
        }

        match self.prob_type {
            11 | 13 | 15 => {
                let mm = self.m1 / self.m2;
                for j in 0..self.m2 {
                    for i in 0..nm1 {
                        let mut sum = 0.0;
                        for k in 0..mm {
                            sum = (sum + self.dfdu[[i, j + k * self.m2]]) / self.fac1;
                        }
                        self.e1[[i, j]] -= sum;
                    }
                }
                ier = dec(nm1, self.e1.view_mut(), self.ip1.view_mut());
            }
            12 | 14 => {
                let mm = self.m1 / self.m2;
                for j in 0..self.m2 {
                    for i in 0..self.mbjac {
                        let mut sum = 0.0;
                        for k in 0..mm {
                            sum = (sum + self.dfdu[[i, j + k * self.m2]]) / self.fac1;
                        }
                        self.e1[[i + self.mle, j]] -= sum;
                    }
                }
                ier = decb(
                    nm1,
                    self.e1.view_mut(),
                    self.mle,
                    self.mue,
                    self.ip1.view_mut(),
                );
            }
            _ => {}
        }

        ier
    }

    /// Perform a decomposition on the complex matrices used for solving ODE using
    /// Radau5
    pub(crate) fn decomp_complex(&mut self) -> usize {
        let n = self.e1.nrows();
        let nm1 = n - self.m1;
        let mut ier = 0;

        match self.prob_type {
            1 => {
                // mass = identity, Jacobian a full matrix
                for j in 0..n {
                    for i in 0..n {
                        self.e2r[[i, j]] = -self.dfdu[[i, j]];
                        self.e2i[[i, j]] = 0.0;
                    }
                    self.e2r[[j, j]] += self.alphn;
                    self.e2i[[j, j]] = self.betan;
                }
                ier = decc(
                    n,
                    self.e2r.view_mut(),
                    self.e2i.view_mut(),
                    self.ip2.view_mut(),
                );
            }
            2 => {
                // mass = identiy, Jacobian a banded matrix
                for j in 0..n {
                    for i in 0..self.mbjac {
                        self.e2r[[i + self.mle, j]] = -self.dfdu[[i, j]];
                        self.e2i[[i + self.mle, j]] = 0.0;
                    }
                    self.e2r[[self.mdiag, j]] += self.alphn;
                    self.e2i[[self.mdiag, j]] = self.betan;
                }
                ier = decbc(
                    n,
                    self.e2r.view_mut(),
                    self.e2i.view_mut(),
                    self.mle,
                    self.mue,
                    self.ip2.view_mut(),
                );
            }
            3 => {
                // mass is a banded matrix, Jacobian a full matrix
                for j in 0..n {
                    for i in 0..n {
                        self.e2r[[i, j]] = -self.dfdu[[i, j]];
                        self.e2i[[i, j]] = 0.0;
                    }
                }
                for j in 0..n {
                    for i in 0.max(j - self.mumas)..n.min(j + self.mlmas + 1) {
                        let bb = self.mass_matrix[[i - j + self.mbdiag - 1, j]];
                        self.e2r[[i, j]] += self.alphn * bb;
                        self.e2i[[i, j]] = self.betan * bb;
                    }
                }
                ier = decc(
                    n,
                    self.e2r.view_mut(),
                    self.e2i.view_mut(),
                    self.ip2.view_mut(),
                );
            }
            4 => {
                // mass is a banded matrix, Jacobian a banded matrix
                for j in 0..n {
                    for i in 0..self.mbjac {
                        self.e2r[[i + self.mle, j]] = -self.dfdu[[i, j]];
                        self.e2i[[i + self.mle, j]] = 0.0;
                    }
                    for i in 0.max(self.mumas - j)..self.mbb.min(self.mumas - j + n) {
                        let bb = self.mass_matrix[[i, j]];
                        self.e2r[[i + self.mdiff, j]] += self.alphn * bb;
                        self.e2i[[i + self.mdiff, j]] = self.betan * bb;
                    }
                }
                ier = decbc(
                    n,
                    self.e2r.view_mut(),
                    self.e2i.view_mut(),
                    self.mle,
                    self.mue,
                    self.ip2.view_mut(),
                );
            }
            5 => {
                // mass is a full matrix, Jacobian a full matrix
                for j in 0..n {
                    for i in 0..n {
                        let bb = self.mass_matrix[[i, j]];
                        self.e2r[[i, j]] = self.alphn * bb - self.dfdu[[i, j]];
                        self.e2i[[i, j]] = self.betan * bb;
                    }
                }
                ier = decc(
                    n,
                    self.e2r.view_mut(),
                    self.e2i.view_mut(),
                    self.ip2.view_mut(),
                );
            }
            7 => {
                // mass = identity, Jacobian a full matrix, Hessenberg-option
                for j in 0..(n - 1) {
                    self.e2r[[j + 1, j]] = -self.dfdu[[j + 1, j]];
                    self.e2i[[j + 1, j]] = 0.0;
                }
                for j in 0..n {
                    for i in 0..(j + 1) {
                        self.e2i[[i, j]] = 0.0;
                        self.e2r[[i, j]] = -self.dfdu[[i, j]];
                    }
                    self.e2r[[j, j]] += self.alphn;
                    self.e2i[[j, j]] = self.betan;
                }
                ier = dechc(
                    n,
                    self.e2r.view_mut(),
                    self.e2i.view_mut(),
                    1,
                    self.ip2.view_mut(),
                );
            }
            11 => {
                // mass = identity, Jacobian a full matrix, second order
                for j in 0..nm1 {
                    for i in 0..nm1 {
                        self.e2r[[i, j]] = -self.dfdu[[i, j + self.m1]];
                        self.e2i[[i, j]] = 0.0;
                    }
                    self.e2r[[j, j]] += self.alphn;
                    self.e2i[[j, j]] = self.betan;
                }
            }
            12 => {
                // mass = identity, Jacobian a banded matrix, second order
                for j in 0..nm1 {
                    for i in 0..self.mbjac {
                        self.e2r[[i + self.mle, j]] = -self.dfdu[[i, j + self.m1]];
                        self.e2i[[i + self.mle, j]] = 0.0;
                    }
                    self.e2r[[self.mdiag, j]] += self.alphn;
                    self.e2i[[self.mdiag, j]] += self.betan;
                }
            }
            13 => {
                // mass is a banded matrix, Jacobian a full matrix, second order
                for j in 0..nm1 {
                    for i in 0..nm1 {
                        self.e2r[[i, j]] = -self.dfdu[[i, j + self.m1]];
                        self.e2i[[i, j]] = 0.0;
                    }
                    for i in 0.max(j - self.mumas)..nm1.min(j + self.mlmas + 1) {
                        let ffma = self.mass_matrix[[i - j + self.mbdiag - 1, j]];
                        self.e2r[[j, j]] += self.alphn * ffma;
                        self.e2i[[j, j]] += self.betan * ffma;
                    }
                }
            }
            14 => {
                // mass is a banded matrix, Jacobian a banded matrix, second order
                for j in 0..nm1 {
                    for i in 0..self.mbjac {
                        self.e2r[[i + self.mle, j]] = -self.dfdu[[i, j + self.m1]];
                        self.e2i[[i + self.mle, j]] = 0.0;
                    }
                    for i in 0..self.mbb {
                        let ffma = self.mass_matrix[[i, j]];
                        self.e2r[[i + self.mdiff, j]] += self.alphn * ffma;
                        self.e2i[[i + self.mdiff, j]] += self.betan * ffma;
                    }
                }
            }
            15 => {
                // mass is a full matrix, Jacobian a full matrix, second order
                for j in 0..nm1 {
                    for i in 0..nm1 {
                        self.e2r[[i, j]] =
                            self.alphn * self.mass_matrix[[i, j]] - self.dfdu[[i, j + self.m1]];
                        self.e2i[[i, j]] = self.betan * self.mass_matrix[[i, j]];
                    }
                }
            }
            _ => {}
        }

        match self.prob_type {
            11 | 13 | 15 => {
                let mm = self.m1 / self.m2;
                let abno = self.alphn * self.alphn + self.betan * self.betan;
                let alp = self.alphn / abno;
                let bet = self.betan / abno;
                for j in 0..self.m2 {
                    for i in 0..nm1 {
                        let mut sumr = 0.0;
                        let mut sumi = 0.0;
                        for k in 0..mm {
                            let sums = sumr + self.dfdu[[i, j + k * self.m2]];
                            sumr = sums * alp + sumi * bet;
                            sumi = sumi * alp - sums * bet;
                        }
                        self.e2r[[i, j]] -= sumr;
                        self.e2i[[i, j]] -= sumi;
                    }
                }
                ier = decc(
                    nm1,
                    self.e2r.view_mut(),
                    self.e2i.view_mut(),
                    self.ip2.view_mut(),
                );
            }
            12 | 14 => {
                let mm = self.m1 / self.m2;
                let abno = self.alphn * self.alphn + self.betan * self.betan;
                let alp = self.alphn / abno;
                let bet = self.betan / abno;
                for j in 0..self.m2 {
                    for i in 0..self.mbjac {
                        let mut sumr = 0.0;
                        let mut sumi = 0.0;
                        for k in 0..mm {
                            let sums = sumr + self.dfdu[[i, j + k * self.m2]];
                            sumr = sums * alp + sumi * bet;
                            sumi = sumi * alp - sums * bet;
                        }
                        self.e2r[[i + self.mle, j]] -= sumr;
                        self.e2i[[i + self.mle, j]] -= sumi;
                    }
                }
                ier = decbc(
                    nm1,
                    self.e2r.view_mut(),
                    self.e2i.view_mut(),
                    self.mle,
                    self.mue,
                    self.ip2.view_mut(),
                );
            }
            _ => {}
        }

        ier
    }
}
