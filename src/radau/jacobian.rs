use crate::de_integrator::ODEIntegrator;
use crate::radau::cache::Radau5Cache;
use ndarray::prelude::*;

impl Radau5Cache {
    pub(crate) fn compute_jacobian<F, J>(&mut self, integrator: &mut ODEIntegrator<F, J>)
    where
        F: Fn(ArrayViewMut1<f64>, ArrayView1<f64>, f64),
        J: Fn(ArrayViewMut2<f64>, ArrayView1<f64>, f64),
    {
        integrator.stats.jacobian_evals += 1;
        if integrator.analytical_jac {
            (integrator.dfdu)(self.dfdu.view_mut(), integrator.u.view(), integrator.t);
        } else {
            let n = self.e1.nrows();
            if self.jband {
                // Jacobian is banded
                let mujacp = self.mujac + 1;
                let md = self.mbjac.min(self.m2);
                for mm1 in 0..(self.m1 / self.m2 as usize + 1) {
                    for k in 0..md {
                        let mut j = k + mm1 * self.m2;
                        loop {
                            self.f1[j] = integrator.u[j];
                            self.f2[j] =
                                (f64::EPSILON * 1.0e-5f64.max(integrator.u[j].abs())).sqrt();
                            integrator.u[j] += self.f2[j];
                            j += md;
                            if j > (mm1 + 1) * self.m2 - 1 {
                                break;
                            }
                        }
                        (integrator.dudt)(self.cont.view_mut(), integrator.u.view(), integrator.t);
                        j = k + mm1 * self.m2;
                        let mut j1 = k;
                        let mut lbeg = 0.max(j1 - self.mujac) + self.m1;
                        let mut lend;
                        let mut mujacj;
                        loop {
                            lend = self.m2.min(j1 + self.mljac) + self.m1;
                            integrator.u[j] = self.f1[j];
                            mujacj = mujacp - j1 - self.m1 - 1;
                            for l in lbeg..(lend + 1) {
                                self.dfdu[[l + mujacj, j]] =
                                    (self.cont[l] - self.u0[l]) / self.f2[j];
                            }
                            j += md;
                            j1 += md;
                            lbeg = lend + 1;
                            if j > (mm1 + 1) * self.m2 - 1 {
                                break;
                            }
                        }
                    }
                }
            } else {
                // Jacobian is full
                let mut delt;
                let mut ysafe;
                for i in 0..n {
                    ysafe = integrator.u[i];
                    delt = (f64::EPSILON * 1.0e-5f64.max(ysafe.abs())).sqrt();
                    integrator.u[i] = ysafe + delt;
                    (integrator.dudt)(self.cont.view_mut(), integrator.u.view(), integrator.t);
                    for j in self.m1..n {
                        self.dfdu[[j - self.m1, i]] = (self.cont[j] - self.u0[j]) / delt;
                    }
                    integrator.u[i] = ysafe;
                }
            }
        }
        self.calhes = true;
        self.caljac = true;
    }
}
