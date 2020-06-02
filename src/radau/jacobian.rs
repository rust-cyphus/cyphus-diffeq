use super::alg::Radau5;
use crate::ode_integrator::OdeIntegrator;
use crate::ode_prob::OdeProblem;
use crate::radau::cache::Radau5Cache;
use ndarray::prelude::*;

impl Radau5 {
    pub(crate) fn compute_jacobian<F, J>(
        &self,
        integrator: &mut OdeIntegrator,
        prob: &OdeProblem<F, J>,
        cache: &mut Radau5Cache,
    ) where
        F: Fn(ArrayViewMut1<f64>, ArrayView1<f64>, f64),
        J: Fn(ArrayViewMut2<f64>, ArrayView1<f64>, f64),
    {
        integrator.stats.jacobian_evals += 1;
        match &prob.dfdu {
            Some(dfdu) => {
                dfdu(cache.dfdu.view_mut(), integrator.u.view(), integrator.t);
            }
            None => {
                let n = cache.e1.nrows();
                if cache.jband {
                    // Jacobian is banded
                    let mujacp = cache.mujac + 1;
                    let md = cache.mbjac.min(cache.m2);
                    for mm1 in 0..(cache.m1 / cache.m2 as usize + 1) {
                        for k in 0..md {
                            let mut j = k + mm1 * cache.m2;
                            loop {
                                cache.f1[j] = integrator.u[j];
                                cache.f2[j] =
                                    (f64::EPSILON * 1.0e-5f64.max(integrator.u[j].abs())).sqrt();
                                integrator.u[j] += cache.f2[j];
                                j += md;
                                if j > (mm1 + 1) * cache.m2 - 1 {
                                    break;
                                }
                            }
                            (prob.dudt)(cache.cont.view_mut(), integrator.u.view(), integrator.t);
                            j = k + mm1 * cache.m2;
                            let mut j1 = k;
                            let mut lbeg = 0.max(j1 - cache.mujac) + cache.m1;
                            let mut lend;
                            let mut mujacj;
                            loop {
                                lend = cache.m2.min(j1 + cache.mljac) + cache.m1;
                                integrator.u[j] = cache.f1[j];
                                mujacj = mujacp - j1 - cache.m1 - 1;
                                for l in lbeg..(lend + 1) {
                                    cache.dfdu[[l + mujacj, j]] =
                                        (cache.cont[l] - cache.u0[l]) / cache.f2[j];
                                }
                                j += md;
                                j1 += md;
                                lbeg = lend + 1;
                                if j > (mm1 + 1) * cache.m2 - 1 {
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
                        (prob.dudt)(cache.cont.view_mut(), integrator.u.view(), integrator.t);
                        for j in cache.m1..n {
                            cache.dfdu[[j - cache.m1, i]] = (cache.cont[j] - cache.u0[j]) / delt;
                        }
                        integrator.u[i] = ysafe;
                    }
                }
            }
        }
        cache.calhes = true;
        cache.caljac = true;
    }
}
