use super::alg::Radau5;
use super::cache::Radau5Cache;
use crate::ode_integrator::OdeIntegrator;
use crate::ode_prob::OdeProblem;
use crate::ode_solution::OdeSolution;
use crate::ode_stats::OdeStats;
use ndarray::prelude::*;

impl Radau5 {
    #[allow(dead_code)]
    pub(crate) fn integrate<F, J>(&self, prob: &OdeProblem<F, J>) -> OdeSolution
    where
        F: Fn(ArrayViewMut1<f64>, ArrayView1<f64>, f64),
        J: Fn(ArrayViewMut2<f64>, ArrayView1<f64>, f64),
    {
        let mut alg = (*self).clone();
        alg.dtmax = prob.tspan.1 - prob.tspan.0;

        let mut integrator = OdeIntegrator {
            u: prob.uinit.clone(),
            t: prob.tspan.0,
            dt: alg.dtstart,
            uprev: prob.uinit.clone(),
            tprev: prob.tspan.1,
            dtprev: alg.dtstart,
            stats: OdeStats::new(),
        };

        let mut solution = OdeSolution {
            ts: Vec::<f64>::with_capacity(200),
            us: Vec::<Array1<f64>>::with_capacity(200),
            stats: OdeStats::new(),
        };
        solution.ts.push(prob.tspan.0);
        solution.us.push(prob.uinit.clone());

        let mut cache = Radau5Cache::new(&alg, &prob).unwrap();

        let tfinal = prob.tspan.1;

        let sq6 = 6f64.sqrt();
        let c1 = (4.0 - sq6) / 10.0;
        let c2 = (4.0 + sq6) / 10.0;
        let c1m1 = c1 - 1.0;
        let c2m1 = c2 - 1.0;
        let c1mc2 = c1 - c2;
        let u1 = 1.0 / ((6.0 + 81f64.powf(1.0 / 3.0) - 9f64.powf(1.0 / 3.0)) / 30.0);
        let mut alph = (12.0 - 81f64.powf(1.0 / 3.0) + 9f64.powf(1.0 / 3.0)) / 60.0;
        let mut beta = (81f64.powf(1.0 / 3.0) + 9f64.powf(1.0 / 3.0)) * 3f64.sqrt() / 60.0;
        let cno = alph * alph + beta * beta;

        {
            let quot = alg.abstol / alg.reltol;
            alg.reltol = 0.1 * alg.reltol.powf(2.0 / 3.0);
            alg.abstol = alg.reltol * quot;
        }

        let n = integrator.u.shape()[0];

        cache.fnewt = if alg.fnewt == 0.0 {
            (10.0 * f64::EPSILON / alg.reltol).max(0.03f64.min(alg.reltol.sqrt()))
        } else {
            alg.fnewt
        };

        alph = alph / cno;
        beta = beta / cno;

        let posneg = 1.0f64.copysign(tfinal - integrator.t);
        let dtmax = (alg.dtmax).abs().min((tfinal - integrator.t).abs());
        let cfac = alg.safe * (1 + 2 * alg.max_newt) as f64;

        // compute mass matrix for implicit case
        cache.mass_matrix = match &prob.mass_matrix {
            Some(mm) => mm.clone(),
            None => Array2::eye(n),
        };

        integrator.dt = (integrator.dt).abs().min(dtmax);
        integrator.dt = integrator.dt.copysign(posneg);
        integrator.dtprev = integrator.dt;

        let mut last = false;
        let mut first = true;

        if integrator.t + integrator.dt * 1.0001 - tfinal * posneg >= 0.0 {
            integrator.dt = tfinal - integrator.t;
            last = true;
        }

        let mut dtopt: f64 = integrator.dt;
        let mut faccon: f64 = 1.0;

        if alg.dense {
            let irtrn = 1;
            for i in (0)..(n) {
                cache.cont[i] = integrator.u[i];
            }
            // irtrn = SolutionOutput();
            if irtrn < 0 {
                println!("exit of RADAU5 at t = {}", integrator.t);
                return solution;
            }
        }
        solution.ts.push(integrator.t);
        solution.us.push(integrator.u.clone());

        for i in (0)..(n) {
            cache.scal[i] = alg.abstol + alg.reltol * integrator.u[i].abs();
        }

        (prob.dudt)(cache.u0.view_mut(), integrator.u.view(), integrator.t);
        integrator.stats.function_evals += 1;

        let mut dtfac = integrator.dt;
        let mut dtacc: f64 = 0.0;
        let mut erracc: f64 = 0.0;
        let mut thqold: f64 = 0.0;
        let mut nsing = 0;
        let mut ier;

        // basic integration step
        alg.compute_jacobian(&mut integrator, prob, &mut cache);
        let mut lloop = true;
        while lloop {
            lloop = false;
            // compute the matrices e1 and e2 and their decompositions
            cache.fac1 = u1 / integrator.dt;
            cache.alphn = alph / integrator.dt;
            cache.betan = beta / integrator.dt;

            ier = alg.decomp_real(&mut cache);

            if ier != 0 {
                nsing += 1;
                if nsing >= 5 {
                    println!("exit of RADAU5 at t = {}", integrator.t);
                    println!("Matrix is repeatedly singular.");
                    return solution;
                }
                integrator.dt *= 0.5;
                dtfac = 0.5;
                cache.reject = true;
                last = false;
                if !cache.caljac {
                    alg.compute_jacobian(&mut integrator, prob, &mut cache);
                }
                lloop = true;
                continue;
            }

            ier = alg.decomp_complex(&mut cache);

            if ier != 0 {
                nsing += 1;
                if nsing >= 5 {
                    println!("exit of RADAU5 at t = {}", integrator.t);
                    println!("Matrix is repeatedly singular.");
                    return solution;
                }
                integrator.dt *= 0.5;
                dtfac = 0.5;
                cache.reject = true;
                last = false;
                if !cache.caljac {
                    alg.compute_jacobian(&mut integrator, prob, &mut cache);
                }
                lloop = true;
                continue;
            }
            integrator.stats.decompositions += 1;

            loop {
                integrator.stats.steps += 1;
                if integrator.stats.steps >= alg.max_steps {
                    println!("exit of RADAU5 at t = {}", integrator.t);
                    println!("More than {} iterations needed.", alg.max_steps);
                    return solution;
                }

                if 0.1 * (integrator.dt).abs() <= (integrator.t).abs() * f64::EPSILON {
                    println!("exit of RADAU5 at t = {}", integrator.t);
                    println!("Step size too small: dt = {}", integrator.dt);
                    return solution;
                }

                // check the index of the problem
                if cache.nind2 != 0 {
                    // is index 2
                    for i in (cache.nind1)..(cache.nind1 + cache.nind2) {
                        cache.scal[i] = cache.scal[i] / dtfac;
                    }
                }

                if cache.nind3 != 0 {
                    // is index 3
                    for i in (cache.nind1 + cache.nind2)..(cache.nind1 + cache.nind2 + cache.nind3)
                    {
                        cache.scal[i] = cache.scal[i] / (dtfac * dtfac);
                    }
                }

                let tph = integrator.t + integrator.dt;
                //  starting values for Newton iteration
                if first || !alg.use_ext_col {
                    for i in (0)..(n) {
                        cache.z1[i] = 0.0;
                        cache.z2[i] = 0.0;
                        cache.z3[i] = 0.0;
                        cache.f1[i] = 0.0;
                        cache.f2[i] = 0.0;
                        cache.f3[i] = 0.0;
                    }
                } else {
                    let c3q = integrator.dt / integrator.dtprev;
                    let c1q = c1 * c3q;
                    let c2q = c2 * c3q;
                    let mut ak1;
                    let mut ak2;
                    let mut ak3;
                    for i in (0)..(n) {
                        ak1 = cache.cont[i + n];
                        ak2 = cache.cont[i + 2 * n];
                        ak3 = cache.cont[i + 3 * n];
                        cache.z1[i] = c1q * (ak1 + (c1q - c2m1) * (ak2 + (c1q - c1m1) * ak3));
                        cache.z2[i] = c2q * (ak1 + (c2q - c2m1) * (ak2 + (c2q - c1m1) * ak3));
                        cache.z3[i] = c3q * (ak1 + (c3q - c2m1) * (ak2 + (c3q - c1m1) * ak3));
                        cache.f1[i] = Radau5::TI11 * cache.z1[i]
                            + Radau5::TI12 * cache.z2[i]
                            + Radau5::TI13 * cache.z3[i];
                        cache.f2[i] = Radau5::TI21 * cache.z1[i]
                            + Radau5::TI22 * cache.z2[i]
                            + Radau5::TI23 * cache.z3[i];
                        cache.f3[i] = Radau5::TI31 * cache.z1[i]
                            + Radau5::TI32 * cache.z2[i]
                            + Radau5::TI33 * cache.z3[i];
                    }
                }

                //  lloop for the simplified Newton iteration
                let mut newt = 0;
                faccon = faccon.max(f64::EPSILON).powf(0.8);
                let mut theta = alg.thet.abs();
                let mut dyno: f64;
                let mut dynold: f64 = 0.0;

                loop {
                    if newt >= alg.max_newt {
                        if ier != 0 {
                            nsing += 1;
                            if nsing >= 5 {
                                println!("exit of RADAU5 at t = {}", integrator.t);
                                println!("matrix is repeatedly singular");
                                return solution;
                            }
                        }
                        integrator.dt *= 0.5;
                        dtfac = 0.5;
                        cache.reject = true;
                        last = false;
                        if !cache.caljac {
                            alg.compute_jacobian(&mut integrator, prob, &mut cache);
                        }
                        lloop = true;
                        break;
                    }
                    // compute the right-hand side
                    for i in (0)..(n) {
                        cache.cont[i] = integrator.u[i] + cache.z1[i];
                    }
                    (prob.dudt)(
                        cache.z1.view_mut(),
                        cache.cont.view(),
                        integrator.t + c1 * integrator.dt,
                    );

                    for i in (0)..(n) {
                        cache.cont[i] = integrator.u[i] + cache.z2[i];
                    }
                    (prob.dudt)(
                        cache.z2.view_mut(),
                        cache.cont.view(),
                        integrator.t + c2 * integrator.dt,
                    );

                    for i in (0)..(n) {
                        cache.cont[i] = integrator.u[i] + cache.z3[i];
                    }
                    (prob.dudt)(cache.z3.view_mut(), cache.cont.view(), tph);

                    integrator.stats.function_evals += 3;

                    // solve the linear systems
                    for i in (0)..(n) {
                        let a1 = cache.z1[i];
                        let a2 = cache.z2[i];
                        let a3 = cache.z3[i];
                        cache.z1[i] = Radau5::TI11 * a1 + Radau5::TI12 * a2 + Radau5::TI13 * a3;
                        cache.z2[i] = Radau5::TI21 * a1 + Radau5::TI22 * a2 + Radau5::TI23 * a3;
                        cache.z3[i] = Radau5::TI31 * a1 + Radau5::TI32 * a2 + Radau5::TI33 * a3;
                    }
                    alg.linear_solve(&mut cache);
                    integrator.stats.linear_solves += 1;
                    newt += 1;
                    dyno = 0.0;
                    let mut denom: f64;
                    for i in (0)..(n) {
                        denom = cache.scal[i];
                        dyno = dyno
                            + (cache.z1[i] / denom).powi(2)
                            + (cache.z2[i] / denom).powi(2)
                            + (cache.z3[i] / denom).powi(2);
                    }
                    dyno = (dyno / ((3 * n) as f64)).sqrt();
                    // bad convergence or number of iterations to large
                    if newt > 1 && (newt < alg.max_newt) {
                        let thq = dyno / dynold;
                        if newt == 2 {
                            theta = thq;
                        } else {
                            theta = (thq * thqold).sqrt();
                        }
                        thqold = thq;
                        if theta < 0.99 {
                            faccon = theta / (1.0 - theta);
                            let dyth = faccon * dyno * theta.powi((alg.max_newt - 1 - newt) as i32)
                                / cache.fnewt;
                            if dyth >= 1.0 {
                                let qnewt: f64 = (1.0e-4f64).max((20.0f64).min(dyth));
                                dtfac = 0.8
                                    * qnewt.powf(-1.0f64 / (4 + alg.max_newt - 1 - newt) as f64);
                                integrator.dt *= dtfac;
                                cache.reject = true;
                                last = false;
                                if cache.caljac {
                                    alg.compute_jacobian(&mut integrator, prob, &mut cache);
                                }
                                lloop = true;
                                break;
                            }
                        } else {
                            if ier != 0 {
                                nsing += 1;
                                if nsing >= 5 {
                                    println!("exit of RADAU5 at t = {}", integrator.t);
                                    println!("matrix is repeatedly singular");
                                    return solution;
                                }
                            }
                            integrator.dt *= 0.5;
                            dtfac = 0.5;
                            cache.reject = true;
                            last = false;
                            if !cache.caljac {
                                alg.compute_jacobian(&mut integrator, prob, &mut cache);
                            }
                            lloop = true;
                            break;
                        }
                    }
                    dynold = (dyno).max(f64::EPSILON);
                    for i in (0)..(n) {
                        cache.f1[i] = cache.f1[i] + cache.z1[i];
                        cache.f2[i] = cache.f2[i] + cache.z2[i];
                        cache.f3[i] = cache.f3[i] + cache.z3[i];
                        cache.z1[i] = Radau5::T11 * cache.f1[i]
                            + Radau5::T12 * cache.f2[i]
                            + Radau5::T13 * cache.f3[i];
                        cache.z2[i] = Radau5::T21 * cache.f1[i]
                            + Radau5::T22 * cache.f2[i]
                            + Radau5::T23 * cache.f3[i];
                        cache.z3[i] = Radau5::T31 * cache.f1[i] + cache.f2[i];
                    }
                    if faccon * dyno <= cache.fnewt {
                        break;
                    }
                }

                if lloop {
                    break;
                }

                // error estimation
                cache.err = 0.0;
                ier = alg.error_estimate(&mut integrator, prob, &mut cache);

                // computation of hnew -- require 0.2 <= hnew/integrator.dt <= 8.
                let fac = (alg.safe).min(cfac / (newt + 2 * alg.max_newt) as f64);
                let mut quot = (alg.facr).max((alg.facl).min(cache.err.powf(0.25) / fac));
                let mut hnew = integrator.dt / quot;

                //  is the error small enough ?
                if cache.err < 1.0 {
                    // step is accepted
                    first = false;
                    integrator.stats.accepts += 1;
                    if alg.modern_pred {
                        // predictive controller of Gustafsson
                        if integrator.stats.accepts > 1 {
                            let mut facgus = (dtacc / (integrator.dt))
                                * (cache.err * cache.err / erracc).powf(0.25)
                                / alg.safe;
                            facgus = (alg.facr).max((alg.facl).min(facgus));
                            quot = (quot).max(facgus);
                            hnew = integrator.dt / quot;
                        }
                        dtacc = integrator.dt;
                        erracc = (1.0e-2f64).max(cache.err);
                    }
                    integrator.tprev = integrator.t;
                    integrator.dtprev = integrator.dt;
                    integrator.t = tph;
                    let mut ak: f64;
                    let mut acont3: f64;
                    for i in (0)..(n) {
                        integrator.u[i] = integrator.u[i] + cache.z3[i];
                        cache.cont[i + n] = (cache.z2[i] - cache.z3[i]) / c2m1;
                        ak = (cache.z1[i] - cache.z2[i]) / c1mc2;
                        acont3 = cache.z1[i] / c1;
                        acont3 = (ak - acont3) / c2;
                        cache.cont[i + 2 * n] = (ak - cache.cont[i + n]) / c1m1;
                        cache.cont[i + 3 * n] = cache.cont[i + 2 * n] - acont3;
                    }
                    for i in (0)..(n) {
                        cache.scal[i] = alg.abstol + alg.reltol * integrator.u[i].abs();
                    }
                    if alg.dense {
                        for i in (0)..(n) {
                            cache.cont[i] = integrator.u[i];
                        }
                        let irtrn = 1;
                        // irtrn = SolutionOutput();
                        if irtrn < 0 {
                            println!("exit of RADAU5 at t = {}", integrator.t);
                            return solution;
                        }
                    }
                    cache.caljac = false;
                    if last {
                        integrator.dt = dtopt;
                        return solution;
                    }
                    (prob.dudt)(cache.u0.view_mut(), integrator.u.view(), integrator.t);
                    integrator.stats.function_evals += 1;
                    hnew = posneg * hnew.abs().min(dtmax);
                    dtopt = integrator.dt.min(hnew);
                    if cache.reject {
                        hnew = posneg * (hnew).abs().min(integrator.dt.abs());
                    }
                    cache.reject = false;
                    if integrator.t + hnew / alg.quot1 - tfinal * posneg >= 0.0 {
                        integrator.dt = tfinal - integrator.t;
                        last = true;
                    } else {
                        let qt = hnew / (integrator.dt);
                        dtfac = integrator.dt;
                        if theta <= alg.thet && (qt >= alg.quot1 && (qt <= alg.quot2)) {
                            continue;
                        }
                        integrator.dt = hnew;
                    }
                    dtfac = integrator.dt;
                    if theta > alg.thet {
                        alg.compute_jacobian(&mut integrator, prob, &mut cache);
                    }
                    lloop = true;
                } else {
                    // step is rejected
                    cache.reject = true;
                    last = false;
                    if first {
                        integrator.dt *= 0.1;
                        dtfac = 0.1;
                    } else {
                        dtfac = hnew / (integrator.dt);
                        integrator.dt = hnew;
                    }
                    if integrator.stats.accepts >= 1 {
                        integrator.stats.rejects += 1;
                    }
                    if !cache.caljac {
                        alg.compute_jacobian(&mut integrator, prob, &mut cache);
                    }
                    lloop = true;
                }
                break;
            }
        }
        solution
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::ode_algorithm::OdeAlgorithm;
    use crate::ode_prob::OdeProblemBuilder;
    use crate::radau::alg::Radau5;
    use ndarray::prelude::*;

    #[test]
    fn test_van_der_pol() {
        let uinit = array![2.0, -0.66];
        let tspan = (0.0, 2.0);
        let mu = 1e-6;
        let dudt = |mut du: ArrayViewMut1<f64>, u: ArrayView1<f64>, t: f64| {
            du[0] = u[1];
            du[1] = ((1.0 - u[0] * u[0]) * u[1] - u[0]) / mu;
        };
        let dfdu = |mut df: ArrayViewMut2<f64>, u: ArrayView1<f64>, t: f64| {
            df[[0, 0]] = 0.0;
            df[[0, 1]] = 1.0;
            df[[1, 0]] = -(2.0 * u[0] * u[1] + 1.0) / mu;
            df[[1, 1]] = (1.0 - u[0] * u[0]) / mu;
        };
        let prob = OdeProblemBuilder::default(dudt, uinit.clone(), tspan)
            .with_jac(dfdu)
            .build()
            .unwrap();
        let mut rad = Radau5::default();
        rad.reltol(1e-7);
        rad.abstol(1e-7);
        // rad.integrate(
        let sol = rad.integrate(&prob);
        for (t, u) in sol.us.iter().zip(sol.ts.iter()) {
            println!("t, u = {:?}, {:?}", t, &u)
        }
    }
}
