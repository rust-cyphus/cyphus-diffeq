use super::alg::Radau5;
use crate::ode_prob::OdeProblem;
use ndarray::prelude::*;

pub(crate) struct Radau5Cache {
    pub(crate) prob_type: usize,
    pub(crate) implct: bool,
    pub(crate) jband: bool,
    pub(crate) caljac: bool,
    pub(crate) calhes: bool,
    pub(crate) first: bool,
    pub(crate) reject: bool,
    pub(crate) mujac: usize,
    pub(crate) mljac: usize,
    pub(crate) mumas: usize,
    pub(crate) mlmas: usize,
    pub(crate) m1: usize,
    pub(crate) m2: usize,
    pub(crate) nind1: usize,
    pub(crate) nind2: usize,
    pub(crate) nind3: usize,
    pub(crate) mle: usize,
    pub(crate) mue: usize,
    pub(crate) mbjac: usize,
    pub(crate) mbb: usize,
    pub(crate) mdiag: usize,
    pub(crate) mdiff: usize,
    pub(crate) mbdiag: usize,
    pub(crate) ldmas: usize,
    pub(crate) ldjac: usize,
    pub(crate) lde1: usize,
    pub(crate) fac1: f64,
    pub(crate) alphn: f64,
    pub(crate) betan: f64,
    pub(crate) fnewt: f64,
    pub(crate) err: f64,
    pub(crate) u0: Array1<f64>,
    pub(crate) scal: Array1<f64>,
    pub(crate) cont: Array1<f64>,
    pub(crate) z1: Array1<f64>,
    pub(crate) z2: Array1<f64>,
    pub(crate) z3: Array1<f64>,
    pub(crate) f1: Array1<f64>,
    pub(crate) f2: Array1<f64>,
    pub(crate) f3: Array1<f64>,
    pub(crate) ip1: Array1<i32>,
    pub(crate) ip2: Array1<i32>,
    pub(crate) iphes: Array1<i32>,
    pub(crate) e1: Array2<f64>,
    pub(crate) e2r: Array2<f64>,
    pub(crate) e2i: Array2<f64>,
    pub(crate) dfdu: Array2<f64>,
    pub(crate) mass_matrix: Array2<f64>,
}

pub(crate) enum Radau5CacheErr {
    HessErr,
}

impl std::fmt::Debug for Radau5CacheErr {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            Radau5CacheErr::HessErr => write!(
                f,
                "Hessenberg option only for explicit equations with full Jacobian.",
            ),
        }
    }
}

impl Radau5Cache {
    pub(crate) fn new<F, J>(
        alg: &Radau5,
        prob: &OdeProblem<F, J>,
    ) -> Result<Radau5Cache, Radau5CacheErr>
    where
        F: Fn(ArrayViewMut1<f64>, ArrayView1<f64>, f64),
        J: Fn(ArrayViewMut2<f64>, ArrayView1<f64>, f64),
    {
        let n = prob.uinit.shape()[0];
        let nm1 = n - prob.m1;
        let implct = prob.mass_matrix.is_none();
        let jband = prob.jac_lbw < nm1;

        let mut prob_type = 1;

        let (mljac, mujac, ldjac, lde1) = if jband {
            (
                prob.jac_lbw,
                prob.jac_ubw,
                prob.jac_lbw + prob.jac_ubw + 1,
                2 * prob.jac_lbw + prob.jac_ubw + 1,
            )
        } else {
            (nm1, nm1, nm1, nm1)
        };
        let mut mlmas = prob.mm_lbw;
        let mut mumas = prob.mm_ubw;
        let mut ldmas = 1;
        if implct {
            if prob.mm_lbw != nm1 {
                ldmas = mlmas + mumas + 1;
                prob_type = if jband { 4 } else { 3 };
            } else {
                mumas = nm1;
                mlmas = nm1;
                prob_type = 5;
            }
        } else {
            ldmas = 0;
            if jband {
                prob_type = 2;
            } else {
                prob_type = 1;
                if n > 2 && alg.hess {
                    prob_type = 7;
                }
            }
        }
        ldmas = ldmas.max(1);

        if (implct || jband) && prob_type == 7 {
            return Err(Radau5CacheErr::HessErr);
        }

        if prob.m1 > 0 {
            prob_type += 10;
        }

        let mle = mljac;
        let mue = mujac;
        let mbjac = mljac + mujac + 1;
        let mbb = mlmas + mumas + 1;
        let mdiag = mle + mue;
        let mdiff = mle + mue - mumas;
        let mbdiag = mumas + 1;

        Ok(Radau5Cache {
            prob_type,
            implct,
            jband,
            caljac: false,
            calhes: false,
            first: true,
            reject: false,
            mujac,
            mljac,
            mumas,
            mlmas,
            m1: prob.m1,
            m2: prob.m2,
            nind1: prob.num_index1_vars,
            nind2: prob.num_index2_vars,
            nind3: prob.num_index3_vars,
            mle,
            mue,
            mbjac,
            mbb,
            mdiag,
            mdiff,
            mbdiag,
            ldmas,
            ldjac,
            lde1,
            fac1: 0.0,
            alphn: 0.0,
            betan: 0.0,
            fnewt: alg.fnewt,
            err: 0.0,
            u0: Array1::<f64>::zeros(n),
            scal: Array1::<f64>::zeros(n),
            /// Coninuous output vectors
            cont: Array1::<f64>::zeros(4 * n),
            z1: Array1::<f64>::zeros(n),
            z2: Array1::<f64>::zeros(n),
            z3: Array1::<f64>::zeros(n),
            f1: Array1::<f64>::zeros(n),
            f2: Array1::<f64>::zeros(n),
            f3: Array1::<f64>::zeros(n),
            ip1: Array1::<i32>::zeros(nm1),
            ip2: Array1::<i32>::zeros(nm1),
            iphes: Array1::<i32>::zeros(nm1),
            e1: Array2::<f64>::zeros((lde1, nm1)),
            e2r: Array2::<f64>::zeros((lde1, nm1)),
            e2i: Array2::<f64>::zeros((lde1, nm1)),
            dfdu: Array2::<f64>::zeros((ldjac, n)),
            mass_matrix: Array2::<f64>::zeros((ldmas, nm1)),
        })
    }
}
