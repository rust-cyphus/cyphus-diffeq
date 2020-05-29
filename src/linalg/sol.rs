//! Collection of algorithms for solving linear systems of equations for various
//! types of matrices.

use ndarray::prelude::*;

pub(crate) fn sol(n: usize, a: ArrayView2<f64>, mut b: ArrayViewMut1<f64>, ip: ArrayView1<i32>) {
    if n != 1 {
        let nm1 = n - 1;
        for k in 0..nm1 {
            let kp1 = k + 1;
            let m = ip[k];
            let t = b[m as usize];
            b[m as usize] = b[k];
            b[k] = t;
            for i in kp1..n {
                b[i] += a[[i, k]] * t;
            }
        }
        for k in 0..nm1 {
            let km1 = n - k - 2;
            let kb = km1 + 1;
            b[kb] = b[kb] / a[[kb, kb]];
            let t = -b[kb];
            for i in 0..(km1 + 1) {
                b[i] += a[[i, kb]] * t;
            }
        }
    }
    b[0] = b[0] / a[[0, 0]];
}

pub(crate) fn solh(
    n: usize,
    a: ArrayView2<f64>,
    lb: usize,
    mut b: ArrayViewMut1<f64>,
    ip: ArrayView1<i32>,
) {
    if n != 1 {
        let nm1 = n - 1;
        for k in 0..nm1 {
            let kp1 = k + 1;
            let m = ip[k] as usize;
            let t = b[m];
            b[m] = b[k];
            b[k] = t;
            let na = n.min(lb + k + 1);
            for i in kp1..na {
                b[i] += a[[i, k]] * t;
            }
        }
        for k in 0..nm1 {
            let km1 = n - k - 2;
            let kb = km1 + 1;
            b[kb] = b[kb] / a[[kb, kb]];
            let t = -b[kb];
            for i in 0..(km1 + 1) {
                b[i] += a[[i, kb]] * t;
            }
        }
    }
    b[0] = b[0] / a[[0, 0]];
}

pub(crate) fn solc(
    n: usize,
    ar: ArrayView2<f64>,
    ai: ArrayView2<f64>,
    mut br: ArrayViewMut1<f64>,
    mut bi: ArrayViewMut1<f64>,
    ip: ArrayView1<i32>,
) {
    if n != 1 {
        let nm1 = n - 1;
        for k in 0..nm1 {
            let kp1 = k + 1;
            let m = ip[k] as usize;
            let tr = br[m];
            let ti = bi[m];
            br[m] = br[k];
            bi[m] = bi[k];
            br[k] = tr;
            bi[k] = ti;
            for i in kp1..n {
                let prodr = ar[[i, k]] * tr - ai[[i, k]] * ti;
                let prodi = ai[[i, k]] * tr + ar[[i, k]] * ti;
                br[i] += prodr;
                bi[i] += prodi;
            }
        }
        for k in 0..nm1 {
            let km1 = n - k - 2;
            let kb = km1 + 1;
            let den = ar[[kb, kb]] * ar[[kb, kb]] + ai[[kb, kb]] * ai[[kb, kb]];
            let mut prodr = br[kb] * ar[[kb, kb]] + bi[kb] * ai[[kb, kb]];
            let mut prodi = bi[kb] * ar[[kb, kb]] - br[kb] * ai[[kb, kb]];
            br[kb] = prodr / den;
            bi[kb] = prodi / den;
            let tr = -br[kb];
            let ti = -bi[kb];
            for i in 0..(km1 + 1) {
                prodr = ar[[i, kb]] * tr - ai[[i, kb]] * ti;
                prodi = ai[[i, kb]] * tr + ar[[i, kb]] * ti;
                br[i] += prodr;
                bi[i] += prodi;
            }
        }
    }
    let den = ar[[0, 0]] * ar[[0, 0]] + ai[[0, 0]] * ai[[0, 0]];
    let prodr = br[0] * ar[[0, 0]] + bi[0] * ai[[0, 0]];
    let prodi = bi[0] * ar[[0, 0]] - br[0] * ai[[0, 0]];
    br[0] = prodr / den;
    bi[0] = prodi / den;
}

pub(crate) fn solhc(
    n: usize,
    ar: ArrayView2<f64>,
    ai: ArrayView2<f64>,
    lb: usize,
    mut br: ArrayViewMut1<f64>,
    mut bi: ArrayViewMut1<f64>,
    ip: ArrayView1<i32>,
) {
    if n != 1 {
        let nm1 = n - 1;
        if lb != 0 {
            for k in 0..nm1 {
                let kp1 = k + 1;
                let m = ip[k] as usize;
                let tr = br[m];
                let ti = bi[m];
                br[m] = br[k];
                bi[m] = bi[k];
                br[k] = tr;
                bi[k] = ti;
                for i in kp1..n.min(lb + k + 1) {
                    let prodr = ar[[i, k]] * tr - ai[[i, k]] * ti;
                    let prodi = ai[[i, k]] * tr + ar[[i, k]] * ti;
                    br[i] += prodr;
                    bi[i] += prodi;
                }
            }
        }
        for k in 0..nm1 {
            let km1 = n - k - 2;
            let kb = km1 + 1;
            let den = ar[[kb, kb]] * ar[[kb, kb]] + ai[[kb, kb]] * ai[[kb, kb]];
            let mut prodr = br[kb] * ar[[kb, kb]] + bi[kb] * ai[[kb, kb]];
            let mut prodi = bi[kb] * ar[[kb, kb]] - br[kb] * ai[[kb, kb]];
            br[kb] = prodr / den;
            bi[kb] = prodi / den;
            let tr = -br[kb];
            let ti = -bi[kb];
            for i in 0..km1 {
                prodr = ar[[i, kb]] * tr - ai[[i, kb]] * ti;
                prodi = ai[[i, kb]] * tr + ar[[i, kb]] * ti;
                br[i] += prodr;
                bi[i] += prodi;
            }
        }
    }
    let den = ar[[0, 0]] * ar[[0, 0]] + ai[[0, 0]] * ai[[0, 0]];
    let prodr = br[0] * ar[[0, 0]] + bi[0] * ai[[0, 0]];
    let prodi = bi[0] * ar[[0, 0]] - br[0] * ai[[0, 0]];
    br[0] = prodr / den;
    bi[0] = prodi / den;
}
pub(crate) fn solb(
    n: usize,
    a: ArrayView2<f64>,
    lb: usize,
    ub: usize,
    mut b: ArrayViewMut1<f64>,
    ip: ArrayView1<i32>,
) {
    let md = lb + ub;
    let md1 = md + 1;
    let mdm = md - 1;
    let nm1 = n - 1;
    if n != 1 {
        if lb != 0 {
            for k in 0..nm1 {
                let m = ip[k] as usize;
                let t = b[m];
                b[m] = b[k];
                b[k] = t;
                let mdl = lb.min(n - k - 1) + md;
                for i in md1..(mdl + 1) {
                    let imd = i + k - md;
                    b[imd] += a[[i, k]] * t;
                }
            }
        }
        for k in 0..nm1 {
            let kb = n - k - 1;
            b[kb] = b[kb] / a[[md, kb]];
            let t = -b[kb];
            let kmd = md - kb;
            let lm = 0.max(kmd);
            for i in lm..(mdm + 1) {
                let imd = i - kmd;
                b[imd] += a[[i, kb]] * t;
            }
        }
    }
    b[0] = b[0] / a[[md, 0]];
}

pub(crate) fn solbc(
    n: usize,
    ar: ArrayView2<f64>,
    ai: ArrayView2<f64>,
    lb: usize,
    ub: usize,
    mut br: ArrayViewMut1<f64>,
    mut bi: ArrayViewMut1<f64>,
    ip: ArrayView1<i32>,
) {
    let md = lb + ub;
    let md1 = md + 1;
    let mdm = md - 1;
    let nm1 = n - 1;
    if n != 1 {
        if lb != 0 {
            for k in 0..nm1 {
                let m = ip[k] as usize;
                let tr = br[m];
                let ti = bi[m];
                br[m] = br[k];
                bi[m] = bi[k];
                br[k] = tr;
                bi[k] = ti;
                let mdl = lb.min(n - k - 1) + md;
                for i in md1..(mdl + 1) {
                    let imd = i + k - md;
                    let prodr = ar[[i, k]] * tr - ai[[i, k]] * ti;
                    let prodi = ai[[i, k]] * tr + ar[[i, k]] * ti;
                    br[imd] += prodr;
                    bi[imd] += prodi;
                }
            }
        }
        for k in 0..nm1 {
            let kb = n - k - 1;
            let den = ar[[md, kb]] * ar[[md, kb]] + ai[[md, kb]] * ai[[md, kb]];
            let mut prodr = br[kb] * ar[[md, kb]] + bi[kb] * ai[[md, kb]];
            let mut prodi = bi[kb] * ar[[md, kb]] - br[kb] * ai[[md, kb]];
            br[kb] = prodr / den;
            bi[kb] = prodi / den;
            let tr = -br[kb];
            let ti = -bi[kb];
            let kmd = md - kb;
            let lm = 0.max(kmd);
            for i in lm..(mdm + 1) {
                let imd = i - kmd;
                prodr = ar[[i, kb]] * tr - ai[[i, kb]] * ti;
                prodi = ai[[i, kb]] * tr + ar[[i, kb]] * ti;
                br[imd] += prodr;
                bi[imd] += prodi;
            }
        }
        let den = ar[[md, 0]] * ar[[md, 0]] + ai[[md, 0]] * ai[[md, 0]];
        let prodr = br[0] * ar[[md, 0]] + bi[0] * ai[[md, 0]];
        let prodi = bi[0] * ar[[md, 0]] - br[0] * ai[[md, 0]];
        br[0] = prodr / den;
        bi[0] = prodi / den;
    }
}

#[cfg(test)]
mod test {
    // TODO: Write tests for other solvers
    use super::*;
    use crate::linalg::dec::*;

    #[test]
    fn test_sol() {
        let mut a = array![
            [1.0, 2.0, 3.0],
            [2f64.sqrt(), 3f64.sqrt(), 4f64.sqrt()],
            [1.0, 4.0, 9.0]
        ];
        let mut ip: Array1<i32> = Array::zeros(3);
        let mut b = array![1.0, 2.0, 3.0];
        let ier = dec(3, a.view_mut(), ip.view_mut());
        sol(3, a.view(), b.view_mut(), ip.view());

        assert!((b[0] - 3.822307555684841).abs() < 1e-15);
        assert!((b[1] + 3.822307555684841).abs() < 1e-15);
        assert!((b[2] - 1.607435851894947).abs() < 1e-15);
    }

    #[test]
    fn test_solh() {
        let mut a = array![
            [1.0, 4.0, 2.0, 3.0],
            [3.0, 4.0, 1.0, 7.0],
            [0.0, 2.0, 3.0, 4.0],
            [0.0, 0.0, 1.0, 3.0],
        ];
        let mut ip = Array1::<i32>::zeros(4);
        let mut b = array![1.0, 2.0, 3.0, 4.0];
        let ier = dech(4, a.view_mut(), 1, ip.view_mut());
        solh(4, a.view(), 1, b.view_mut(), ip.view());

        assert!((b[0] + 6.000000000000000).abs() < 1e-15);
        assert!((b[1] - 1.571428571428571).abs() < 1e-15);
        assert!((b[2] + 3.285714285714286).abs() < 1e-15);
        assert!((b[3] - 2.428571428571429).abs() < 1e-15);
    }

    #[test]
    fn test_solc() {
        let mut ar = array![
            [2f64.sqrt(), 5f64.sqrt(), -(3f64.sqrt())],
            [-(5f64.sqrt()), 2f64.sqrt(), 0.0],
            [3f64.sqrt(), 0.0, 2f64.sqrt()]
        ];
        let mut ai = array![
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 2f64.sqrt()],
            [0.0, -(2f64.sqrt()), 0.0]
        ];
        let mut ip = Array1::<i32>::zeros(3);
        let mut br = array![1.0, 2.0, 3.0];
        let mut bi = array![2f64.sqrt(), 0.0, 3f64.sqrt()];

        let ier = decc(3, ar.view_mut(), ai.view_mut(), ip.view_mut());
        solc(
            3,
            ar.view(),
            ai.view(),
            br.view_mut(),
            bi.view_mut(),
            ip.view(),
        );
        assert!((br[0] + 0.3936208598125455).abs() < 1e-15);
        assert!((br[1] - 2.190371768992845).abs() < 1e-15);
        assert!((br[2] - 1.929017439028988).abs() < 1e-15);

        assert!((bi[0] - 1.646538193454640).abs() < 1e-15);
        assert!((bi[1] - 0.6743880338588176).abs() < 1e-15);
        assert!((bi[2] - 1.398527432400491).abs() < 1e-15);
    }
}
