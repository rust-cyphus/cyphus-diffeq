//! Module for performing decompositions of various matrices.

use ndarray::prelude::*;

/// Perform a triangular decomposition on a real dense matrix `a`. The upper and
/// lower triangular matricies are stored in `a` upon completion and the pivots
/// are stored in `ip`. Returns an error code which will be equal to zero if
/// decomposition is successful and a non-zero number which stage matrix was
/// found to be singular otherwise.
pub(crate) fn dec(n: usize, mut a: ArrayViewMut2<f64>, mut ip: ArrayViewMut1<i32>) -> usize {
    let mut ier = 0;
    ip[n - 1] = 1;
    if n != 1 {
        let nm1 = n - 1;
        for k in 0..nm1 {
            let kp1 = k + 1;
            let mut m = k;
            for i in kp1..n {
                if a[[i, k]].abs() > a[[m, k]].abs() {
                    m = i;
                }
            }
            ip[k] = m as i32;
            let mut t = a[[m, k]];
            if m != k {
                ip[n - 1] = -ip[n - 1];
                a[[m, k]] = a[[k, k]];
                a[[k, k]] = t;
            }
            if t == 0.0 {
                ier = k;
                ip[n - 1] = 0;
                return ier;
            }
            t = t.recip();
            for i in kp1..n {
                a[[i, k]] *= -t;
            }
            for j in kp1..n {
                t = a[[m, j]];
                a[[m, j]] = a[[k, j]];
                a[[k, j]] = t;
                if t != 0.0 {
                    for i in kp1..n {
                        a[[i, j]] += a[[i, k]] * t;
                    }
                }
            }
        }
    }
    if a[[n - 1, n - 1]] == 0.0 {
        ier = n;
        ip[n - 1] = 0;
    }

    return ier;
}

/// Perform a triangular decomposition on a real Hessenberg matrix `a` with
/// lower-bandwidth of `lb`. The upper and
/// lower triangular matricies are stored in `a` upon completion and the pivots
/// are stored in `ip`. Returns an error code which will be equal to zero if
/// decomposition is successful and a non-zero number which stage matrix was
/// found to be singular otherwise.
pub(crate) fn dech(
    n: usize,
    mut a: ArrayViewMut2<f64>,
    lb: usize,
    mut ip: ArrayViewMut1<i32>,
) -> usize {
    let mut ier = 0;
    ip[n - 1] = 1;
    if n != 1 {
        let nm1 = n - 1;
        for k in 0..nm1 {
            let kp1 = k + 1;
            let mut m = k;
            let na = n.min(lb + k + 1);
            for i in kp1..na {
                if a[[i, k]].abs() > a[[m, k]].abs() {
                    m = i;
                }
            }
            ip[k] = m as i32;
            let mut t = a[[m, k]];
            if m != k {
                ip[n - 1] = -ip[n - 1];
                a[[m, k]] = a[[k, k]];
                a[[k, k]] = t;
            }
            if t == 0.0 {
                ier = k;
                ip[n - 1] = 0;
                return ier;
            }
            t = 1.0 / t;
            for i in kp1..n {
                a[[i, k]] *= -t;
            }
            for j in kp1..n {
                t = a[[m, j]];
                a[[m, j]] = a[[k, j]];
                a[[k, j]] = t;
                if t != 0.0 {
                    for i in kp1..na {
                        a[[i, j]] += a[[i, k]] * t;
                    }
                }
            }
        }
    }
    if a[[n - 1, n - 1]] == 0.0 {
        ier = n;
        ip[n - 1] = 0;
    }

    return ier;
}

/// Perform a triangular decomposition on a complex matrix with real and
/// imaginary components `ar` and `ai`. The upper and
/// lower triangular matricies are stored in `a` upon completion and the pivots
/// are stored in `ip`. Returns an error code which will be equal to zero if
/// decomposition is successful and a non-zero number which stage matrix was
/// found to be singular otherwise.
pub(crate) fn decc(
    n: usize,
    mut ar: ArrayViewMut2<f64>,
    mut ai: ArrayViewMut2<f64>,
    mut ip: ArrayViewMut1<i32>,
) -> usize {
    let mut ier = 0;
    ip[n - 1] = 1;
    if n != 1 {
        let nm1 = n - 1;
        for k in 0..nm1 {
            let kp1 = k + 1;
            let mut m = k;
            for i in kp1..n {
                if ar[[i, k]].abs() + ai[[i, k]].abs() > ar[[m, k]].abs() + ai[[m, k]].abs() {
                    m = i;
                }
            }
            ip[k] = m as i32;
            let mut tr = ar[[m, k]];
            let mut ti = ai[[m, k]];
            if m != k {
                ip[n - 1] = -ip[n - 1];
                ar[[m, k]] = ar[[k, k]];
                ai[[m, k]] = ai[[k, k]];
                ar[[k, k]] = tr;
                ai[[k, k]] = ti;
            }
            if tr.abs() + ti.abs() == 0.0 {
                ier = k;
                ip[n - 1] = 0;
                return ier;
            }
            let den = tr * tr + ti * ti;
            tr = tr / den;
            ti = -ti / den;
            for i in kp1..n {
                let prodr = ar[[i, k]] * tr - ai[[i, k]] * ti;
                let prodi = ai[[i, k]] * tr + ar[[i, k]] * ti;
                ar[[i, k]] = -prodr;
                ai[[i, k]] = -prodi;
            }
            for j in kp1..n {
                tr = ar[[m, j]];
                ti = ai[[m, j]];
                ar[[m, j]] = ar[[k, j]];
                ai[[m, j]] = ai[[k, j]];
                ar[[k, j]] = tr;
                ai[[k, j]] = ti;
                if tr.abs() + ti.abs() == 0.0 {
                } else if ti == 0.0 {
                    for i in kp1..n {
                        let prodr = ar[[i, k]] * tr;
                        let prodi = ai[[i, k]] * tr;
                        ar[[i, j]] += prodr;
                        ai[[i, j]] += prodi;
                    }
                } else if tr == 0.0 {
                    for i in kp1..n {
                        let prodr = -ai[[i, k]] * ti;
                        let prodi = ar[[i, k]] * ti;
                        ar[[i, j]] += prodr;
                        ai[[i, j]] += prodi;
                    }
                } else {
                    for i in kp1..n {
                        let prodr = ar[[i, k]] * tr - ai[[i, k]] * ti;
                        let prodi = ai[[i, k]] * tr + ar[[i, k]] * ti;
                        ar[[i, j]] += prodr;
                        ai[[i, j]] += prodi;
                    }
                }
            }
        }
    }
    if ar[[n - 1, n - 1]].abs() + ai[[n - 1, n - 1]].abs() == 0.0 {
        ier = n;
        ip[n - 1] = 0;
    }

    return ier;
}

/// Perform a triangular decomposition on a complex Hessenberg matrix with real
/// and imaginary components `ar` and `ai` and lower-bandwidth `lb`. The upper and
/// lower triangular matricies are stored in `a` upon completion and the pivots
/// are stored in `ip`. Returns an error code which will be equal to zero if
/// decomposition is successful and a non-zero number which stage matrix was
/// found to be singular otherwise.
pub(crate) fn dechc(
    n: usize,
    mut ar: ArrayViewMut2<f64>,
    mut ai: ArrayViewMut2<f64>,
    lb: usize,
    mut ip: ArrayViewMut1<i32>,
) -> usize {
    let mut ier = 0;
    ip[n - 1] = 1;
    if (n != 1) && (lb != 0) {
        let nm1 = n - 1;
        for k in (0)..(nm1) {
            let kp1 = k + 1;
            let mut m = k;
            let na = n.min(lb + k + 1);
            for i in (kp1)..(na) {
                if ar[[i, k]].abs() + ai[[i, k]].abs() > ar[[m, k]].abs() + ai[[m, k]].abs() {
                    m = i;
                }
            }
            ip[k] = m as i32;
            let mut tr = ar[[m, k]];
            let mut ti = ai[[m, k]];
            if m != k {
                ip[n - 1] = -ip[n - 1];
                ar[[m, k]] = ar[[k, k]];
                ai[[m, k]] = ai[[k, k]];
                ar[[k, k]] = tr;
                ai[[k, k]] = ti;
            }
            if tr.abs() + ti.abs() == 0.0 {
                ier = k;
                ip[n - 1] = 0;
                return ier;
            }
            let den = tr * tr + ti * ti;
            tr = tr / den;
            ti = -ti / den;
            for i in (kp1)..(na) {
                let prodr = ar[[i, k]] * tr - ai[[i, k]] * ti;
                let prodi = ai[[i, k]] * tr + ar[[i, k]] * ti;
                ar[[i, k]] = -prodr;
                ai[[i, k]] = -prodi;
            }
            for j in (kp1)..(n) {
                tr = ar[[m, j]];
                ti = ai[[m, j]];
                ar[[m, j]] = ar[[k, j]];
                ai[[m, j]] = ai[[k, j]];
                ar[[k, j]] = tr;
                ai[[k, j]] = ti;
                if tr.abs() + ti.abs() == 0.0 {
                } else if ti == 0.0 {
                    for i in kp1..na {
                        let prodr = ar[[i, k]] * tr;
                        let prodi = ai[[i, k]] * tr;
                        ar[[i, j]] += prodr;
                        ai[[i, j]] += prodi;
                    }
                } else if tr == 0.0 {
                    for i in (kp1)..(na) {
                        let prodr = -ai[[i, k]] * ti;
                        let prodi = ar[[i, k]] * ti;
                        ar[[i, j]] += prodr;
                        ai[[i, j]] += prodi;
                    }
                } else {
                    for i in kp1..na {
                        let prodr = ar[[i, k]] * tr - ai[[i, k]] * ti;
                        let prodi = ai[[i, k]] * tr + ar[[i, k]] * ti;
                        ar[[i, j]] += prodr;
                        ai[[i, j]] += prodi;
                    }
                }
            }
        }
    }
    if ar[[n - 1, n - 1]].abs() + ai[[n - 1, n - 1]].abs() == 0.0 {
        ier = n;
        ip[n - 1] = 0;
    }

    return ier;
}

/// Perform a triangular decomposition on a real banded matrix `a` with upper
/// and lower-bandwidths of `lb` and `ub`. The upper and
/// lower triangular matricies are stored in `a` upon completion and the pivots
/// are stored in `ip`. Returns an error code which will be equal to zero if
/// decomposition is successful and a non-zero number which stage matrix was
/// found to be singular otherwise.
pub(crate) fn decb(
    n: usize,
    mut a: ArrayViewMut2<f64>,
    lb: usize,
    ub: usize,
    mut ip: ArrayViewMut1<i32>,
) -> usize {
    let mut ier = 0;
    ip[n - 1] = 1;
    let md = ub + lb;
    let md1 = md + 1;
    let mut ju = 0;
    if (n != 1) && (lb != 0) {
        if n >= ub + 2 {
            for j in (ub + 1)..n {
                for i in 0..lb {
                    a[[i, j]] = 0.0;
                }
            }
        }
        let nm1 = n - 1;
        for k in 0..nm1 {
            let kp1 = k + 1;
            let mut m = md;
            let mdl = lb.min(n - k - 1) + md;
            for i in md1..mdl {
                if a[[i, k]].abs() > a[[m, k]].abs() {
                    m = i;
                }
            }
            ip[k] = (m + k - md) as i32;
            let mut t = a[[m, k]];
            if m != md {
                ip[n - 1] = -ip[n - 1];
                a[[m, k]] = a[[md, k]];
                a[[md, k]] = t;
            }
            if t == 0.0 {
                ier = k;
                ip[n - 1] = 0;
                return ier;
            }
            t = 1.0 / t;
            for i in md1..(mdl + 1) {
                a[[i, k]] *= -t;
            }
            ju = ju.max(ub + ip[k] as usize + 1).min(n);
            let mm = md;
            if ju >= kp1 {
                for j in kp1..ju {
                    m = m - 1;
                    let mm = mm - 1;
                    t = a[[m, j]];
                    if m != mm {
                        a[[m, j]] = a[[mm, j]];
                        a[[mm, j]] = t;
                    }
                    if t != 0.0 {
                        let jk = j - k;
                        for i in md1..mdl {
                            let ijk = i - jk;
                            a[[ijk, j]] += a[[i, k]] * t;
                        }
                    }
                }
            }
        }
    }
    if a[[md, n - 1]] == 0.0 {
        ier = n;
        ip[n - 1] = 0;
    }

    return ier;
}

/// Perform a triangular decomposition on a complex banded matrix with real and
/// imaginary components `ar` and `ai` and upper and lower-bandwidths of `lb`
/// and `ub`. The upper and
/// lower triangular matricies are stored in `a` upon completion and the pivots
/// are stored in `ip`. Returns an error code which will be equal to zero if
/// decomposition is successful and a non-zero number which stage matrix was
/// found to be singular otherwise.
pub(crate) fn decbc(
    n: usize,
    mut ar: ArrayViewMut2<f64>,
    mut ai: ArrayViewMut2<f64>,
    lb: usize,
    ub: usize,
    mut ip: ArrayViewMut1<i32>,
) -> usize {
    let mut ier = 0;
    ip[n - 1] = 1;
    let md = lb + ub;
    let md1 = md + 1;
    let mut ju = 0;
    if (n != 1) && (lb != 0) {
        if n >= ub + 2 {
            for j in (ub + 1)..n {
                for i in (0)..(lb) {
                    ar[[i, j]] = 0.0;
                    ai[[i, j]] = 0.0;
                }
            }
        }
        let nm1 = n - 1;
        for k in 0..nm1 {
            let kp1 = k + 1;
            let mut m = md;
            let mdl = lb.min(n - k - 1) + md;
            for i in md1..mdl {
                if ar[[i, k]].abs() + ai[[i, k]].abs() > ar[[m, k]].abs() + ai[[m, k]].abs() {
                    m = i;
                }
            }
            ip[k] = (m + k - md) as i32;
            let mut tr = ar[[m, k]];
            let mut ti = ai[[m, k]];
            if m != k {
                ip[n - 1] = -ip[n - 1];
                ar[[m, k]] = ar[[md, k]];
                ai[[m, k]] = ai[[md, k]];
                ar[[md, k]] = tr;
                ai[[md, k]] = ti;
            }
            if tr.abs() + ti.abs() == 0.0 {
                ier = k;
                ip[n - 1] = 0;
                return ier;
            }
            let den = tr * tr + ti * ti;
            tr = tr / den;
            ti = -ti / den;
            for i in md1..mdl {
                let prodr = ar[[i, k]] * tr - ai[[i, k]] * ti;
                let prodi = ai[[i, k]] * tr + ar[[i, k]] * ti;
                ar[[i, k]] = -prodr;
                ai[[i, k]] = -prodi;
            }
            ju = ju.max(ub + ip[k] as usize + 1).min(n);
            let mut mm = md;
            if ju >= kp1 {
                for j in kp1..ju {
                    m -= 1;
                    mm -= 1;
                    tr = ar[[m, j]];
                    ti = ai[[m, j]];
                    if m != mm {
                        ar[[m, j]] = ar[[mm, j]];
                        ai[[m, j]] = ai[[mm, j]];
                        ar[[mm, j]] = tr;
                        ai[[mm, j]] = ti;
                    }
                    if tr.abs() + ti.abs() == 0.0 {
                    } else if ti == 0.0 {
                        let jk = j - k;
                        for i in md1..mdl {
                            let ijk = i - jk;
                            let prodr = ar[[i, k]] * tr;
                            let prodi = ai[[i, k]] * tr;
                            ar[[ijk, j]] += prodr;
                            ai[[ijk, j]] += prodi;
                        }
                    } else if tr == 0.0 {
                        let jk = j - k;
                        for i in md1..mdl {
                            let ijk = i - jk;
                            let prodr = -ai[[i, k]] * ti;
                            let prodi = ar[[i, k]] * ti;
                            ar[[ijk, j]] += prodr;
                            ai[[ijk, j]] += prodi;
                        }
                    } else {
                        let jk = j - k;
                        for i in md1..mdl {
                            let ijk = i - jk;
                            let prodr = ar[[i, k]] * tr - ai[[i, k]] * ti;
                            let prodi = ai[[i, k]] * tr + ar[[i, k]] * ti;
                            ar[[ijk, j]] += prodr;
                            ai[[ijk, j]] += prodi;
                        }
                    }
                }
            }
        }
    }
    if ar[[md, n - 1]].abs() + ai[[md, n - 1]].abs() == 0.0 {
        ier = n;
        ip[n - 1] = 0;
    }

    return ier;
}

pub(crate) fn elmhes(
    n: usize,
    low: usize,
    igh: usize,
    mut a: ArrayViewMut2<f64>,
    mut inter: ArrayViewMut1<i32>,
) {
    let la = igh - 2;
    let kp1 = low + 1;
    if la < kp1 {
        return;
    }
    for m in kp1..(la + 1) {
        let mm1 = m - 1;
        let mut x: f64 = 0.0;
        let mut ii = m;
        for j in m..igh {
            if a[[j, mm1]].abs() > x.abs() {
                x = a[[j, mm1]];
                ii = j;
            }
        }
        inter[m] = ii as i32;
        if ii != m {
            // interchange rows and columns of a
            for j in mm1..n {
                let y = a[[ii, j]];
                a[[ii, j]] = a[[m, j]];
                a[[m, j]] = y;
            }
            for j in 0..igh {
                let y = a[[j, ii]];
                a[[j, ii]] = a[[j, m]];
                a[[j, m]] = y;
            }
        }
        if x != 0.0 {
            let mp1 = m + 1;
            for i in mp1..igh {
                let mut y = a[[i, mm1]];
                if y == 0.0 {
                    return;
                }
                y = y / x;
                a[[i, mm1]] = y;
                for j in m..n {
                    a[[i, j]] -= y * a[[m, j]];
                }
                for j in 0..igh {
                    a[[i, m]] += y * a[[j, i]];
                }
            }
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_dec() {
        let mut a = array![
            [0.510063, 0.0657526, 0.280477],
            [0.00369809, 0.462848, 0.901661],
            [0.875906, 0.662457, 0.130995]
        ];
        let mut ip: Array1<i32> = Array::zeros(3);
        let ier = dec(3, a.view_mut(), ip.view_mut());
        println!("a = {:?}", a);
        println!("ip = {:?}", ip);
        println!("ier = {:?}", ier);
    }
}
