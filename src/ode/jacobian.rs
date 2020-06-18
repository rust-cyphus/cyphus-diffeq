use ndarray::prelude::*;

pub(crate) fn jacobian_u<Dudt, Dfdu, Params>(
    dudt: Dudt,
    dfdu: Option<Dfdu>,
    mut df: ArrayViewMut2<f64>,
    u: ArrayView1<f64>,
    t: f64,
    p: &Params,
) where
    Dudt: Fn(ArrayViewMut1<f64>, ArrayView1<f64>, f64, &Params),
    Dfdu: Fn(ArrayViewMut2<f64>, ArrayView1<f64>, f64, &Params),
{
    match dfdu {
        Some(f) => f(df.view_mut(), u.view(), t, p),
        None => {
            let mut utemp = Array::zeros(u.raw_dim());
            let mut du0 = Array::zeros(u.raw_dim());
            let mut du1 = Array::zeros(u.raw_dim());
            utemp.assign(&u);
            dudt(du0.view_mut(), u.view(), t, p);
            for i in 0..u.len() {
                let usafe = utemp[i];
                let del = (f64::EPSILON * usafe.max(1e-5)).sqrt();
                utemp[i] = usafe + del;
                dudt(du1.view_mut(), utemp.view(), t, p);
                for j in 0..u.len() {
                    df[[j, i]] = (du1[j] - du0[j]) / del;
                }
                utemp[i] = usafe;
            }
        }
    }
}

pub(crate) fn jacobian_t<Dudt, Dfdt, Params>(
    dudt: Dudt,
    dfdt: Option<Dfdt>,
    mut df: ArrayViewMut1<f64>,
    u: ArrayView1<f64>,
    t: f64,
    p: &Params,
) where
    Dudt: Fn(ArrayViewMut1<f64>, ArrayView1<f64>, f64, &Params),
    Dfdt: Fn(ArrayViewMut1<f64>, ArrayView1<f64>, f64, &Params),
{
    match dfdt {
        Some(f) => f(df.view_mut(), u.view(), t, p),
        None => {
            let mut du = Array::zeros(u.raw_dim());
            dudt(du.view_mut(), u.view(), t, p);
            dudt(df.view_mut(), u.view(), t, p);
            let dt = f64::EPSILON * t.max(1e-5);
            df.assign(&((&df - &du) / dt));
        }
    }
}
