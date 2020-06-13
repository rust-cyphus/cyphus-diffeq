use ndarray::prelude::*;

/// Core trait implementing the RHS of the differential equation.
///
/// # Examples
/// Harmonic Oscillator:
/// ```
/// pub struct HO;
/// impl HO {
///     fn dudt(&mut self, du: ArrayViewMut1<f64>, u: ArrayView1<f64>, t: f64){
///         du[0] = u[1];
///         du[1] = -u[0];
///     }
/// }
/// ```
pub trait OdeFunction {
    /// RHS of the ordinary differential equation. Should be of the form:
    /// dudt = dudt(du, u, t) where `du` is modified in place.
    fn dudt(&mut self, du: ArrayViewMut1<f64>, u: ArrayView1<f64>, t: f64);
    /// Jacobian of the RHS of the ordinary differential equation w.r.t. `u`.
    /// Should be of the form: dfdu = dfdu(df, u, t) where the jacobian `df`
    /// is modified in place.
    fn dfdu(&mut self, mut df: ArrayViewMut2<f64>, u: ArrayView1<f64>, t: f64) {
        let mut utemp = Array::zeros(u.raw_dim());
        let mut du0 = Array::zeros(u.raw_dim());
        let mut du1 = Array::zeros(u.raw_dim());

        utemp.assign(&u);
        self.dudt(du0.view_mut(), u.view(), t);

        for i in 0..u.len() {
            let usafe = utemp[i];
            let del = (f64::EPSILON * usafe.max(1e-5)).sqrt();
            utemp[i] = usafe + del;
            self.dudt(du1.view_mut(), utemp.view(), t);
            for j in 0..u.len() {
                df[[j, i]] = (du1[j] - du0[j]) / del;
            }
            utemp[i] = usafe;
        }
    }
    /// Jacobian of the RHS of the ordinary differential equation w.r.t time.
    /// Should be of the form: dfdu = dfdu(df, u, t) where the jacobian `df`
    /// is modified in
    fn dfdt(&mut self, mut df: ArrayViewMut1<f64>, u: ArrayView1<f64>, t: f64) {
        let mut du = Array::zeros(u.raw_dim());
        self.dudt(du.view_mut(), u.view(), t);
        self.dudt(df.view_mut(), u.view(), t);
        let dt = f64::EPSILON * t.max(1e-5);
        df.assign(&((&df - &du) / dt));
    }
}
