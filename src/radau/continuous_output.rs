use super::alg::Radau5;
use super::cache::Radau5Cache;
use crate::ode_integrator::OdeIntegrator;
use ndarray::prelude::*;

impl Radau5 {
    pub(crate) fn dense_output(
        &self,
        t: f64,
        integrator: &mut OdeIntegrator,
        cache: &mut Radau5Cache,
    ) -> Array1<f64> {
        let sq6 = 6f64.sqrt();
        let c1 = (4.0 - sq6) / 10.0;
        let c2 = (4.0 + sq6) / 10.0;
        let c1m1 = c1 - 1.0;
        let c2m1 = c2 - 1.0;

        let s = (t - integrator.t) / integrator.dtprev;
        let n = integrator.u.shape()[0];

        let mut res: Array1<f64> = Array1::<f64>::zeros(n);

        res.assign(&((s - c1m1) * &cache.cont.slice(s![(3 * n)..])));
        res = (s - c2m1) * (res + s * &cache.cont.slice(s![(2 * n)..(3 * n)]));
        res = s * (res + &cache.cont.slice(s![n..2 * n]));
        res + &cache.cont.slice(s![0..n])
    }
}
