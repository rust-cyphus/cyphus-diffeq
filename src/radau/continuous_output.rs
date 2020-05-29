use super::cache::Radau5Cache;
use crate::de_integrator::ODEIntegrator;
use ndarray::prelude::*;

impl Radau5Cache {
    pub(crate) fn continue_output<F, J>(
        &self,
        t: f64,
        integrator: &mut ODEIntegrator<F, J>,
        i: usize,
    ) -> f64
    where
        F: Fn(ArrayViewMut1<f64>, ArrayView1<f64>, f64),
        J: Fn(ArrayViewMut2<f64>, ArrayView1<f64>, f64),
    {
        let sq6 = 6f64.sqrt();
        let c1 = (4.0 - sq6) / 10.0;
        let c2 = (4.0 + sq6) / 10.0;
        let c1m1 = c1 - 1.0;
        let c2m1 = c2 - 1.0;

        let s = (t - integrator.t) / integrator.dtprev;
        let n = integrator.u.shape()[0];

        self.cont[i]
            + s * (self.cont[i + n]
                + (s - c2m1) * (self.cont[i + 2 * n] + (s - c1m1) * self.cont[i + 3 * n]))
    }
}
