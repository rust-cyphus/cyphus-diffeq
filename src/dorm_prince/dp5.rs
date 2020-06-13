pub(crate) mod algorithm;
pub(crate) mod cache;
pub(crate) mod constants;
pub(crate) mod dense;
pub(crate) mod error;
pub(crate) mod prepare;
pub(crate) mod stages;

pub struct DormandPrince5;

#[cfg(test)]
mod test {
    use super::*;
    use crate::ode::*;
    use ndarray::prelude::*;

    #[test]
    fn test_ho() {
        struct HO;
        impl OdeFunction for HO {
            fn dudt(&mut self, mut du: ArrayViewMut1<f64>, u: ArrayView1<f64>, _t: f64) {
                du[0] = u[1];
                du[1] = -u[0];
            }
        }
        let uinit = array![0.0, 1.0];
        let tspan = (0.0, 2.0);

        let prob = OdeProblemBuilder::default(HO, uinit.clone(), tspan)
            .build()
            .unwrap();

        let mut integrator = DormandPrince5::init(prob);

        while let Some(_i) = integrator.step() {
            let dx = integrator.u[0] - integrator.t.sin();
            let dy = integrator.u[1] - integrator.t.cos();
            assert!(dx.abs() < integrator.opts.reltol * integrator.t.sin().abs() * 10.0);
            assert!(dy.abs() < integrator.opts.reltol * integrator.t.cos().abs() * 10.0);
        }
    }
}
