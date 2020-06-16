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

        let prob = OdeProblem::new(HO, uinit.clone(), tspan);
        let integrator = OdeIntegratorBuilder::default(prob, DormandPrince5).build();
        let reltol = integrator.opts.reltol;

        for (t, u) in integrator.into_iter() {
            println!("{}, {}", t, u);
            let dx = u[0] - t.sin();
            let dy = u[1] - t.cos();
            assert!(dx.abs() <= reltol * t.sin().abs() * 10.0);
            assert!(dy.abs() <= reltol * t.cos().abs() * 10.0);
        }
    }
}
