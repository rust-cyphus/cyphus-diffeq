pub(crate) mod algorithm;
pub(crate) mod cache;
pub(crate) mod constants;
pub(crate) mod continuous_output;
pub(crate) mod dec;
pub(crate) mod error_estimate;
pub(crate) mod linear_solve;
pub(crate) mod newton;
pub(crate) mod step_size_control;

/// The Radua5 algorithm.
pub struct Radau5;

#[cfg(test)]
mod test {
    use super::*;
    use crate::ode::*;
    use ndarray::prelude::*;

    #[test]
    fn test_van_der_pol() {
        struct VanDerPol {
            mu: f64,
        };
        let uinit = array![2.0, -0.66];
        let tspan = (0.0, 2.0);
        impl OdeFunction for VanDerPol {
            fn dudt(&mut self, mut du: ArrayViewMut1<f64>, u: ArrayView1<f64>, _t: f64) {
                du[0] = u[1];
                du[1] = ((1.0 - u[0] * u[0]) * u[1] - u[0]) / self.mu;
            }
            fn dfdu(&mut self, mut df: ArrayViewMut2<f64>, u: ArrayView1<f64>, _t: f64) {
                df[[0, 0]] = 0.0;
                df[[0, 1]] = 1.0;
                df[[1, 0]] = -(2.0 * u[0] * u[1] + 1.0) / self.mu;
                df[[1, 1]] = (1.0 - u[0] * u[0]) / self.mu;
            }
        }
        let func = VanDerPol { mu: 1e-6 };
        let prob = OdeProblem::new(func, uinit.clone(), tspan);
        let mut integrator = OdeIntegratorBuilder::default(prob, Radau5)
            .reltol(1e-7)
            .abstol(1e-7)
            .build();

        for (t, u) in (&mut integrator).into_iter() {
            println!("{}, {}", t, u);
        }
        println!("{:?}", integrator.sol.retcode);
    }
}
