use super::function::OdeFunction;
use ndarray::prelude::*;

/// Structure for specifying an ordinary differential equation problem of the
/// form:
///     ODE:                u'_{i}(t) = f_{i}(u_1,...,u_n, t)
///     Initial conditions: u'_{i}(t_0) = u_{0,i}
///     Domain:             t0 <= t <= tf
#[derive(Clone)]
pub struct OdeProblem<T: OdeFunction> {
    /// Function structure representing the RHS of the ODE.
    pub func: T,
    /// Initial conditions.
    pub uinit: Array1<f64>,
    /// Initial and final time values.
    pub tspan: (f64, f64),
}

impl<T: OdeFunction> OdeProblem<T> {
    pub fn new(func: T, uinit: Array1<f64>, tspan: (f64, f64)) -> OdeProblem<T> {
        OdeProblem { func, uinit, tspan }
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_construction() {
        struct HO;
        impl OdeFunction for HO {
            fn dudt(&mut self, mut du: ArrayViewMut1<f64>, u: ArrayView1<f64>, t: f64) {
                du[0] = u[1];
                du[1] = -u[0];
            }
        }

        let uinit = array![0.0, 1.0];
        let tspan = (0.0, 1.0);
        let mass_matrix = Array2::<f64>::zeros((2, 2));

        let prob = OdeProblem::new(HO, uinit.clone(), tspan);

        let mut u = array![0.0, 1.0];
        let mut du = Array1::<f64>::zeros(2);
        let mut df = Array2::<f64>::zeros((2, 2));

        //prob.func.dudt(du.view_mut(), u.view(), 1.0);
        //(prob.dfdu.unwrap())(df.view_mut(), u.view(), 1.0);

        println!("du = {:?}", &du);
        println!("df = {:?}", &df);
    }
}
