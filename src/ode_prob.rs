use ndarray::prelude::*;

pub enum OdeProblemBuildErr {
    InvalidIndexPars(usize, usize, usize),
    Invalid2ndOrderPars(usize, usize),
    InvlaidMassBw(usize, usize, usize, usize),
}

impl std::fmt::Debug for OdeProblemBuildErr {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            OdeProblemBuildErr::InvalidIndexPars(idx1, idx2, idx3) => write!(
                f,
                "Invalid index varibles: {}, {}, {}. Sum must be equal to size of the system.",
                idx1, idx2, idx3
            ),
            OdeProblemBuildErr::Invalid2ndOrderPars(m1, m2) => write!(
                f,
                "Invalid 2nd order parameters: m1={}, m2={}. Sum must be less that size of system.",
                m1, m2
            ),
            OdeProblemBuildErr::InvlaidMassBw(jl, ju, ml, mu) => write!(
                f,
                "Bandwidth of mass matrix cannot be larger than bandwidths of Jacobian: jl = {}, \
                ju = {}, ml = {}, mu = {}.",
                jl, ju, ml, mu
            ),
        }
    }
}

/// Structure for specifying an ordinary differential equation problem of the
/// form:
///     ODE:                u'_{i}(t) = f_{i}(u_1,...,u_n, t)
///     Initial conditions: u'_{i}(t_0) = u_{0,i}
///     Domain:             t0 <= t <= tf
pub struct OdeProblem<F, J>
where
    F: Fn(ArrayViewMut1<f64>, ArrayView1<f64>, f64),
    J: Fn(ArrayViewMut2<f64>, ArrayView1<f64>, f64),
{
    /// RHS of the ordinary differential equation. Should be of the form:
    /// dudt = dudt(du, u, t) where `du` is modified in place.
    pub dudt: F,
    /// Jacobian of the RHS of the ordinary differential equation. Should be of
    /// the form: dfdu = dfdu(df, u, t) where the jacobian `df` is modified in
    /// place.
    pub dfdu: Option<J>,
    /// Mass matrix for DAE.
    pub mass_matrix: Option<Array2<f64>>,
    /// Initial conditions.
    pub uinit: Array1<f64>,
    /// Initial and final time values.
    pub tspan: (f64, f64),
    /// Number of index 1 variables. For ODEs, this is equal to the size of the
    /// system
    pub num_index1_vars: usize,
    /// Number of index 2 variables. For ODEs, this is equal to zero.
    pub num_index2_vars: usize,
    /// Number of index 3 variables. For ODEs, this is equal to zero.
    pub num_index3_vars: usize,
    /// Second order ODE structure constant: u[i]' = u[i+m2] for i = 1,...,m1.
    pub m1: usize,
    /// Second order ODE structure constant: u[i]' = u[i+m2] for i = 1,...,m1.
    pub m2: usize,
    /// Upper bandwidth of the Jacobian
    pub jac_ubw: usize,
    /// Lower bandwidth of the Jacobian
    pub jac_lbw: usize,
    /// Upper bandwidth of the mass matrix
    pub mm_ubw: usize,
    /// Lower bandwidth of the mass matrix
    pub mm_lbw: usize,
}

pub struct OdeProblemBuilder<F, J>
where
    F: Fn(ArrayViewMut1<f64>, ArrayView1<f64>, f64),
    J: Fn(ArrayViewMut2<f64>, ArrayView1<f64>, f64),
{
    /// RHS of the ordinary differential equation. Should be of the form:
    /// dudt = dudt(du, u, t) where `du` is modified in place.
    pub dudt: F,
    /// Jacobian of the RHS of the ordinary differential equation. Should be of
    /// the form: dfdu = dfdu(df, u, t) where the jacobian `df` is modified in
    /// place.
    pub dfdu: Option<J>,
    /// Mass matrix of the system (differential-algebraic system): Mu'(t) = f(u,t).
    pub mass_matrix: Option<Array2<f64>>,
    /// Initial conditions.
    pub uinit: Array1<f64>,
    /// Initial and final time values.
    pub tspan: (f64, f64),
    /// Number of index 1 variables. For ODEs, this is equal to the size of the
    /// system
    pub(crate) num_index1_vars: Option<usize>,
    /// Number of index 2 variables. For ODEs, this is equal to zero.
    pub(crate) num_index2_vars: Option<usize>,
    /// Number of index 3 variables. For ODEs, this is equal to zero.
    pub(crate) num_index3_vars: Option<usize>,
    /// Second order ODE structure constant: u[i]' = u[i+m2] for i = 1,...,m1.
    pub(crate) m1: Option<usize>,
    /// Second order ODE structure constant: u[i]' = u[i+m2] for i = 1,...,m1.
    pub(crate) m2: Option<usize>,
    /// Upper bandwidth of the Jacobian
    pub(crate) jac_ubw: Option<usize>,
    /// Lower bandwidth of the Jacobian
    pub(crate) jac_lbw: Option<usize>,
    /// Upper bandwidth of the mass matrix
    pub(crate) mm_ubw: Option<usize>,
    /// Lower bandwidth of the mass matrix
    pub(crate) mm_lbw: Option<usize>,
}

impl<F, J> OdeProblemBuilder<F, J>
where
    F: Fn(ArrayViewMut1<f64>, ArrayView1<f64>, f64),
    J: Fn(ArrayViewMut2<f64>, ArrayView1<f64>, f64),
{
    pub fn default(dudt: F, uinit: Array1<f64>, tspan: (f64, f64)) -> OdeProblemBuilder<F, J> {
        let n = uinit.shape()[0];
        OdeProblemBuilder {
            dudt,
            dfdu: None,
            mass_matrix: None,
            uinit,
            tspan,
            num_index1_vars: None,
            num_index2_vars: None,
            num_index3_vars: None,
            m1: None,
            m2: None,
            jac_ubw: None,
            jac_lbw: None,
            mm_ubw: None,
            mm_lbw: None,
        }
    }
    /// Add a jacobian to the builder.
    pub fn with_jac(mut self, dfdu: J) -> OdeProblemBuilder<F, J> {
        self.dfdu = Some(dfdu);
        self
    }
    /// Add a mass matrix to the builder.
    pub fn with_mass(mut self, mass_matrix: Array2<f64>) -> OdeProblemBuilder<F, J> {
        self.mass_matrix = Some(mass_matrix);
        self
    }
    /// Specify the number of index-1 variables of the differential-algebraic
    /// system. An index-1 problem is one in which the system can be converted
    /// into an ODE by taking a single derivive of one of the equations.
    pub fn num_index1_vars(mut self, val: usize) -> OdeProblemBuilder<F, J> {
        self.num_index1_vars = Some(val);
        self
    }
    /// Specify the number of index-2 variables of the differential-algebraic
    /// system. An index-2 problem is one in which the system can be converted
    /// into an ODE by taking derivatives of two of the equations.
    pub fn num_index2_vars(mut self, val: usize) -> OdeProblemBuilder<F, J> {
        self.num_index2_vars = Some(val);
        self
    }
    /// Specify the number of index-3 variables of the differential-algebraic
    /// system. An index-3 problem is one in which the system can be converted
    /// into an ODE by taking derivatives of three of the equations.
    pub fn num_index3_vars(mut self, val: usize) -> OdeProblemBuilder<F, J> {
        self.num_index3_vars = Some(val);
        self
    }
    /// Specify the second order parameters m1, and m2 of the problem. These
    /// are defined such that u'[i] = u[i+m2] for i = 1,...,m1 which is a
    /// structure that often occurs for second-order differential equations.
    pub fn second_order_params(mut self, val: (usize, usize)) -> OdeProblemBuilder<F, J> {
        self.m1 = Some(val.0);
        self.m2 = Some(val.1);
        self
    }
    /// Specify the lower and upper bandwidths of the Jacobian.
    pub fn jac_bandwidths(mut self, val: (usize, usize)) -> OdeProblemBuilder<F, J> {
        self.jac_lbw = Some(val.0);
        self.jac_ubw = Some(val.1);
        self
    }
    /// Specify the lower and upper bandwidths of the mass matrix.
    pub fn mass_bandwidths(mut self, val: (usize, usize)) -> OdeProblemBuilder<F, J> {
        self.mm_lbw = Some(val.0);
        self.mm_ubw = Some(val.1);
        self
    }
    /// Try to build the OdeProblem.
    pub fn build(self) -> Result<OdeProblem<F, J>, OdeProblemBuildErr> {
        let n = self.uinit.shape()[0];

        let nind1 = self.num_index1_vars.unwrap_or(n);
        let nind2 = self.num_index2_vars.unwrap_or(0);
        let nind3 = self.num_index3_vars.unwrap_or(0);

        if nind1 + nind2 + nind3 != n {
            return Err(OdeProblemBuildErr::InvalidIndexPars(nind1, nind2, nind3));
        }

        let m1 = self.m1.unwrap_or(0);
        let mut m2 = self.m1.unwrap_or(0);
        if m1 == 0 {
            m2 = n;
        } else if m2 == 0 {
            m2 = m1;
        }
        if m1 + m2 > n {
            return Err(OdeProblemBuildErr::Invalid2ndOrderPars(m1, m2));
        }

        let mut jac_ubw = self.jac_ubw.unwrap_or(n);
        let mut jac_lbw = self.jac_lbw.unwrap_or(n);

        let mm_ubw = self.mm_ubw.unwrap_or(n);
        let mm_lbw = self.mm_lbw.unwrap_or(n);

        let nm1 = n - m1;
        let jband = jac_lbw < nm1;

        if !jband {
            jac_ubw = nm1;
            jac_lbw = nm1;
        }

        if !self.mass_matrix.is_none() {
            if mm_lbw > jac_lbw || mm_ubw > jac_ubw {
                return Err(OdeProblemBuildErr::InvlaidMassBw(
                    jac_lbw, jac_ubw, mm_lbw, mm_ubw,
                ));
            }
        }

        Ok(OdeProblem {
            dudt: self.dudt,
            dfdu: self.dfdu,
            mass_matrix: self.mass_matrix,
            uinit: self.uinit,
            tspan: self.tspan,
            num_index1_vars: nind1,
            num_index2_vars: nind2,
            num_index3_vars: nind3,
            m1,
            m2,
            jac_ubw,
            jac_lbw,
            mm_ubw,
            mm_lbw,
        })
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_construction() {
        let dudt = |mut du: ArrayViewMut1<f64>, u: ArrayView1<f64>, t: f64| {
            du[0] = u[1];
            du[1] = -u[0];
        };
        let dfdu = |mut df: ArrayViewMut2<f64>, u: ArrayView1<f64>, t: f64| {
            df[[0, 0]] = 0.0;
            df[[0, 1]] = 0.0;
            df[[1, 0]] = -1.0;
            df[[1, 1]] = 0.0;
        };

        let uinit = array![0.0, 1.0];
        let tspan = (0.0, 1.0);
        let mass_matrix = Array2::<f64>::zeros((2, 2));

        let prob = OdeProblemBuilder::default(dudt, uinit.clone(), tspan)
            .with_jac(dfdu)
            .with_mass(mass_matrix)
            .build()
            .unwrap();

        let mut u = array![0.0, 1.0];
        let mut du = Array1::<f64>::zeros(2);
        let mut df = Array2::<f64>::zeros((2, 2));

        (prob.dudt)(du.view_mut(), u.view(), 1.0);
        (prob.dfdu.unwrap())(df.view_mut(), u.view(), 1.0);

        println!("du = {:?}", &du);
        println!("df = {:?}", &df);
    }
}
