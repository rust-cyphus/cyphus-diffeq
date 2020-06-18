use super::Radau5;
use crate::linalg::dec::*;
use crate::ode::*;
use ndarray::prelude::*;

impl Radau5 {
    /// Perform a decomposition on the real matrices used for solving ODE using
    /// Radau5
    pub(crate) fn decomp_real<Params>(integrator: &mut OdeIntegrator<Params, Self>) -> usize {
        // E1 = fac1 * I - dfdu
        integrator.cache.e1.assign(
            &(integrator.cache.fac1 * Array::eye(integrator.cache.e1.shape()[0])
                - &integrator.cache.dfdu),
        );

        dec(
            integrator.cache.e1.nrows(),
            integrator.cache.e1.view_mut(),
            integrator.cache.ip1.view_mut(),
        )
    }

    /// Perform a decomposition on the complex matrices used for solving ODE using
    /// Radau5
    pub(crate) fn decomp_complex<Params>(integrator: &mut OdeIntegrator<Params, Self>) -> usize {
        let n = integrator.cache.e2r.nrows();
        let iden = Array::eye(n);

        // E2r = alpha * I - dfdu
        integrator
            .cache
            .e2r
            .assign(&(integrator.cache.alphn * &iden - &integrator.cache.dfdu));
        // E2i = beta * I
        integrator
            .cache
            .e2i
            .assign(&(integrator.cache.betan * &iden));
        decc(
            n,
            integrator.cache.e2r.view_mut(),
            integrator.cache.e2i.view_mut(),
            integrator.cache.ip2.view_mut(),
        )
    }
    /// Perform the needed decompositions for the Radau5 algorithm. Returns
    /// true if the decompositions were successful. False otherwise.
    pub(super) fn perform_decompositions<Params>(
        integrator: &mut OdeIntegrator<Params, Self>,
    ) -> bool {
        // compute the matrices e1 and e2 and their decompositions
        integrator.cache.fac1 = Radau5::U1 / integrator.dt;
        integrator.cache.alphn = Radau5::ALPHA / integrator.dt;
        integrator.cache.betan = Radau5::BETA / integrator.dt;

        if Self::decomp_real(integrator) == 0 {
            if Self::decomp_complex(integrator) == 0 {
                integrator.stats.decompositions += 1;
                return true;
            }
        }

        integrator.cache.nsing += 1;
        if integrator.cache.nsing >= 5 {
            integrator.sol.retcode = OdeRetCode::SingularMatrix;
            return false;
        }
        integrator.dt *= 0.5;
        integrator.cache.dtfac = 0.5;
        integrator.cache.reject = true;
        integrator.cache.last = false;
        if !integrator.cache.caljac {
            jacobian_u(
                integrator.dudt,
                integrator.dfdu,
                integrator.cache.dfdu.view_mut(),
                integrator.u.view(),
                integrator.t,
                &integrator.params,
            );
        }
        return false;
    }
}
