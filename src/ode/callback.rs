use super::algorithm::OdeAlgorithm;
use super::integrator::OdeIntegrator;

pub trait OdeCallBack {
    fn call<Params, Alg: OdeAlgorithm>(integrator: &mut OdeIntegrator<Params, Alg>);
}

Ode
