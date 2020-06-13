//! The `OdeAlgorithm` trait defines the basic behavior that an algorithm
//! for solving an ODE must exhibit. Each algorithm must be able to:
//! - yield default values for all the integrator parameters (`default`),
//! - yield an associated cache struct used to store specific working variables
//!   needed for integration of the ODE (`gen_cache`),
//! - and advance the integrator to the next state (`next`).

use super::function::OdeFunction;
use super::integrator::OdeIntegrator;
use super::problem::OdeProblem;

pub trait OdeAlgorithm {
    /// The `Cache` type is the cache struct associated with the algorithm.
    type Cache;
    /// Construct an integrator.
    fn init<T: OdeFunction>(prob: OdeProblem<T>) -> OdeIntegrator<T, Self>
    where
        Self: Sized;
    /// Advance the integrator to the next state.
    fn step<T: OdeFunction>(integrator: &mut OdeIntegrator<T, Self>) -> bool
    where
        Self: Sized;
}
