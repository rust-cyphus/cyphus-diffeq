/// Statistics for a differential equation integration
#[derive(Debug, Clone)]
pub struct OdeStatistics {
    /// Number of steps taken
    pub steps: usize,
    /// Number of function evaluations
    pub function_evals: usize,
    /// Number of jacobian evaluations
    pub jacobian_evals: usize,
    /// Number of decompositions performed
    pub decompositions: usize,
    /// Number of linear solves performed
    pub linear_solves: usize,
    /// Number of accepted steps
    pub accepts: usize,
    /// Number of rejected steps
    pub rejects: usize,
}

impl OdeStatistics {
    pub fn new() -> Self {
        OdeStatistics {
            /// Number of steps taken
            steps: 0,
            function_evals: 0,
            jacobian_evals: 0,
            decompositions: 0,
            linear_solves: 0,
            accepts: 0,
            rejects: 0,
        }
    }
}
