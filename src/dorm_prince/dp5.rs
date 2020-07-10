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
        struct HO {
            w: f64,
        };
        let dudt = |mut du: ArrayViewMut1<f64>, u: ArrayView1<f64>, _t: f64, p: &HO| {
            du[0] = u[1];
            du[1] = -p.w * u[0];
        };
        let uinit = array![0.0, 1.0];
        let tspan = (0.0, 2.0);
        let mut integrator = OdeIntegratorBuilder::default(
            &dudt,
            uinit.clone(),
            tspan,
            DormandPrince5,
            HO { w: 1.0 },
        )
        .build();
        let reltol = integrator.opts.reltol;

        for (t, u) in (&mut integrator).into_iter() {
            println!("{}, {}", t, u);
            let dx = u[0] - t.sin();
            let dy = u[1] - t.cos();
            assert!(dx.abs() <= reltol * t.sin().abs() * 10.0);
            assert!(dy.abs() <= reltol * t.cos().abs() * 10.0);
        }
    }
    #[test]
    fn test_balldrop() {
        struct BallDrop {
            g: f64,
            m: f64,
            b: f64,
        };
        let dudt = |mut du: ArrayViewMut1<f64>, u: ArrayView1<f64>, _t: f64, p: &BallDrop| {
            du[0] = u[1];
            du[1] = -p.g - p.b / p.m * u[1];
        };
        let callback = |integrator: &mut OdeIntegrator<BallDrop, DormandPrince5>| {
            if integrator.u[0] < 0.0 && integrator.uprev[0] > 0.0 {
                integrator.u[1] *= -1.0;
            }
        };
        let uinit = array![0.0, 1.0];
        let tspan = (0.0, 10.0);
        let mut integrator = OdeIntegratorBuilder::default(
            &dudt,
            uinit.clone(),
            tspan,
            DormandPrince5,
            BallDrop {
                g: 1.0,
                m: 1.0,
                b: 1.0,
            },
        )
        .callback(&callback)
        .build();
        let reltol = integrator.opts.reltol;

        for (t, u) in (&mut integrator).into_iter() {
            println!("{}, {}", t, u);
        }
    }
}
