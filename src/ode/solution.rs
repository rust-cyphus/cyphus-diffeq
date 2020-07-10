use super::code::OdeRetCode;
use ndarray::prelude::*;

/// Solution object for holding solution to an ODE.
pub struct OdeSolution {
    pub ts: Vec<f64>,
    pub us: Vec<Array1<f64>>,
    pub retcode: OdeRetCode,
}

impl OdeSolution {
    pub fn new() -> OdeSolution {
        OdeSolution {
            ts: vec![],
            us: vec![],
            retcode: OdeRetCode::Continue,
        }
    }
}

pub struct OdeSolutionIterator {
    pub iter_t: std::vec::IntoIter<f64>,
    pub iter_u: std::vec::IntoIter<Array1<f64>>,
}

impl<'a> IntoIterator for OdeSolution {
    type Item = (f64, Array1<f64>);
    type IntoIter = OdeSolutionIterator;

    fn into_iter(self) -> Self::IntoIter {
        OdeSolutionIterator {
            iter_t: self.ts.into_iter(),
            iter_u: self.us.into_iter(),
        }
    }
}

impl Iterator for OdeSolutionIterator {
    type Item = (f64, Array1<f64>);

    fn next(&mut self) -> Option<Self::Item> {
        let t = self.iter_t.next();
        let u = self.iter_u.next();
        if t.is_some() && u.is_some() {
            Some((t.unwrap(), u.unwrap()))
        } else {
            None
        }
    }
}

pub struct OdeSolutionMutIterator<'a> {
    iter_t: std::slice::Iter<'a, f64>,
    iter_u: std::slice::Iter<'a, Array1<f64>>,
}

impl<'a> IntoIterator for &'a mut OdeSolution {
    type Item = (&'a f64, &'a Array1<f64>);
    type IntoIter = OdeSolutionMutIterator<'a>;

    fn into_iter(self) -> Self::IntoIter {
        OdeSolutionMutIterator {
            iter_t: self.ts.iter(),
            iter_u: self.us.iter(),
        }
    }
}

impl<'a> Iterator for OdeSolutionMutIterator<'a> {
    type Item = (&'a f64, &'a Array1<f64>);

    fn next(&mut self) -> Option<Self::Item> {
        let t = self.iter_t.next();
        let u = self.iter_u.next();
        if t.is_some() && u.is_some() {
            Some((t.unwrap(), u.unwrap()))
        } else {
            None
        }
    }
}
