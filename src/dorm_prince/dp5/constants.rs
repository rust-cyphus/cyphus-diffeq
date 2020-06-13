use super::DormandPrince5;

impl DormandPrince5 {
    // Butcher c_i's
    pub(crate) const C2: f64 = 0.2;
    pub(crate) const C3: f64 = 0.3;
    pub(crate) const C4: f64 = 0.8;
    pub(crate) const C5: f64 = 8.0 / 9.0;
    // But_ij's
    pub(crate) const A21: f64 = 0.2;
    pub(crate) const A31: f64 = 3.0 / 40.0;
    pub(crate) const A32: f64 = 9.0 / 40.0;
    pub(crate) const A41: f64 = 44.0 / 45.0;
    pub(crate) const A42: f64 = -56.0 / 15.0;
    pub(crate) const A43: f64 = 32.0 / 9.0;
    pub(crate) const A51: f64 = 19372.0 / 6561.0;
    pub(crate) const A52: f64 = -25360.0 / 2187.0;
    pub(crate) const A53: f64 = 64448.0 / 6561.0;
    pub(crate) const A54: f64 = -212.0 / 729.0;
    pub(crate) const A61: f64 = 9017.0 / 3168.0;
    pub(crate) const A62: f64 = -355.0 / 33.0;
    pub(crate) const A63: f64 = 46732.0 / 5247.0;
    pub(crate) const A64: f64 = 49.0 / 176.0;
    pub(crate) const A65: f64 = -5103.0 / 18656.0;
    // But_i's: y1 = y0 + h * (b1 * k1 + ... + bs * ks)
    pub(crate) const A71: f64 = 35.0 / 384.0;
    pub(crate) const A73: f64 = 500.0 / 1113.0;
    pub(crate) const A74: f64 = 125.0 / 192.0;
    pub(crate) const A75: f64 = -2187.0 / 6784.0;
    pub(crate) const A76: f64 = 11.0 / 84.0;
    // Errimation constants: b_i^* - b_i
    pub(crate) const E1: f64 = 71.0 / 57600.0;
    pub(crate) const E3: f64 = -71.0 / 16695.0;
    pub(crate) const E4: f64 = 71.0 / 1920.0;
    pub(crate) const E5: f64 = -17253.0 / 339200.0;
    pub(crate) const E6: f64 = 22.0 / 525.0;
    pub(crate) const E7: f64 = -1.0 / 40.0;
    // Cons output parameters
    pub(crate) const D1: f64 = -12715105075.0 / 11282082432.0;
    pub(crate) const D3: f64 = 87487479700.0 / 32700410799.0;
    pub(crate) const D4: f64 = -10690763975.0 / 1880347072.0;
    pub(crate) const D5: f64 = 701980252875.0 / 199316789632.0;
    pub(crate) const D6: f64 = -1453857185.0 / 822651844.0;
    pub(crate) const D7: f64 = 69997945.0 / 29380423.0;
}
