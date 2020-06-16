use super::Radau5;

// Declare all the constants needed for Radau5
impl Radau5 {
    pub(crate) const T11: f64 = 9.1232394870892942792e-02;
    pub(crate) const T12: f64 = -0.14125529502095420843;
    pub(crate) const T13: f64 = -3.0029194105147424492e-02;
    pub(crate) const T21: f64 = 0.24171793270710701896;
    pub(crate) const T22: f64 = 0.20412935229379993199;
    pub(crate) const T23: f64 = 0.38294211275726193779;
    pub(crate) const T31: f64 = 0.96604818261509293619;
    pub(crate) const TI11: f64 = 4.3255798900631553510;
    pub(crate) const TI12: f64 = 0.33919925181580986954;
    pub(crate) const TI13: f64 = 0.54177053993587487119;
    pub(crate) const TI21: f64 = -4.1787185915519047273;
    pub(crate) const TI22: f64 = -0.32768282076106238708;
    pub(crate) const TI23: f64 = 0.47662355450055045196;
    pub(crate) const TI31: f64 = -0.50287263494578687595;
    pub(crate) const TI32: f64 = 2.5719269498556054292;
    pub(crate) const TI33: f64 = -0.59603920482822492497;
    pub(crate) const C1: f64 = 0.1550510257216822;
    pub(crate) const C2: f64 = 0.6449489742783178;
    pub(crate) const C1M1: f64 = -0.8449489742783178;
    pub(crate) const C2M1: f64 = -0.3550510257216822;
    pub(crate) const C1MC2: f64 = -0.4898979485566356;
    pub(crate) const ALPHA: f64 = 2.681082873627752;
    pub(crate) const BETA: f64 = 3.050430199247411;
    pub(crate) const U1: f64 = 3.637834252744496;
}
