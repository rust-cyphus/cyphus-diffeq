pub(crate) mod stages;
use ndarray::prelude::*;

pub(crate) struct DormandPrince8Cache {
    n_stiff: usize,
    n_non_stiff: usize,
    reject: bool,
    last: bool,
    k2: Array1<f64>,
    k3: Array1<f64>,
    k4: Array1<f64>,
    k5: Array1<f64>,
    k6: Array1<f64>,
    k7: Array1<f64>,
    k8: Array1<f64>,
    k9: Array1<f64>,
    k10: Array1<f64>,
    rcont1: Array1<f64>,
    rcont2: Array1<f64>,
    rcont3: Array1<f64>,
    rcont4: Array1<f64>,
    rcont5: Array1<f64>,
    rcont6: Array1<f64>,
    rcont7: Array1<f64>,
    rcont8: Array1<f64>,
    unew: Array1<f64>,
    du: Array1<f64>,
    dunew: Array1<f64>,
    uerr: Array1<f64>,
    ustiff: Array1<f64>,
}

pub(crate) struct DormandPrince8 {}

impl DormandPrince8 {
    // Butcher c_i's
    pub(crate) const c2: f64 = 0.526001519587677318785587544488E-01;
    pub(crate) const c3: f64 = 0.789002279381515978178381316732E-01;
    pub(crate) const c4: f64 = 0.118350341907227396726757197510E+00;
    pub(crate) const c5: f64 = 0.281649658092772603273242802490E+00;
    pub(crate) const c6: f64 = 0.333333333333333333333333333333E+00;
    pub(crate) const c7: f64 = 0.25E+00;
    pub(crate) const c8: f64 = 0.307692307692307692307692307692E+00;
    pub(crate) const c9: f64 = 0.651282051282051282051282051282E+00;
    pub(crate) const c10: f64 = 0.6E+00;
    pub(crate) const c11: f64 = 0.857142857142857142857142857142E+00;
    pub(crate) const c14: f64 = 0.1E+00;
    pub(crate) const c15: f64 = 0.2E+00;
    pub(crate) const c16: f64 = 0.777777777777777777777777777778E+00;
    // Butcher b_i's
    pub(crate) const b1: f64 = 5.42937341165687622380535766363E-2;
    pub(crate) const b6: f64 = 4.45031289275240888144113950566E0;
    pub(crate) const b7: f64 = 1.89151789931450038304281599044E0;
    pub(crate) const b8: f64 = -5.8012039600105847814672114227E0;
    pub(crate) const b9: f64 = 3.1116436695781989440891606237E-1;
    pub(crate) const b10: f64 = -1.52160949662516078556178806805E-1;
    pub(crate) const b11: f64 = 2.01365400804030348374776537501E-1;
    pub(crate) const b12: f64 = 4.47106157277725905176885569043E-2;
    pub(crate) const bhh1: f64 = 0.244094488188976377952755905512E+00;
    pub(crate) const bhh2: f64 = 0.733846688281611857341361741547E+00;
    pub(crate) const bhh3: f64 = 0.220588235294117647058823529412E-01;
    // Butcher a_ij's
    pub(crate) const a21: f64 = 5.26001519587677318785587544488E-2;
    pub(crate) const a31: f64 = 1.97250569845378994544595329183E-2;
    pub(crate) const a32: f64 = 5.91751709536136983633785987549E-2;
    pub(crate) const a41: f64 = 2.95875854768068491816892993775E-2;
    pub(crate) const a43: f64 = 8.87627564304205475450678981324E-2;
    pub(crate) const a51: f64 = 2.41365134159266685502369798665E-1;
    pub(crate) const a53: f64 = -8.84549479328286085344864962717E-1;
    pub(crate) const a54: f64 = 9.24834003261792003115737966543E-1;
    pub(crate) const a61: f64 = 3.7037037037037037037037037037E-2;
    pub(crate) const a64: f64 = 1.70828608729473871279604482173E-1;
    pub(crate) const a65: f64 = 1.25467687566822425016691814123E-1;
    pub(crate) const a71: f64 = 3.7109375E-2;
    pub(crate) const a74: f64 = 1.70252211019544039314978060272E-1;
    pub(crate) const a75: f64 = 6.02165389804559606850219397283E-2;
    pub(crate) const a76: f64 = -1.7578125E-2;
    pub(crate) const a81: f64 = 3.70920001185047927108779319836E-2;
    pub(crate) const a84: f64 = 1.70383925712239993810214054705E-1;
    pub(crate) const a85: f64 = 1.07262030446373284651809199168E-1;
    pub(crate) const a86: f64 = -1.53194377486244017527936158236E-2;
    pub(crate) const a87: f64 = 8.27378916381402288758473766002E-3;
    pub(crate) const a91: f64 = 6.24110958716075717114429577812E-1;
    pub(crate) const a94: f64 = -3.36089262944694129406857109825E0;
    pub(crate) const a95: f64 = -8.68219346841726006818189891453E-1;
    pub(crate) const a96: f64 = 2.75920996994467083049415600797E1;
    pub(crate) const a97: f64 = 2.01540675504778934086186788979E1;
    pub(crate) const a98: f64 = -4.34898841810699588477366255144E1;
    pub(crate) const a101: f64 = 4.77662536438264365890433908527E-1;
    pub(crate) const a104: f64 = -2.48811461997166764192642586468E0;
    pub(crate) const a105: f64 = -5.90290826836842996371446475743E-1;
    pub(crate) const a106: f64 = 2.12300514481811942347288949897E1;
    pub(crate) const a107: f64 = 1.52792336328824235832596922938E1;
    pub(crate) const a108: f64 = -3.32882109689848629194453265587E1;
    pub(crate) const a109: f64 = -2.03312017085086261358222928593E-2;
    pub(crate) const a111: f64 = -9.3714243008598732571704021658E-1;
    pub(crate) const a114: f64 = 5.18637242884406370830023853209E0;
    pub(crate) const a115: f64 = 1.09143734899672957818500254654E0;
    pub(crate) const a116: f64 = -8.14978701074692612513997267357E0;
    pub(crate) const a117: f64 = -1.85200656599969598641566180701E1;
    pub(crate) const a118: f64 = 2.27394870993505042818970056734E1;
    pub(crate) const a119: f64 = 2.49360555267965238987089396762E0;
    pub(crate) const a1110: f64 = -3.0467644718982195003823669022E0;
    pub(crate) const a121: f64 = 2.27331014751653820792359768449E0;
    pub(crate) const a124: f64 = -1.05344954667372501984066689879E1;
    pub(crate) const a125: f64 = -2.00087205822486249909675718444E0;
    pub(crate) const a126: f64 = -1.79589318631187989172765950534E1;
    pub(crate) const a127: f64 = 2.79488845294199600508499808837E1;
    pub(crate) const a128: f64 = -2.85899827713502369474065508674E0;
    pub(crate) const a129: f64 = -8.87285693353062954433549289258E0;
    pub(crate) const a1210: f64 = 1.23605671757943030647266201528E1;
    pub(crate) const a1211: f64 = 6.43392746015763530355970484046E-1;
    pub(crate) const a141: f64 = 5.61675022830479523392909219681E-2;
    pub(crate) const a147: f64 = 2.53500210216624811088794765333E-1;
    pub(crate) const a148: f64 = -2.46239037470802489917441475441E-1;
    pub(crate) const a149: f64 = -1.24191423263816360469010140626E-1;
    pub(crate) const a1410: f64 = 1.5329179827876569731206322685E-1;
    pub(crate) const a1411: f64 = 8.20105229563468988491666602057E-3;
    pub(crate) const a1412: f64 = 7.56789766054569976138603589584E-3;
    pub(crate) const a1413: f64 = -8.298E-3;
    pub(crate) const a151: f64 = 3.18346481635021405060768473261E-2;
    pub(crate) const a156: f64 = 2.83009096723667755288322961402E-2;
    pub(crate) const a157: f64 = 5.35419883074385676223797384372E-2;
    pub(crate) const a158: f64 = -5.49237485713909884646569340306E-2;
    pub(crate) const a1511: f64 = -1.08347328697249322858509316994E-4;
    pub(crate) const a1512: f64 = 3.82571090835658412954920192323E-4;
    pub(crate) const a1513: f64 = -3.40465008687404560802977114492E-4;
    pub(crate) const a1514: f64 = 1.41312443674632500278074618366E-1;
    pub(crate) const a161: f64 = -4.28896301583791923408573538692E-1;
    pub(crate) const a166: f64 = -4.69762141536116384314449447206E0;
    pub(crate) const a167: f64 = 7.68342119606259904184240953878E0;
    pub(crate) const a168: f64 = 4.06898981839711007970213554331E0;
    pub(crate) const a169: f64 = 3.56727187455281109270669543021E-1;
    pub(crate) const a1613: f64 = -1.39902416515901462129418009734E-3;
    pub(crate) const a1614: f64 = 2.9475147891527723389556272149E0;
    pub(crate) const a1615: f64 = -9.15095847217987001081870187138E0;
    // Error estimation constants
    pub(crate) const er1: f64 = 0.1312004499419488073250102996E-01;
    pub(crate) const er6: f64 = -0.1225156446376204440720569753E+01;
    pub(crate) const er7: f64 = -0.4957589496572501915214079952E+00;
    pub(crate) const er8: f64 = 0.1664377182454986536961530415E+01;
    pub(crate) const er9: f64 = -0.3503288487499736816886487290E+00;
    pub(crate) const er10: f64 = 0.3341791187130174790297318841E+00;
    pub(crate) const er11: f64 = 0.8192320648511571246570742613E-01;
    pub(crate) const er12: f64 = -0.2235530786388629525884427845E-01;
    // Continuous output parameters
    pub(crate) const d41: f64 = -0.84289382761090128651353491142E+01;
    pub(crate) const d46: f64 = 0.56671495351937776962531783590E+00;
    pub(crate) const d47: f64 = -0.30689499459498916912797304727E+01;
    pub(crate) const d48: f64 = 0.23846676565120698287728149680E+01;
    pub(crate) const d49: f64 = 0.21170345824450282767155149946E+01;
    pub(crate) const d410: f64 = -0.87139158377797299206789907490E+00;
    pub(crate) const d411: f64 = 0.22404374302607882758541771650E+01;
    pub(crate) const d412: f64 = 0.63157877876946881815570249290E+00;
    pub(crate) const d413: f64 = -0.88990336451333310820698117400E-01;
    pub(crate) const d414: f64 = 0.18148505520854727256656404962E+02;
    pub(crate) const d415: f64 = -0.91946323924783554000451984436E+01;
    pub(crate) const d416: f64 = -0.44360363875948939664310572000E+01;
    pub(crate) const d51: f64 = 0.10427508642579134603413151009E+02;
    pub(crate) const d56: f64 = 0.24228349177525818288430175319E+03;
    pub(crate) const d57: f64 = 0.16520045171727028198505394887E+03;
    pub(crate) const d58: f64 = -0.37454675472269020279518312152E+03;
    pub(crate) const d59: f64 = -0.22113666853125306036270938578E+02;
    pub(crate) const d510: f64 = 0.77334326684722638389603898808E+01;
    pub(crate) const d511: f64 = -0.30674084731089398182061213626E+02;
    pub(crate) const d512: f64 = -0.93321305264302278729567221706E+01;
    pub(crate) const d513: f64 = 0.15697238121770843886131091075E+02;
    pub(crate) const d514: f64 = -0.31139403219565177677282850411E+02;
    pub(crate) const d515: f64 = -0.93529243588444783865713862664E+01;
    pub(crate) const d516: f64 = 0.35816841486394083752465898540E+02;
    pub(crate) const d61: f64 = 0.19985053242002433820987653617E+02;
    pub(crate) const d66: f64 = -0.38703730874935176555105901742E+03;
    pub(crate) const d67: f64 = -0.18917813819516756882830838328E+03;
    pub(crate) const d68: f64 = 0.52780815920542364900561016686E+03;
    pub(crate) const d69: f64 = -0.11573902539959630126141871134E+02;
    pub(crate) const d610: f64 = 0.68812326946963000169666922661E+01;
    pub(crate) const d611: f64 = -0.10006050966910838403183860980E+01;
    pub(crate) const d612: f64 = 0.77771377980534432092869265740E+00;
    pub(crate) const d613: f64 = -0.27782057523535084065932004339E+01;
    pub(crate) const d614: f64 = -0.60196695231264120758267380846E+02;
    pub(crate) const d615: f64 = 0.84320405506677161018159903784E+02;
    pub(crate) const d616: f64 = 0.11992291136182789328035130030E+02;
    pub(crate) const d71: f64 = -0.25693933462703749003312586129E+02;
    pub(crate) const d76: f64 = -0.15418974869023643374053993627E+03;
    pub(crate) const d77: f64 = -0.23152937917604549567536039109E+03;
    pub(crate) const d78: f64 = 0.35763911791061412378285349910E+03;
    pub(crate) const d79: f64 = 0.93405324183624310003907691704E+02;
    pub(crate) const d710: f64 = -0.37458323136451633156875139351E+02;
    pub(crate) const d711: f64 = 0.10409964950896230045147246184E+03;
    pub(crate) const d712: f64 = 0.29840293426660503123344363579E+02;
    pub(crate) const d713: f64 = -0.43533456590011143754432175058E+02;
    pub(crate) const d714: f64 = 0.96324553959188282948394950600E+02;
    pub(crate) const d715: f64 = -0.39177261675615439165231486172E+02;
    pub(crate) const d716: f64 = -0.14972683625798562581422125276E+03;
}