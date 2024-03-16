use dft::{dft, gen_v_ion};
use num_complex::Complex64;
use vol::{Volume, VolumeBasisSet};

use crate::dft::IonData;

pub mod dft;
pub mod linear;
pub mod vol;

fn main() {
    let dr = 1.0;
    let basis: VolumeBasisSet<Complex64, 4, 4, 4> = VolumeBasisSet::plane_waves([2, 2, 2]);
    let v_ion = gen_v_ion(
        &[([2.0, 2.0, 2.0], IonData::Simple { z: 1 })],
        0,
        dr,
        &basis,
    );
    let mut n = Volume::new_from_total_density(1.0);
    let (_phi_0, spectrum) = dft(&mut n, &basis, &v_ion, dr, 100);
    dbg!(n, spectrum);
}
