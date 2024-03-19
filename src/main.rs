use std::thread;

use dft::{dft, gen_v_ion};
use num_complex::Complex64;
use vol::{Volume, VolumeBasisSet};

use crate::dft::IonData;

pub mod dft;
pub mod linear;
pub mod vol;

fn main() {
    thread::Builder::new()
        .stack_size(1 << 30)
        .name(format!("cool thread"))
        .spawn(|| {
            let dr = 1.0;
            let basis: VolumeBasisSet<Complex64, 10, 10, 10> =
                VolumeBasisSet::plane_waves([5, 5, 5]);
            let v_ion = gen_v_ion(
                &[([2.0, 2.0, 2.0], IonData::Simple { z: 1 })],
                0,
                dr,
                &basis,
            );
            let mut n = Volume::new_from_total_density(1.0);
            let (_phi_0, spectrum) = dft(&mut n, &basis, &v_ion, dr, 100);
            dbg!(n, spectrum);
        })
        .unwrap()
        .join()
        .unwrap();
}
