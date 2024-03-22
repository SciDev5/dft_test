use std::thread;

use dft::{dft, gen_v_ion};
use num_complex::Complex64;
use vol::{Volume, VolumeBasisSet};

use crate::{
    dft::IonData,
    out::{fs_write_displayable, Spectra},
};

pub mod dft;
pub mod linear;
pub mod out;
pub mod vol;

fn main() {
    thread::Builder::new()
        .stack_size(0x1000_0000_0000)
        .name(format!("dft thread"))
        .spawn(|| {
            dbg!("dft thread spawned");

            const D: usize = 32;
            const N_ELECTRONS: usize = 2;
            let box_size = 8.0;

            let dr = box_size / D as f64;

            dbg!("building basis");
            let basis: VolumeBasisSet<Complex64, D, 16, 16> =
                VolumeBasisSet::plane_waves([8, 5, 5]);

            const TEST_COUNT: usize = 10;
            for i in 1..=TEST_COUNT {
                let sep = (i as f64 / TEST_COUNT as f64) * 1.5 + 0.5;
                dbg!(sep);

                dbg!("building V_ion");
                let (v_ion_ion, v_ion) = gen_v_ion(
                    &[
                        ([4.0 - sep / 2.0, 2.0, 2.0], IonData::Simple { z: 1 }),
                        ([4.0 + sep / 2.0, 2.0, 2.0], IonData::Simple { z: 1 }),
                    ],
                    0,
                    dr,
                    &basis,
                );

                dbg!("entering DFT loop");
                let mut n = Volume::new_from_total_density(N_ELECTRONS as f64);
                let (_phi, spectrum) = dft(&mut n, N_ELECTRONS, &basis, &v_ion, dr, 10);

                dbg!("writing data");
                fs_write_displayable(format!("./out/h2_exp/{i}.volume"), &n);
                fs_write_displayable(
                    format!("./out/h2_exp/{i}.spectra"),
                    &Spectra(v_ion_ion, &spectrum[..]),
                );

                // dbg!(n, spectrum);
            }
        })
        .unwrap()
        .join()
        .unwrap();
}
