use num_complex::Complex64;
use rayon::iter::{IntoParallelIterator, ParallelBridge, ParallelIterator};

use crate::{
    linear::{BasisSet, LinOperator, Linear},
    vol::{Volume, VolumeBasisSet, VolumeLinOp},
};

/// Run the DFT algorithm to get the ground state potential of the
/// electron wavefunction.
///
/// Returns the hamiltonian energy eigenvalues.
pub fn dft<'a, const D0: usize, const D1: usize, const D2: usize>(
    n: &mut Volume<f64, D0, D1, D2>,
    n_electrons: usize,
    basis: &'a VolumeBasisSet<Complex64, D0, D1, D2>,
    v_ion: &VolumeLinOp<'a, Complex64, D0, D1, D2>,
    dr: f64,
    max_iters: usize,
) -> (Vec<Volume<Complex64, D0, D1, D2>>, Vec<f64>) {
    let mut last_total_energy = f64::INFINITY;

    // -T
    dbg!("calculating kinetic term");
    let kinetic = VolumeLinOp::new_with(&basis, |e_i| {
        // \nabla^2 e_i
        e_i.nabla_sq(dr)
    }) * Complex64::new(-1.0, 0.0);

    for i in 0..max_iters {
        // V_H
        dbg!("calculating V_H...");
        let hartree_potential = Volume::new_with_dr(dr, |[x0, y0, z0], _| {
            n.volume_integral(dr, |[x1, y1, z1], _, n_r| {
                n_r / (((x0 - x1) * (x0 - x1) + (y0 - y1) * (y0 - y1) + (z0 - z1) * (z0 - z1))
                    .sqrt()
                    .max(dr / 2.0))
            })
            .into()
        });
        let v_hartree = VolumeLinOp::new_with(basis, |e_i| {
            Volume::merge(e_i, &hartree_potential, |a, b| a * b)
        });

        // V_{XC} (Local Density Approximation - LDA)
        dbg!("calculating V_XC...");
        let v_x = n.volume_integral(dr, |_, _, n| n.powf(4.0 / 3.0))
            * (-0.75 * (3.0 / std::f64::consts::PI).powf(1.0 / 3.0));

        let v_c = n.volume_integral(dr, |_, _, n| {
            let r_s = (0.75 / std::f64::consts::PI / n).powf(1.0 / 3.0);
            const A: f64 = 1.0;
            const B: f64 = 1.0;
            const C: f64 = 1.0;
            const D: f64 = 1.0;
            let eta_c = A * r_s.ln() + B + r_s * (C * r_s.ln() + D);

            n * eta_c
        });

        dbg!("calculating H, diagonalizing...");

        // H = -T + V_{ion} + V_H + V_{XC}
        let hamiltonian =
            kinetic.clone() + v_ion.clone() + v_hartree + Complex64::new(v_x + v_c, 0.0);

        // dbg!(hamiltonian.n_matrix_elements());

        let hamiltonian_mat = hamiltonian.to_faer();
        dbg!(hamiltonian_mat.nrows());
        // dbg!("1");
        let eigendecomposition = hamiltonian_mat.selfadjoint_eigendecomposition(faer::Side::Lower);
        // let eigendecomposition: faer::solvers::Eigendecomposition<Complex64> =
        //     hamiltonian_mat.eigendecomposition();
        // dbg!("2");
        let energies = eigendecomposition
            .s()
            .column_vector()
            .try_as_slice()
            .unwrap()
            .re
            .iter()
            .copied()
            .collect::<Vec<_>>();
        // energies.sort(); // puts energies in ascending order

        let total_ground_energy = energies[0..n_electrons].iter().copied().sum();

        let phi = (0..n_electrons)
            .into_par_iter()
            .map(|i| {
                // j
                let phi_i = (0..basis.len())
                    .map(|j| eigendecomposition.u().col(i).get(j))
                    .map(|v| Complex64::new(*v.re, *v.im))
                    .collect::<Vec<_>>();
                basis.from_basis(phi_i)
            })
            .collect::<Vec<_>>();

        dbg!("step done");
        dbg!(total_ground_energy);
        // let psi_0 = basis.from_basis(psi_0);
        *n = Volume::zero();
        for phi_i in &phi {
            *n = n.clone() + phi_i.map_field(|v| v.norm_sqr());
        }

        if f64::abs(last_total_energy - total_ground_energy) < 1.0e-4 {
            // if converged or jumped back up
            dbg!("finished!");
            return (phi, energies);
        }
        if i == max_iters - 1 {
            dbg!("convergence failed!");
            return (phi, energies);
        }
        last_total_energy = total_ground_energy;
    }
    unreachable!();
}

#[derive(Debug, Clone, Copy)]
pub enum IonData {
    Simple { z: i32 },
}

/// Generate the potential energy operator V_ion for the attraction between the electrons and the ions.
pub fn gen_v_ion<'a, const D0: usize, const D1: usize, const D2: usize>(
    ions: &[([f64; 3], IonData)],
    n_world_loops: u32,
    dr: f64,
    basis: &'a VolumeBasisSet<Complex64, D0, D1, D2>,
) -> (f64, VolumeLinOp<'a, Complex64, D0, D1, D2>) {
    let ions_ = {
        let world_dim = [D0 as f64 * dr, D1 as f64 * dr, D2 as f64 * dr];
        let n = n_world_loops as i32;
        (-n..=n)
            .flat_map(move |ix| {
                let x_off = ix as f64 * world_dim[0];
                (-n..=n).flat_map(move |iy| {
                    let y_off = iy as f64 * world_dim[1];
                    (-n..=n).flat_map(move |iz| {
                        let z_off = iz as f64 * world_dim[2];
                        ions.iter()
                            .copied()
                            .map(move |([x, y, z], ion)| ([x + x_off, y + y_off, z + z_off], ion))
                    })
                })
            })
            .collect::<Vec<_>>()
    };
    // dbg!(&ions);
    #[inline]
    fn dist(r0: [f64; 3], r1: [f64; 3]) -> f64 {
        #[inline]
        fn sq(x: f64) -> f64 {
            x * x
        }
        (sq(r0[0] - r1[0]) + sq(r0[1] - r1[1]) + sq(r0[2] - r1[2])).sqrt()
    }
    let electric_potential = Volume::new_with_dr(dr, |r_electron, _| {
        let sum = ions_
            .iter()
            .par_bridge()
            .into_par_iter()
            .map(|(r_ion, ion)| {
                const SUBDIV: u32 = 3;
                const DVK: f64 = 1.0/(SUBDIV*SUBDIV*SUBDIV) as f64;
                let dlk: f64 = dr / SUBDIV as f64;
                match *ion {
                    IonData::Simple { z } => {
                        - (z as f64)
                            // * e^2
                            * (0..SUBDIV*SUBDIV*SUBDIV).map(|k|{
                                let k = [(k%SUBDIV) as f64,((k/SUBDIV)%SUBDIV) as f64, (k/(SUBDIV*SUBDIV)) as f64];

                                DVK / (dist(*r_ion, [r_electron[0]+k[0]*dlk,r_electron[1]+k[1]*dlk,r_electron[2]+k[2]*dlk])
                               .max(dr * (2.0 / SUBDIV as f64)))
                            }).sum::<f64>()
                    }
                }
            })
            .sum();
        Complex64::new(sum, 0.0)

        // let mut sum = Complex64::zero();
        // for ([ion_x, ion_y, ion_z], ion) in ions.iter().copied() {
        //     match ion {
        //         IonData::Simple { z } => {
        //             sum = sum
        //                 - (z as f64)
        //                     // * e^2
        //                     / (((ion_x - elec_x) * (ion_x - elec_x)
        //                     + (ion_y - elec_y) * (ion_y - elec_y)
        //                     + (ion_z - elec_z) * (ion_z - elec_z))
        //                     .sqrt()
        //                     .max(dr / 2.0))
        //         }
        //     }
        // }

        // sum
    });
    let proton_electric_potential_energy: f64 = ions
        .iter()
        .map(|(r0, ion_0)| {
            ions_
                .iter()
                .map(|(r1, ion_1)| {
                    if r0 == r1 {
                        0.0
                    } else {
                        match (*ion_0, *ion_1) {
                            (IonData::Simple { z: z0 }, IonData::Simple { z: z1 }) => {
                                ((z0*z1) as f64)
                            // * e^2
                            / dist(*r0, *r1)
                            }
                        }
                    }
                })
                .sum::<f64>()
        })
        .sum();
    crate::out::fs_write_displayable("./out/electric_potential.volume", &electric_potential);
    // let mut ones = Volume::new_with(|_| Complex64::new(1.0, 0.0));
    // ones = ones.clone() / ones.abs_sq().sqrt().into();
    // let electric_potential =
    //     electric_potential.clone() - ones.clone() * Linear::inner(&ones, &electric_potential);
    (
        proton_electric_potential_energy,
        VolumeLinOp::new_with(basis, |phi_i| {
            // <r|V_ion|phi_i>   V_ion[phi_i(r)]
            electric_potential.merge(phi_i, |a, b| a * b)
        }),
    )
}

#[test]
fn test_make_v_ion() {
    use num_complex::ComplexFloat;

    use crate::linear::Scalar;

    let basis: VolumeBasisSet<Complex64, 5, 5, 5> = VolumeBasisSet::plane_waves([5, 5, 5]);
    let (_v_ion_ion, v_ion) = gen_v_ion(
        &[([0.5, 0.5, 0.5], IonData::Simple { z: 2 })],
        0,
        0.1,
        &basis,
    );
    for (i, e) in basis.iter_basis_vectors().enumerate() {
        let x = basis.to_basis(e.clone());
        let energy = (&v_ion * x.clone())
            .into_iter()
            .zip(x.into_iter())
            .map(|(a, b)| a.conj() * b)
            .sum::<Complex64>();
        assert!(energy.im().abs() < 1.0e-10);
        dbg!((i, energy.re()));
    }
    {
        let mut x = vec![Complex64::zero(); basis.len()];
        x[3] = 0.5.sqrt().into();
        x[4] = 0.5.sqrt().into();
        let energy = (&v_ion * x.clone())
            .into_iter()
            .zip(x.into_iter())
            .map(|(a, b)| a.conj() * b)
            .sum::<Complex64>();
        assert!(energy.im().abs() < 1.0e-10);
        dbg!(energy.re());
    }
}
