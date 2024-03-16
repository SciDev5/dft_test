use num_complex::Complex64;

use crate::{
    linear::{BasisSet, LinOperator, Linear, Scalar},
    vol::{Volume, VolumeBasisSet, VolumeLinOp},
};

/// Run the DFT algorithm to get the ground state potential of the
/// electron wavefunction.
///
/// Returns the hamiltonian energy eigenvalues.
pub fn dft<'a, const D0: usize, const D1: usize, const D2: usize>(
    n: &mut Volume<f64, D0, D1, D2>,
    basis: &'a VolumeBasisSet<Complex64, D0, D1, D2>,
    v_ion: &VolumeLinOp<'a, Complex64, D0, D1, D2>,
    dr: f64,
    max_iters: usize,
) -> (Volume<Complex64, D0, D1, D2>, Vec<f64>) {
    let mut last_ground_energy = f64::INFINITY;

    for i in 0..max_iters {
        let v_hartree = VolumeLinOp::new_with(basis, |phi_i| {
            let hartree_potential = Volume::new_with_dr(dr, |[x0, y0, z0], _| {
                n.volume_integral(dr, |[x1, y1, z1], _, n_r| {
                    n_r / (((x0 - x1) * (x0 - x1) + (y0 - y1) * (y0 - y1) + (z0 - z1) * (z0 - z1))
                        .sqrt()
                        .max(dr / 2.0))
                })
                .into()
            });
            Volume::merge(phi_i, &hartree_potential, |a, b| a * b)
        });
        let v_xc = VolumeLinOp::new_with(basis, |_| Volume::zero());
        let kinetic = VolumeLinOp::new_with(&basis, |phi_i| {
            // \nabla^2 e_i
            phi_i.nabla_sq(dr)
        }) * Complex64::new(-1.0, 0.0);

        let hamiltonian = kinetic + v_ion.clone() + v_hartree + v_xc;

        // dbg!(hamiltonian.n_matrix_elements());

        let hamiltonian_mat = hamiltonian.to_faer();
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
            .re;
        let (psi_0_index, ground_energy) = energies
            .iter()
            .copied()
            .enumerate()
            .min_by(|(_, a), (_, b)| a.partial_cmp(b).expect("i hate floating points sometimes"))
            .unwrap();

        let psi_0 = (0..basis.len())
            .map(|i| eigendecomposition.u().col(psi_0_index).get(i))
            .map(|v| Complex64::new(*v.re, *v.im))
            .collect::<Vec<_>>();

        dbg!(ground_energy);
        let psi_0 = basis.from_basis(psi_0);
        *n = psi_0.map_field(|v| v.norm_sqr());

        if (last_ground_energy - ground_energy).abs() < 1.0e-4 {
            dbg!("finished!");
            return (psi_0, energies.iter().copied().collect());
        }
        if i == max_iters - 1 {
            dbg!("convergence failed!");
            return (psi_0, energies.iter().copied().collect());
        }
        last_ground_energy = ground_energy;
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
) -> VolumeLinOp<'a, Complex64, D0, D1, D2> {
    let ions = {
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
    let electric_potential = Volume::new_with_dr(dr, |[elec_x, elec_y, elec_z], _| {
        let mut sum = Complex64::zero();

        for ([ion_x, ion_y, ion_z], ion) in ions.iter().copied() {
            match ion {
                IonData::Simple { z } => {
                    sum = sum
                        - (z as f64)
                            // * e^2
                            / (((ion_x - elec_x) * (ion_x - elec_x)
                            + (ion_y - elec_y) * (ion_y - elec_y)
                            + (ion_z - elec_z) * (ion_z - elec_z))
                            .sqrt()
                            .max(dr / 2.0))
                }
            }
        }

        sum
    });
    // let mut ones = Volume::new_with(|_| Complex64::new(1.0, 0.0));
    // ones = ones.clone() / ones.abs_sq().sqrt().into();
    // let electric_potential =
    //     electric_potential.clone() - ones.clone() * Linear::inner(&ones, &electric_potential);
    VolumeLinOp::new_with(basis, |phi_i| {
        // <r|V_ion|phi_i>   V_ion[phi_i(r)]
        electric_potential.merge(phi_i, |a, b| a * b)
    })
}

#[test]
fn test_make_v_ion() {
    use num_complex::ComplexFloat;

    let basis: VolumeBasisSet<Complex64, 5, 5, 5> = VolumeBasisSet::plane_waves([5, 5, 5]);
    let v_ion = gen_v_ion(
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
