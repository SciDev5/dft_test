use std::{
    ops::{Add, Div, Index, Mul, Sub},
    rc::Rc,
};

use crate::linear::{orthonormalize_basis, Linear, Scalar};

#[derive(Debug, Clone)]
pub struct Volume<T: Scalar, const DIM0: usize, const DIM1: usize, const DIM2: usize> {
    field: Rc<[T]>,
}
impl<T: Scalar, const DIM0: usize, const DIM1: usize, const DIM2: usize>
    Volume<T, DIM0, DIM1, DIM2>
{
    pub fn new() -> Self {
        Self {
            field: vec![T::zero(); DIM0 * DIM1 * DIM2].into(),
        }
    }
    pub fn new_with<F: Fn([usize; 3]) -> T>(f: F) -> Self {
        let mut field = vec![T::zero(); DIM0 * DIM1 * DIM2];
        for i0 in 0..DIM0 {
            for i1 in 0..DIM1 {
                for i2 in 0..DIM2 {
                    field[Self::convert_index([i0, i1, i2])] = f([i0, i1, i2]);
                }
            }
        }
        Self {
            field: field.into(),
        }
    }
    #[inline]
    fn convert_index([i0, i1, i2]: [usize; 3]) -> usize {
        debug_assert!(i0 < DIM0 && i1 < DIM1 && i2 < DIM2);
        i0 + i1 * DIM0 + i2 * DIM0 * DIM1
    }
}
impl<T: Scalar, const DIM0: usize, const DIM1: usize, const DIM2: usize> Index<[usize; 3]>
    for Volume<T, DIM0, DIM1, DIM2>
{
    type Output = T;
    fn index(&self, index: [usize; 3]) -> &Self::Output {
        &self.field[Self::convert_index(index)]
    }
}

macro_rules! volume_op {
    (
        $trait_name: ident ;
        $fun: ident;
        < $t: ident >
        | $rhs_i: ident : $rhs_t: ident, $i: ident |
        $rhs_eval: expr
    ) => {
        impl<$t: Scalar, const DIM0: usize, const DIM1: usize, const DIM2: usize>
            $trait_name<$rhs_t> for Volume<$t, DIM0, DIM1, DIM2>
        {
            type Output = Self;

            fn $fun(self, $rhs_i: $rhs_t) -> Self::Output {
                let mut field = self.field.iter().copied().collect::<Vec<_>>();

                for $i in 0..field.len() {
                    field[$i] = <$t as $trait_name>::$fun(field[$i], $rhs_eval);
                }

                Self {
                    field: field.into(),
                }
            }
        }
    };
}
volume_op!(Add; add; <T>|rhs: Self, i| rhs.field[i]);
volume_op!(Sub; sub; <T>|rhs: Self, i| rhs.field[i]);
volume_op!(Mul; mul; <T>|rhs: T, i| rhs);
volume_op!(Div; div; <T>|rhs: T, i| rhs);

impl<T: Scalar, const DIM0: usize, const DIM1: usize, const DIM2: usize> Linear<T>
    for Volume<T, DIM0, DIM1, DIM2>
{
    fn zero() -> Self {
        Self::new()
    }

    fn inner(bra: &Self, ket: &Self) -> T {
        let mut x = T::zero();
        for i in 0..bra.field.len() {
            x = x + bra.field[i].hc() * ket.field[i];
        }
        x
    }
}

#[derive(Debug, Clone)]
pub struct VolumeBasisSet<S: Scalar, const DIM0: usize, const DIM1: usize, const DIM2: usize> {
    basis: Vec<Volume<S, DIM0, DIM1, DIM2>>,
}
impl<S: Scalar, const DIM0: usize, const DIM1: usize, const DIM2: usize>
    VolumeBasisSet<S, DIM0, DIM1, DIM2>
{
    pub fn plane_waves(count: [usize; 3]) -> Self {
        let mut basis: Vec<Volume<S, DIM0, DIM1, DIM2>> =
            Vec::with_capacity(count.into_iter().product());
        macro_rules! zero_centered_range {
            ($len: expr) => {{
                let len = $len;
                -(len as isize / 2)..(len - len / 2) as isize
            }};
        }
        for i0 in zero_centered_range!(count[0]) {
            for i1 in zero_centered_range!(count[1]) {
                for i2 in zero_centered_range!(count[2]) {
                    basis.push(Volume::new_with(|[j0, j1, j2]| {
                        S::into_imag(
                            (
                                j0 as f64 / DIM0 as f64 * i0 as f64
                                    + j1 as f64 / DIM1 as f64 * i1 as f64
                                    + j2 as f64 / DIM2 as f64 * i2 as f64
                                //* s2
                            ) * std::f64::consts::TAU,
                        )
                        .exp()
                    }))
                }
            }
        }
        orthonormalize_basis(&mut basis);
        Self { basis }
    }

    pub fn volume_to_basis(&self, v: Volume<S, DIM0, DIM1, DIM2>) -> Vec<S> {
        self.basis
            .iter()
            .map(|e_i| Linear::inner(e_i, &v))
            .collect()
    }
    pub fn basis_to_volume(&self, v: Vec<S>) -> Volume<S, DIM0, DIM1, DIM2> {
        debug_assert_eq!(v.len(), self.basis.len());
        let mut v_out = Volume::<S, DIM0, DIM1, DIM2>::zero();

        for i in 0..self.basis.len() {
            v_out = v_out + self.basis[i].clone() * v[i];
        }

        v_out
    }
}

#[cfg(test)]
mod test {
    use num_complex::Complex64;

    use crate::{
        linear::Linear,
        vol::{Volume, VolumeBasisSet},
    };

    #[test]
    fn test_volume_basis_set() {
        let basis = VolumeBasisSet::<Complex64, 4, 1, 3>::plane_waves([4, 1, 3]);
        let v0 = Volume::new_with(|[x, _, _]| {
            Complex64::new((x as f64 / 2.0 * std::f64::consts::PI).sin(), 0.0)
        });
        assert!(
            (basis.basis_to_volume(basis.volume_to_basis(v0.clone())) - v0.clone()).abs_sq()
                < 1.0e-10
        );
    }
}
