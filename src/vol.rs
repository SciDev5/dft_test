use std::{
    ops::{Add, Div, Index, IndexMut, Mul, Sub},
    rc::Rc,
};

use num_complex::Complex64;

use crate::linear::{orthonormalize_basis, BasisSet, LinOperator, Linear, Scalar};

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
    pub fn new_from_total_density(n: f64) -> Self {
        let n_dr = T::new_re(n / ((DIM0 * DIM1 * DIM2) as f64).sqrt());
        Self {
            field: vec![n_dr; DIM0 * DIM1 * DIM2].into(),
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
    pub fn new_with_dr<F: Fn([f64; 3], [usize; 3]) -> T>(dr: f64, f: F) -> Self {
        let mut field = vec![T::zero(); DIM0 * DIM1 * DIM2];
        for i0 in 0..DIM0 {
            for i1 in 0..DIM1 {
                for i2 in 0..DIM2 {
                    field[Self::convert_index([i0, i1, i2])] = f(
                        [i0 as f64 * dr, i1 as f64 * dr, i2 as f64 * dr],
                        [i0, i1, i2],
                    );
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

    fn index_to_r(&self, i: [usize; 3], dr: f64) -> [f64; 3] {
        #[inline]
        fn conv(i: usize, dr: f64) -> f64 {
            (i as f64) * dr
        }
        [conv(i[0], dr), conv(i[1], dr), conv(i[2], dr)]
    }

    pub fn volume_integral<F: Fn([f64; 3], [usize; 3], T) -> f64>(&self, dr: f64, f: F) -> f64 {
        let mut accum = 0.0;
        let dv = dr * dr * dr;
        for i0 in 0..DIM0 {
            for i1 in 0..DIM1 {
                for i2 in 0..DIM2 {
                    let r = self.index_to_r([i0, i1, i2], dr);
                    accum += f(r, [i0, i1, i2], self[[i0, i1, i2]]) * dv
                }
            }
        }
        accum
    }
    pub fn nabla_sq(&self, dr: f64) -> Self {
        let k = T::new_re(1.0 / (dr * dr));
        Self {
            field: (0..DIM2)
                .flat_map(move |i2| {
                    (0..DIM1).flat_map(move |i1| {
                        (0..DIM0).map(move |i0| {
                            let current = self[[i0, i1, i2]] * T::new_re(2.0);
                            let dx = (self[[(i0 + 1) % DIM0, i1, i2]]
                                + self[[(i0 + DIM0 - 1) % DIM0, i1, i2]]
                                - current)
                                * k;
                            let dy = (self[[i0, (i1 + 1) % DIM1, i2]]
                                + self[[i0, (i1 + DIM1 - 1) % DIM1, i2]]
                                - current)
                                * k;
                            let dz = (self[[i0, i1, (i2 + 1) % DIM2]]
                                + self[[i0, i1, (i2 + DIM2 - 1) % DIM2]]
                                - current)
                                * k;

                            dx * dx + dy * dy + dz * dz
                        })
                    })
                })
                .collect(),
        }
    }
    pub fn map_field<F: Fn(T) -> S, S: Scalar>(&self, f: F) -> Volume<S, DIM0, DIM1, DIM2> {
        Volume {
            field: self.field.iter().copied().map(f).collect(),
        }
    }
    pub fn merge<F: Fn(T, T) -> T>(&self, rhs: &Self, f: F) -> Self {
        Volume {
            field: self
                .field
                .iter()
                .copied()
                .enumerate()
                .map(|(i, v)| f(v, rhs.field[i]))
                .collect(),
        }
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
    basis_vectors: Vec<Volume<S, DIM0, DIM1, DIM2>>,
}
impl<S: Scalar, const DIM0: usize, const DIM1: usize, const DIM2: usize>
    BasisSet<Volume<S, DIM0, DIM1, DIM2>, S> for VolumeBasisSet<S, DIM0, DIM1, DIM2>
{
    #[allow(deprecated_where_clause_location)] // allowing this because fixing it breaks the autoformatter.
    type BasisVectorIterator<'a>
    where
        Self: 'a,
        Volume<S, DIM0, DIM1, DIM2>: 'a,
    = std::slice::Iter<'a, Volume<S, DIM0, DIM1, DIM2>>;

    fn len(&self) -> usize {
        self.basis_vectors.len()
    }

    fn iter_basis_vectors<'a>(&'a self) -> Self::BasisVectorIterator<'a> {
        self.basis_vectors.iter()
    }
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
                        S::new_im(
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
        Self {
            basis_vectors: basis,
        }
    }
}

#[derive(Debug, Clone)]
pub struct VolumeLinOp<'a, S: Scalar, const D0: usize, const D1: usize, const D2: usize> {
    basis: &'a VolumeBasisSet<S, D0, D1, D2>,
    matrix_elements: Vec<S>,
    len: usize,
}
impl<'a, S: Scalar, const D0: usize, const D1: usize, const D2: usize>
    VolumeLinOp<'a, S, D0, D1, D2>
{
    pub fn n_matrix_elements(&self) -> usize {
        self.matrix_elements.len()
    }
    pub fn new_with<F: Fn(&Volume<S, D0, D1, D2>) -> Volume<S, D0, D1, D2>>(
        basis: &'a VolumeBasisSet<S, D0, D1, D2>,
        f: F,
    ) -> Self {
        let mut matrix_elements = Vec::with_capacity(basis.len());
        for i1 in 0..basis.len() {
            let ket = f(&basis.basis_vectors[i1]);
            for i0 in 0..basis.len() {
                matrix_elements.push(Linear::inner(&basis.basis_vectors[i0], &ket));
            }
        }
        Self {
            basis,
            matrix_elements,
            len: basis.len(),
        }
    }
}
impl<'a, S: Scalar, const D0: usize, const D1: usize, const D2: usize> Index<[usize; 2]>
    for VolumeLinOp<'a, S, D0, D1, D2>
{
    type Output = S;
    fn index(&self, index: [usize; 2]) -> &Self::Output {
        &self.matrix_elements[index[0] + self.len * index[1]]
    }
}
impl<'a, S: Scalar, const D0: usize, const D1: usize, const D2: usize> IndexMut<[usize; 2]>
    for VolumeLinOp<'a, S, D0, D1, D2>
{
    fn index_mut(&mut self, index: [usize; 2]) -> &mut Self::Output {
        &mut self.matrix_elements[index[0] + self.len * index[1]]
    }
}
impl<'a, 'b, S: Scalar, const D0: usize, const D1: usize, const D2: usize> Mul<Vec<S>>
    for &'b VolumeLinOp<'a, S, D0, D1, D2>
{
    type Output = Vec<S>;
    fn mul(self, rhs: Vec<S>) -> Self::Output {
        assert_eq!(self.len, rhs.len());
        let mut out = vec![S::zero(); self.len];
        for i in 0..self.len {
            for j in 0..self.len {
                out[i] = out[i] + self[[i, j]] * rhs[j];
            }
        }
        out
    }
}
impl<'a, S: Scalar, const D0: usize, const D1: usize, const D2: usize> Add
    for VolumeLinOp<'a, S, D0, D1, D2>
{
    type Output = Self;
    fn add(mut self, rhs: Self) -> Self::Output {
        assert_eq!(
            self.basis as *const VolumeBasisSet<S, D0, D1, D2>,
            rhs.basis as *const VolumeBasisSet<S, D0, D1, D2>
        );
        for ij in 0..self.matrix_elements.len() {
            let t_ij = &mut self.matrix_elements[ij];
            *t_ij = *t_ij + rhs.matrix_elements[ij];
        }
        self
    }
}
impl<'a, S: Scalar, const D0: usize, const D1: usize, const D2: usize> Sub
    for VolumeLinOp<'a, S, D0, D1, D2>
{
    type Output = Self;
    fn sub(mut self, rhs: Self) -> Self::Output {
        assert_eq!(
            self.basis as *const VolumeBasisSet<S, D0, D1, D2>,
            rhs.basis as *const VolumeBasisSet<S, D0, D1, D2>
        );
        for ij in 0..self.matrix_elements.len() {
            let t_ij = &mut self.matrix_elements[ij];
            *t_ij = *t_ij - rhs.matrix_elements[ij];
        }
        self
    }
}
impl<'a, S: Scalar, const D0: usize, const D1: usize, const D2: usize> Mul<S>
    for VolumeLinOp<'a, S, D0, D1, D2>
{
    type Output = Self;
    fn mul(mut self, rhs: S) -> Self::Output {
        for ij in 0..self.matrix_elements.len() {
            let t_ij = &mut self.matrix_elements[ij];
            *t_ij = *t_ij * rhs;
        }
        self
    }
}
impl<'a, S: Scalar, const D0: usize, const D1: usize, const D2: usize> Div<S>
    for VolumeLinOp<'a, S, D0, D1, D2>
{
    type Output = Self;
    fn div(mut self, rhs: S) -> Self::Output {
        for ij in 0..self.matrix_elements.len() {
            let t_ij = &mut self.matrix_elements[ij];
            *t_ij = *t_ij / rhs;
        }
        self
    }
}
impl<'a, const D0: usize, const D1: usize, const D2: usize> LinOperator<Vec<Complex64>, Complex64>
    for VolumeLinOp<'a, Complex64, D0, D1, D2>
{
    fn trace(&self) -> Complex64 {
        (0..self.len)
            .map(|i| self[[i, i]])
            .reduce(|a, b| a + b)
            .unwrap_or(Complex64::zero())
    }

    fn to_faer(&self) -> faer::Mat<Complex64> {
        // faer::Mat::from_fn(self.len, self.len, |i, j| self[[i, j]])
        faer::Mat::from_fn(self.len, self.len, |i, j| self[[j, i]])
    }
}

#[cfg(test)]
mod test {
    use num_complex::Complex64;

    use crate::{
        linear::{BasisSet, Linear},
        vol::{Volume, VolumeBasisSet},
    };

    #[test]
    fn test_volume_basis_set() {
        let basis = VolumeBasisSet::<Complex64, 4, 1, 3>::plane_waves([4, 1, 3]);
        let v0 = Volume::new_with(|[x, _, _]| {
            Complex64::new((x as f64 / 2.0 * std::f64::consts::PI).sin(), 0.0)
        });
        assert!((basis.from_basis(basis.to_basis(v0.clone())) - v0.clone()).abs_sq() < 1.0e-10);
    }

    #[test]
    fn test_volume_integral() {
        const DR: f64 = 0.1;
        let v: Volume<f64, 10, 10, 10> = Volume::new_with(|[ix, _, _]| ix as f64);
        assert!(
            f64::abs(
                v.volume_integral(DR, |_, _, v| { v * 2.0 })
                    - (0..10)
                        .into_iter()
                        .map(|v| v as f64 * DR * 2.0)
                        .sum::<f64>()
            ) < 1.0e-10
        );
        let v: Volume<f64, 10, 10, 10> = Volume::new_with(|[_, _, _]| 0.0);
        assert!(
            f64::abs(
                v.volume_integral(DR, |[x, _, _], _, _| { x / DR * 2.0 })
                    - (0..10)
                        .into_iter()
                        .map(|v| v as f64 * DR * 2.0)
                        .sum::<f64>()
            ) < 1.0e-10
        );
    }
}
