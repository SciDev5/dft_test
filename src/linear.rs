use std::{
    fmt::Debug,
    ops::{Add, Div, Mul, Sub},
};

use num_complex::{Complex64, ComplexFloat};

pub trait Scalar:
    Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Div<Output = Self>
    + Debug
    + Clone
    + Copy
{
    fn zero() -> Self;
    fn exp(self) -> Self;
    fn hc(self) -> Self;
    fn abs(self) -> f64;
    fn new_im(im: f64) -> Self;
    fn new_re(re: f64) -> Self;
}

impl Scalar for Complex64 {
    fn zero() -> Self {
        0.0.into()
    }
    fn exp(self) -> Self {
        <Self as ComplexFloat>::exp(self)
    }
    fn hc(self) -> Self {
        self.conj()
    }
    fn abs(self) -> f64 {
        <Self as ComplexFloat>::abs(self)
    }
    fn new_im(im: f64) -> Self {
        Self::new(0.0, im)
    }
    fn new_re(re: f64) -> Self {
        Self::new(re, 0.0)
    }
}
impl Scalar for f64 {
    fn zero() -> Self {
        0.0
    }
    fn exp(self) -> Self {
        f64::exp(self)
    }
    fn hc(self) -> Self {
        self
    }
    fn abs(self) -> f64 {
        f64::abs(self)
    }
    fn new_im(_im: f64) -> Self {
        panic!("f64 is strictly real");
    }
    fn new_re(re: f64) -> Self {
        re
    }
}

pub trait Linear<S>:
    Sized
    + Add<Output = Self>
    + Sub<Output = Self>
    // + Mul<Output = Self>
    + Mul<S, Output = Self>
    + Div<S, Output = Self>
    + Debug
    + Clone
where
    S: Scalar,
{
    fn zero() -> Self;
    fn inner(bra: &Self, ket: &Self) -> S;
    fn abs_sq(&self) -> f64 {
        Self::inner(&self, &self).abs()
    }
}

pub fn orthonormalize_basis<T: Linear<S>, S: Scalar>(e: &mut [T]) {
    for i in 0..e.len() {
        for j in 0..i {
            // |e_i> = norm(|v_i> - |e_j><e_j| |v_i>) ; j in 0..i
            e[i] = e[i].clone() - e[j].clone() * T::inner(&e[j], &e[i]);
        }
        if e[i].abs_sq().abs() < 1.0e-5 {
            e[i] = T::zero();
        } else {
            e[i] = e[i].clone() / S::new_re(e[i].abs_sq().abs().sqrt());
        }
    }
}

pub trait BasisSet<L: Linear<S>, S: Scalar> {
    type BasisVectorIterator<'a>: Iterator<Item = &'a L>
    where
        Self: 'a,
        L: 'a;
    fn len(&self) -> usize;
    fn iter_basis_vectors<'a>(&'a self) -> Self::BasisVectorIterator<'a>;
    fn to_basis(&self, v: L) -> Vec<S> {
        self.iter_basis_vectors()
            .map(|e_i| Linear::inner(e_i, &v))
            .collect()
    }
    fn from_basis(&self, v: Vec<S>) -> L {
        debug_assert_eq!(v.len(), self.len());
        let mut v_out = L::zero();

        for (i, e_i) in self.iter_basis_vectors().enumerate() {
            v_out = v_out + e_i.clone() * v[i];
        }

        v_out
    }
}

pub trait LinOperator<L, S: Scalar + faer::Entity>
where
    for<'a> &'a Self: Mul<L, Output = L>,
{
    /// Takes the trace
    fn trace(&self) -> S;

    fn to_faer(&self) -> faer::Mat<S>;
    // /// Returns a Vec of eigenvalues.
    // fn calculate_eigendecomposition(&self) -> Vec<f64>;
}
