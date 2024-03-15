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
    fn into_imag(im: f64) -> Self;
    fn into_real(re: f64) -> Self;
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
    fn into_imag(im: f64) -> Self {
        Self::new(0.0, im)
    }
    fn into_real(re: f64) -> Self {
        Self::new(re, 0.0)
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
            e[i] = e[i].clone() / S::into_real(e[i].abs_sq().abs().sqrt());
        }
    }
}
