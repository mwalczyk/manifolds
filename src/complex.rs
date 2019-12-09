use std::cmp::PartialEq;
use std::fmt::{Debug, Display};
use std::ops::{Add, Div, Mul, Sub};

pub struct Complex {
    /// The real part of this complex number
    pub re: f32,

    /// The imaginary part of this complex number
    pub im: f32,
}

impl Complex {
    pub fn new(re: f32, im: f32) -> Complex {
        Complex {
            re,
            im
        }
    }

    /// Returns the complex conjugate `z*` of this complex number, which has
    /// the same magnitude but opposite angle
    pub fn conjugate(&self) -> Complex {
        Complex::new(self.re, -self.im)
    }

    /// Returns the magnitude `|z|` of this complex number, which is its distance
    /// to the origin on the complex plane
    pub fn magnitude(&self) -> f32 {
        (self.re * self.re + self.im * self.im).sqrt()
    }

    /// Another name for "magnitude" - see the comments above
    pub fn modulus(&self) -> f32 {
        self.magnitude()
    }

    /// Returns the phase `φ` of this complex number, which is the angle it makes
    /// with the positive real axis
    pub fn phase(&self) -> f32 {
        self.im.atan2(self.re)
    }

    /// Another name for "phase" - see the comments above
    pub fn argument(&self) -> f32 {
        self.phase()
    }

    /// Returns this complex number in polar form
    pub fn polar_form(&self) -> Complex {
        let magnitude = self.magnitude();
        let phase = self.phase();

        return Complex::new(
          magnitude,
          phase
        );
    }

    /// Returns the reciprocal `1 / z` of this complex number
    pub fn reciprocal(&self) -> Complex {
        let denom = self.re * self.re + self.im * self.im;
        Complex::new(self.re / denom, -self.im / denom)
    }

    /// Returns the sine of this complex number
    pub fn sin(&self) -> Complex {
        Complex::new(
            self.re.sin() * self.im.cosh(),
            self.re.cos() * self.im.sinh()
        )
    }

    /// Returns the cosine of this complex number
    pub fn cos(&self) -> Complex {
        Complex::new(
            self.re.cos() * self.im.cosh(),
            -self.re.sin() * self.im.sinh()
        )
    }

    /// Returns `e` raised to the power of this complex number
    pub fn exp(&self) -> Complex {
        let coeff = self.re.exp();
        Complex::new(
            coeff * self.im.cos(),
            coeff * self.im.sin()
        )
    }

    /// Returns the result of raising this complex number to a real-number power
    pub fn pow(&self, scalar: f32) -> Complex {
        let polar = self.polar_form();
        let r = polar.re;
        let phi = polar.im;

        Complex::new(
            r.powf(scalar) * (scalar * phi).cos(),
            r.powf(scalar) * (scalar * phi).sin()
        )
    }
}

/// Equality of two complex numbers
impl PartialEq for Complex {
    fn eq(&self, other: &Self) -> bool {
        (self.re == other.re) && (self.im == other.im)
    }
}

/// Debug (print) for a complex number
impl Debug for Complex {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Complex {{ re: {}, im: {} }}", self.re, self.im)
    }
}

/// Display (print) for a complex number
impl Display for Complex {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{} + {}j", self.re, self.im)
    }
}

/// Addition of two complex numbers
impl Add for Complex {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        Self {
            re: self.re + other.re,
            im: self.im + other.im,
        }
    }
}

/// Division of two complex numbers
impl Div for Complex {
    type Output = Self;

    fn div(self, other: Self) -> Self {
        // Given two complex numbers:
        //      w = a + b*i
        //      z = c + d*i
        // The operation `w / z` is the same as:
        //      w * (1 / z)
        // Which can be found using the reciprocal of `z`
        self * other.reciprocal()
    }
}

/// Multiplication of two complex numbers
impl Mul for Complex {
    type Output = Self;

    fn mul(self, other: Self) -> Self {
        // Given two complex numbers:
        //      z = a + b*i
        //      w = c + d*i
        // Their product `z * w` equals:
        //      (ac - bd) + (ad + bc)i
        Self {
            re: self.re * other.re - self.im * other.im,
            im: self.re * other.im + self.im * other.re
        }
    }
}

/// Multiplication of a complex number by a scalar
impl Mul<f32> for Complex {
    type Output = Complex;

    fn mul(self, scalar: f32) -> Complex {
        Self {
            re: self.re * scalar,
            im: self.im * scalar
        }
    }
}

/// Subtraction of two complex numbers
impl Sub for Complex {
    type Output = Self;

    fn sub(self, other: Self) -> Self {
        Self {
            re: self.re - other.re,
            im: self.im - other.im,
        }
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_reciprocal() {
        println!("testing `reciprocal` function...");
        let z = Complex::new(0.0, 2.0);
        let result = z.reciprocal();
        println!("{}", result);

        // from python: (-0.5j)
    }

    #[test]
    fn test_mul() {
        println!("testing `mul` function...");
        let z = Complex::new(1.0, 2.0);
        let w = Complex::new(3.0, 4.0);
        let result = z * w;
        println!("{}", result);

        assert_eq!(result, Complex::new(-5.0, 10.0));
        // from python: (-5+10j)
    }

    #[test]
    fn test_div() {
        println!("testing `div` function...");
        let z = Complex::new(1.0, 2.0);
        let w = Complex::new(3.0, 4.0);
        let result = z / w;
        println!("{}", result);

        assert_eq!(result, Complex::new(0.44, 0.08));
        // from python: (0.44+0.08j)
    }

    #[test]
    fn test_sin() {
        println!("testing `sin` function...");
        let z = Complex::new(1.0, 2.0);
        let result = z.sin();
        println!("{}", result);

        // from python: (3.165778513216168+1.959601041421606j)
    }

    #[test]
    fn test_cos() {
        println!("testing `cos` function...");
        let z = Complex::new(1.0, 2.0);
        let result = z.cos();
        println!("{}", result);

        // from python: (2.0327230070196656-3.0518977991517997j)
    }

    #[test]
    fn test_exp() {
        println!("testing `exp` function...");
        let z = Complex::new(1.0, 2.0);
        let result = z.exp();
        println!("{}", result);

        // from python: (-1.1312043837568135+2.4717266720048188j)
    }
}