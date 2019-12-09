use crate::complex::Complex;

use std::fs::File;
use std::io::LineWriter;
use std::io::prelude::*;
use std::path::Path;

fn linear_spacing(lower: f32, upper: f32, steps: usize) -> Vec<f32> {
    (0..steps)
        .collect::<Vec<_>>()
        .iter()
        .map(|i| lower + (*i as f32) * (upper - lower) / (steps - 1) as f32)
        .collect::<Vec<f32>>()
}

type Point = [f32; 3];
type Polygon = [usize; 3];

pub struct FermatSurface {
    /// The exponent used to generate this surface
    n: u32,
    a: f32,
    points: Vec<Point>,
    polygons: Vec<Polygon>,
}

impl FermatSurface {
    pub fn new(n: u32, a: f32) -> FermatSurface {
        let mut surface = FermatSurface {
            n,
            a,
            points: vec![],
            polygons: vec![]
        };

        surface.build_topology();
        surface
    }

    pub fn build_topology(&mut self) {
        let steps = 20usize;

        // First, calculate points in 3-space
        for k1 in 0..self.n {
            for k2 in 0..self.n {
                let thetas = linear_spacing(0.0, std::f32::consts::FRAC_PI_2, steps);
                for theta in thetas.iter() {

                    let mut etas = linear_spacing(-1.0, 1.0, steps - 1);
                    etas.push(0.0);
                    etas.sort_by(|a, b| a.partial_cmp(b).unwrap());
                    //println!("{:?}", etas);
                    assert_eq!(steps, etas.len());

                    for eta in etas.iter() {
                        let phi1 = (2.0 * std::f32::consts::PI * k1 as f32) / self.n as f32;
                        let phi2 = (2.0 * std::f32::consts::PI * k2 as f32) / self.n as f32;

                        let s1 = Complex::new(0.0, phi1).exp();
                        let s2 = Complex::new(0.0, phi2).exp();

                        let mut u1 = (
                            Complex::new(*eta, *theta).exp() +
                            Complex::new(-*eta, -*theta).exp()
                        ) * 0.5;

                        let coeff = Complex::new(0.0, 2.0).reciprocal();
                        let mut u2 = coeff * (Complex::new(*eta, *theta).exp() - Complex::new(-*eta, -*theta).exp());

                        u1 = u1.pow(2.0 / self.n as f32);
                        u2 = u2.pow(2.0 / self.n as f32);
                        //println!("z1: {}", u1);
                        //println!("z2: {}", u2);
                        let mut z1 = s1 * u1;
                        let mut z2 = s2 * u2;

                        if *theta == 0.0 && *eta == 0.0 {
                            z1 = Complex::new(0.0, 2.0 * std::f32::consts::PI * k1 as f32 / self.n as f32).exp();
                            z2 = Complex::new(0.0, 0.0);
                        }


                        // Color each patch of the surface
                        let h = (k1 + k2) as f32 / self.n as f32;
                        let s = 1.0f32;
                        let v = 1.0f32;

                        // Calculate cartesian coordinates of this point (XYZ) in 3-space
                        let x = z1.re;
                        let y = z2.re;
                        let z = z1.im * self.a.cos() - z2.im * self.a.sin();
                        self.points.push([x, y, z]);
                    }
                }
            }
        }

        // Then, connect the points to form a closed surface
        for i in 0..self.points.len() {
            let col = i % steps;
            let row = i / steps;

            if row % steps == (steps - 1) {
                continue;
            }

            if i % steps != (steps - 1) {
                // First triangle
                self.polygons.push([
                    i,
                    i + 1,
                    i + steps
                ]);

                // Second triangle
                self.polygons.push([
                    i + 1,
                    i + steps + 1,
                    i + steps
                ]);
            }
        }
    }

    pub fn save(&self, path: &Path) -> std::io::Result<()> {
        let file = File::create(path)?;
        let mut file = LineWriter::new(file);

        for point in self.points.iter() {
            let buffer = format!("v {} {} {}\n", point[0], point[1], point[2]);
            file.write_all(buffer.as_bytes())?;
        }

        for polygon in self.polygons.iter() {
            let buffer = format!("f {} {} {}\n", polygon[0] + 1, polygon[1] + 1, polygon[2] + 1);
            file.write_all(buffer.as_bytes())?;
        }

        Ok(())
    }
}