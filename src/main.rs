#![allow(warnings)]
mod complex;
mod fermat_surface;

use fermat_surface::FermatSurface;

use std::path::Path;

fn main() {
    let surface = FermatSurface::new(6, 1.0);
    surface.save(&Path::new("fermat_surface.obj"));
}
