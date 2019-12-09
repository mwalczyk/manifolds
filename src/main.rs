#![allow(warnings)]
mod complex;
mod fermat_surface;

use fermat_surface::FermatSurface;

use std::path::Path;

fn main() {

    let surface = FermatSurface::new();
    surface.save(&Path::new("fermat_surface.obj"));
}
