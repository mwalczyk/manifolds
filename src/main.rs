#![allow(warnings)]
mod application;
mod complex;
mod fermat_surface;

use application::Application;
use fermat_surface::FermatSurface;

use std::path::Path;

fn main() {
    //let surface = FermatSurface::new(6, 1.0);
    //surface.save(&Path::new("fermat_surface.obj"));

    let event_loop = winit::event_loop::EventLoop::new();
    let _window = Application::initialize_window(&event_loop);

    let app = Application::new();
    app.main_loop(event_loop);
}
