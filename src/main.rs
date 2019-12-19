#![allow(warnings)]
mod application;
mod complex;
mod fermat_surface;
mod utils;

use application::Application;
use fermat_surface::FermatSurface;

use ::slice_of_array::prelude::*;
use winit::event::{ElementState, Event, KeyboardInput, VirtualKeyCode, WindowEvent};
use winit::event_loop::{ControlFlow, EventLoop};

use ash::version::InstanceV1_0;
use std::path::Path;
use std::sync::Arc;

fn main() {
    // Create the initial topology
    let surface = FermatSurface::new(9, 2.0);
    let save_obj = true;
    if save_obj {
        surface.save(&Path::new("fermat_surface.obj"));
    }

    // Create the rendering application
    let event_loop = winit::event_loop::EventLoop::new();
    let window = Application::initialize_window(&event_loop);
    let mut app = Application::new(&window);

    // Create vertex + index buffers on the GPU
    let vertices: Vec<_> = surface.points.flat().to_vec();
    let indices: Vec<_> = surface.polygons.flat().to_vec();
    app.add_mesh(&vertices, &indices);

    // Start the main draw loop
    let mut now = std::time::Instant::now();
    event_loop.run(move |event, _, control_flow| match event {
        Event::WindowEvent { event, .. } => match event {
            WindowEvent::CloseRequested => *control_flow = ControlFlow::Exit,
            WindowEvent::KeyboardInput { input, .. } => match input {
                KeyboardInput {
                    virtual_keycode,
                    state,
                    ..
                } => match (virtual_keycode, state) {
                    (Some(VirtualKeyCode::Escape), ElementState::Pressed) => {
                        *control_flow = ControlFlow::Exit
                    }
                    _ => {}
                },
            },
            _ => {}
        },
        Event::EventsCleared => {
            let t = now.elapsed().as_secs_f32();
            app.push_constants.time = t;
            let a = (t.sin() * 0.5 + 0.5) * 4.0 + 1.0;
            //let surface = FermatSurface::new(9, a);
            //let vertices: Vec<_> = surface.points.flat().to_vec();
            //let indices: Vec<_> = surface.polygons.flat().to_vec();
            //app.update_mesh(0, &vertices, &indices);
            app.draw_frame();
        }
        _ => (),
    })
}
