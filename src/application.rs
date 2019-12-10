use std::ffi::{CStr, CString};
use std::os::raw::c_void;
use std::ptr;

use ash::extensions::ext::DebugUtils;
use ash::extensions::khr::{Surface, Win32Surface};
use ash::version::{EntryV1_0, InstanceV1_0};
use ash::vk;
use winit::event::{ElementState, Event, KeyboardInput, VirtualKeyCode, WindowEvent};
use winit::event_loop::{ControlFlow, EventLoop};

const WINDOW_TITLE: &'static str = "Fermat Surfaces";
const WINDOW_WIDTH: u32 = 800;
const WINDOW_HEIGHT: u32 = 800;

unsafe extern "system" fn debug_utils_callback(
    message_severity: vk::DebugUtilsMessageSeverityFlagsEXT,
    message_type: vk::DebugUtilsMessageTypeFlagsEXT,
    p_callback_data: *const vk::DebugUtilsMessengerCallbackDataEXT,
    _p_user_data: *mut c_void,
) -> vk::Bool32 {
    let severity = match message_severity {
        vk::DebugUtilsMessageSeverityFlagsEXT::VERBOSE => "[Verbose]",
        vk::DebugUtilsMessageSeverityFlagsEXT::WARNING => "[Warning]",
        vk::DebugUtilsMessageSeverityFlagsEXT::ERROR => "[Error]",
        vk::DebugUtilsMessageSeverityFlagsEXT::INFO => "[Info]",
        _ => "[Unknown]",
    };
    let types = match message_type {
        vk::DebugUtilsMessageTypeFlagsEXT::GENERAL => "[General]",
        vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE => "[Performance]",
        vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION => "[Validation]",
        _ => "[Unknown]",
    };
    let message = CStr::from_ptr((*p_callback_data).p_message);
    println!("[Debug]{}{}{:?}", severity, types, message);

    vk::FALSE
}

pub struct Application {
    _entry: ash::Entry,
    instance: ash::Instance,
    debug_utils_loader: ash::extensions::ext::DebugUtils,
    debug_utils_messenger: vk::DebugUtilsMessengerEXT,
    physical_device: vk::PhysicalDevice,
}

impl Application {
    pub fn new() -> Application {
        // Initialize the Vulkan SDK
        let entry = ash::Entry::new().unwrap();

        // Create Vulkan objects
        let instance = Application::create_instance(&entry);
        let (debug_utils_loader, debug_utils_messenger) =
            Application::setup_debug_utils(&entry, &instance);
        let physical_device = Application::create_physical_device(&instance)
            .expect("No suitable physical device was found");
        let queue_family_index = Application::create_queue(&instance, &physical_device)
            .expect("No suitable queue was found");

        Application {
            _entry: entry,
            instance,
            debug_utils_loader,
            debug_utils_messenger,
            physical_device,
        }
    }

    pub fn initialize_window(event_loop: &EventLoop<()>) -> winit::window::Window {
        winit::window::WindowBuilder::new()
            .with_title(WINDOW_TITLE)
            .with_inner_size((WINDOW_WIDTH, WINDOW_HEIGHT).into())
            .build(event_loop)
            .expect("Failed to create window")
    }

    fn get_debug_utils_messenger_create_info() -> vk::DebugUtilsMessengerCreateInfoEXT {
        vk::DebugUtilsMessengerCreateInfoEXT {
            s_type: vk::StructureType::DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT,
            p_next: ptr::null(),
            flags: vk::DebugUtilsMessengerCreateFlagsEXT::empty(),
            message_severity: vk::DebugUtilsMessageSeverityFlagsEXT::WARNING |
                // vk::DebugUtilsMessageSeverityFlagsEXT::VERBOSE |
                // vk::DebugUtilsMessageSeverityFlagsEXT::INFO |
                vk::DebugUtilsMessageSeverityFlagsEXT::ERROR,
            message_type: vk::DebugUtilsMessageTypeFlagsEXT::GENERAL
                | vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE
                | vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION,
            pfn_user_callback: Some(debug_utils_callback),
            p_user_data: ptr::null_mut(),
        }
    }

    fn create_instance(entry: &ash::Entry) -> ash::Instance {
        let debug_utils_messenger_create_info =
            Application::get_debug_utils_messenger_create_info();

        // Primarily here to test validation layers
        let application_info = vk::ApplicationInfo {
            api_version: ash::vk_make_version!(2, 0, 92),
            ..Default::default()
        }; // Or: vk::ApplicationInfo::default()

        // For debugging purposes
        let validation_layers = [CString::new("VK_LAYER_KHRONOS_validation").unwrap()];

        let enabled_layers: Vec<*const i8> = validation_layers
            .iter()
            .map(|layer_name| layer_name.as_ptr())
            .collect();

        let enabled_extensions = vec![
            Surface::name().as_ptr(),      // VK_KHR_surface
            Win32Surface::name().as_ptr(), // VK_KHR_win32_surface
            DebugUtils::name().as_ptr(),   // VK_EXT_debug_utils
        ];

        let instance_create_info = vk::InstanceCreateInfo {
            s_type: vk::StructureType::INSTANCE_CREATE_INFO,
            p_next: &debug_utils_messenger_create_info
                as *const vk::DebugUtilsMessengerCreateInfoEXT as *const c_void,
            flags: vk::InstanceCreateFlags::empty(),
            p_application_info: &application_info,
            pp_enabled_layer_names: enabled_layers.as_ptr(),
            enabled_layer_count: enabled_layers.len() as u32,
            pp_enabled_extension_names: enabled_extensions.as_ptr(),
            enabled_extension_count: enabled_extensions.len() as u32,
        };

        let instance = unsafe {
            entry
                .create_instance(&instance_create_info, None)
                .expect("Failed to create instance")
        };

        instance
    }

    fn setup_debug_utils(
        entry: &ash::Entry,
        instance: &ash::Instance,
    ) -> (ash::extensions::ext::DebugUtils, vk::DebugUtilsMessengerEXT) {
        let debug_utils_loader = ash::extensions::ext::DebugUtils::new(entry, instance);
        let debug_utils_messenger_create_info =
            Application::get_debug_utils_messenger_create_info();

        let debug_utils_messenger = unsafe {
            debug_utils_loader
                .create_debug_utils_messenger(&debug_utils_messenger_create_info, None)
                .expect("Debug utils callback")
        };

        (debug_utils_loader, debug_utils_messenger)
    }

    fn create_physical_device(instance: &ash::Instance) -> Option<vk::PhysicalDevice> {
        unsafe {
            let physical_devices = instance
                .enumerate_physical_devices()
                .expect("Enumerate physical devices");

            if physical_devices.len() == 0 {
                return None;
            }

            println!("Found {} physical device", physical_devices.len());
            for physical_device in physical_devices.iter() {
                let physical_device_properties =
                    instance.get_physical_device_properties(*physical_device);
                let physical_device_queue_family_properties =
                    instance.get_physical_device_queue_family_properties(*physical_device);

                // For now, return the first device with a discrete GPU
                if let vk::PhysicalDeviceType::DISCRETE_GPU = physical_device_properties.device_type
                {
                    println!("Physical device (discrete GPU) found");
                    println!("{:?}", physical_device_properties);
                    return Some(*physical_device);
                }
            }
            None
        }
    }

    fn create_queue(instance: &ash::Instance, physical_device: &vk::PhysicalDevice) -> Option<u32> {
        unsafe {
            let physical_device_queue_family_properties =
                instance.get_physical_device_queue_family_properties(*physical_device);

            for (i, queue_family_properties) in
                physical_device_queue_family_properties.iter().enumerate()
            {
                if queue_family_properties
                    .queue_flags
                    .contains(vk::QueueFlags::GRAPHICS)
                {
                    println!(
                        "Found queue at index {} that supports graphics operations",
                        i
                    );
                    return Some(i as u32);
                }
            }
            None
        }
    }

    fn draw_frame(&mut self) {
        // Drawing will be here
    }

    pub fn main_loop(mut self, event_loop: EventLoop<()>) {
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
                self.draw_frame();
            }
            _ => (),
        })
    }
}

impl Drop for Application {
    fn drop(&mut self) {
        unsafe {
            self.debug_utils_loader
                .destroy_debug_utils_messenger(self.debug_utils_messenger, None);
            self.instance.destroy_instance(None);
        }
    }
}
