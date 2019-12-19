use crate::utils;

use ash::extensions::ext::DebugUtils;
use ash::extensions::khr::{Surface, Swapchain, Win32Surface};
use ash::version::{DeviceV1_0, EntryV1_0, InstanceV1_0};
use ash::{vk, Instance};

extern crate nalgebra as na;
extern crate nalgebra_glm as glm;
use glm::{Mat4, Vec2, Vec3};
use na::Orthographic3;
use na::Perspective3;

use winapi::_core::mem::swap;
use winapi::um::combaseapi::CLSIDFromString;
use winapi::um::winbase::ApplicationRecoveryFinished;
use winit::event::{ElementState, Event, KeyboardInput, VirtualKeyCode, WindowEvent};
use winit::event_loop::{ControlFlow, EventLoop};
use winit::platform::windows::WindowExtWindows;
use winapi::shared::windef::HWND;
use winapi::um::libloaderapi::GetModuleHandleW;

use std::ffi::CString;
use std::os::raw::c_void;
use std::path::Path;
use std::ptr;

const WINDOW_TITLE: &'static str = "Fermat Surfaces";
const WINDOW_WIDTH: u32 = 800;
const WINDOW_HEIGHT: u32 = 800;

pub struct SwapchainDetails {
    loader: ash::extensions::khr::Swapchain,
    swapchain: vk::SwapchainKHR,
    swapchain_image_format: vk::Format,
    swapchain_image_extent: vk::Extent2D,
    swapchain_images: Vec<vk::Image>,
    swapchain_image_views: Vec<vk::ImageView>,
}

pub struct PhysicalDeviceDetails {
    physical_device: vk::PhysicalDevice,
    properties: vk::PhysicalDeviceProperties,
    features: vk::PhysicalDeviceFeatures,
    memory_properties: vk::PhysicalDeviceMemoryProperties,
    queue_family_properties: Vec<vk::QueueFamilyProperties>,
}

pub struct Mesh {
    vertices: Vec<f32>,
    indices: Vec<u32>,
    vertex_buffer: vk::Buffer,
    vertex_buffer_memory: vk::DeviceMemory,
    index_buffer: vk::Buffer,
    index_buffer_memory: vk::DeviceMemory,
}

#[repr(align(8))]
pub struct PushConstants {
    pub time: f32,
    pub _padding: f32,
    pub resolution: Vec2,
    pub view_projection: Mat4,
}

pub struct Application {
    entry: ash::Entry,
    instance: ash::Instance,
    debug_utils_loader: ash::extensions::ext::DebugUtils,
    debug_utils_messenger: vk::DebugUtilsMessengerEXT,
    physical_device_details: PhysicalDeviceDetails,
    device: ash::Device,
    queue: vk::Queue,
    swapchain_details: SwapchainDetails,
    depth_image: vk::Image,
    depth_image_memory: vk::DeviceMemory,
    depth_image_view: vk::ImageView,
    render_pass: vk::RenderPass,
    framebuffers: Vec<vk::Framebuffer>,
    pipeline_layout: vk::PipelineLayout,
    pipeline: vk::Pipeline,
    command_pool: vk::CommandPool,
    command_buffers: Vec<vk::CommandBuffer>,
    semaphores: [vk::Semaphore; 2],
    fences: Vec<vk::Fence>,
    meshes: Vec<Mesh>,
    pub push_constants: PushConstants,
}

impl Application {
    pub fn new(window: &winit::window::Window) -> Application {
        // Initialize the Vulkan SDK
        let entry = ash::Entry::new().expect("Failed to initialize Vulkan API entrypoint");

        // Create Vulkan objects
        let instance = Application::create_instance(&entry);
        let (debug_utils_loader, debug_utils_messenger) =
            Application::setup_debug_utils(&entry, &instance);
        let surface = Application::create_surface(&entry, &instance, window);
        let physical_device_details = Application::find_physical_device(&instance)
            .expect("No suitable physical device was found");
        let queue_family_index = Application::find_queue_family_index(&instance, &physical_device_details.physical_device)
            .expect("No suitable queue family index was found");
        let (device, queue) = Application::create_device(&instance, &physical_device_details.physical_device);
        let swapchain_details =
            Application::create_swapchain(&entry, &instance, &device, &physical_device_details.physical_device, &surface);

        let physical_device_memory_properties =
            unsafe { instance.get_physical_device_memory_properties(physical_device_details.physical_device) };
        let (depth_image, depth_image_memory, depth_image_view) =
            Application::create_depth_image(&device, &physical_device_details.memory_properties);

        let render_pass = Application::create_render_pass(&device, &swapchain_details);
        let framebuffers = Application::create_framebuffers(
            &device,
            &render_pass,
            &swapchain_details,
            &depth_image_view,
        );
        let (pipeline, pipeline_layout) =
            Application::create_graphics_pipeline(&device, &render_pass);

        let command_pool = Application::create_command_pool(&device, queue_family_index);
        let command_buffers =
            Application::create_command_buffers(&device, &command_pool, framebuffers.len() as u32);
        let (semaphores, fences) =
            Application::initialize_synchronization_primitives(&device, framebuffers.len() as u32);

        // Set up initial push constant state
        let view = glm::look_at(
            &Vec3::new(0.0, 0.0, 3.0),
            &Vec3::new(0.0, 0.0, 0.0),
            &Vec3::y_axis(),
        );
        let projection = glm::perspective(
            WINDOW_WIDTH as f32 / WINDOW_HEIGHT as f32,
            3.14f32 / 2.0f32,
            0.1f32,
            100.0f32,
        );
        let push_constants = PushConstants {
            time: 0.0,
            _padding: 0.0,
            resolution: Vec2::new(WINDOW_WIDTH as f32, WINDOW_HEIGHT as f32),
            view_projection: projection * view,
        };

        Application {
            entry,
            instance,
            debug_utils_loader,
            debug_utils_messenger,
            physical_device_details,
            device,
            queue,
            swapchain_details,
            depth_image,
            depth_image_memory,
            depth_image_view,
            render_pass,
            framebuffers,
            pipeline_layout,
            pipeline,
            command_pool,
            command_buffers,
            semaphores,
            fences,
            meshes: vec![],
            push_constants,
        }
    }

    pub fn initialize_window(event_loop: &EventLoop<()>) -> winit::window::Window {
        winit::window::WindowBuilder::new()
            .with_title(WINDOW_TITLE)
            .with_inner_size((WINDOW_WIDTH, WINDOW_HEIGHT).into())
            .build(event_loop)
            .expect("Failed to create window")
    }

    pub fn add_mesh(&mut self, vertices: &Vec<f32>, indices: &Vec<u32>) {
        if vertices.len() % 3 != 0 {
            eprintln!("The number of (flat) vertex coordinates passed to the model constructor \
            is not a multiple of 3 - this constructor assumes that the vertices are laid out in \
            a single flat array as x, y, z triplets")
        }
        let vertex_buffer_size = (std::mem::size_of::<f32>() * vertices.len()) as u64;
        let (vertex_buffer, vertex_buffer_memory) = utils::create_buffer(
            &self.device,
            &self.physical_device_details.memory_properties,
            vk::BufferUsageFlags::VERTEX_BUFFER,
            vertex_buffer_size,
        );
        let index_buffer_size = (std::mem::size_of::<u32>() * indices.len()) as u64;
        let (index_buffer, index_buffer_memory) = utils::create_buffer(
            &self.device,
            &self.physical_device_details.memory_properties,
            vk::BufferUsageFlags::INDEX_BUFFER,
            index_buffer_size,
        );
        println!(
            "Successfully added mesh with {} vertices to application",
            vertices.len() / 3
        );

        self.meshes.push(Mesh {
            vertices: vec![],
            indices: vec![],
            vertex_buffer,
            vertex_buffer_memory,
            index_buffer,
            index_buffer_memory,
        });

        self.update_mesh(self.meshes.len() - 1, vertices, indices);
    }

    pub fn update_mesh(&mut self, index: usize, vertices: &Vec<f32>, indices: &Vec<u32>) {
        self.meshes[index].vertices = vertices.clone();
        self.meshes[index].indices = indices.clone();

        unsafe {
            {
                let vertex_buffer_size = (std::mem::size_of::<f32>() * vertices.len()) as u64;
                let data_ptr = self
                    .device
                    .map_memory(
                        self.meshes[index].vertex_buffer_memory,
                        0,
                        vertex_buffer_size,
                        vk::MemoryMapFlags::empty(),
                    )
                    .expect("Failed to map vertex buffer memory")
                    as *mut f32;

                data_ptr.copy_from_nonoverlapping(vertices.as_ptr(), vertices.len());
                self.device
                    .unmap_memory(self.meshes[index].vertex_buffer_memory);
            }
            {
                let index_buffer_size = (std::mem::size_of::<u32>() * indices.len()) as u64;
                let data_ptr = self
                    .device
                    .map_memory(
                        self.meshes[index].index_buffer_memory,
                        0,
                        index_buffer_size,
                        vk::MemoryMapFlags::empty(),
                    )
                    .expect("Failed to map index buffer memory")
                    as *mut u32;

                data_ptr.copy_from_nonoverlapping(indices.as_ptr(), indices.len());
                self.device
                    .unmap_memory(self.meshes[index].index_buffer_memory);
            }
        }
    }

    fn get_debug_utils_messenger_create_info() -> vk::DebugUtilsMessengerCreateInfoEXT {
        vk::DebugUtilsMessengerCreateInfoEXT {
            s_type: vk::StructureType::DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT,
            p_next: ptr::null(),
            flags: vk::DebugUtilsMessengerCreateFlagsEXT::empty(),
            message_severity: vk::DebugUtilsMessageSeverityFlagsEXT::WARNING |
                vk::DebugUtilsMessageSeverityFlagsEXT::ERROR,
            message_type: vk::DebugUtilsMessageTypeFlagsEXT::GENERAL
                | vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE
                | vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION,
            pfn_user_callback: Some(utils::debug_utils_callback),
            p_user_data: ptr::null_mut(),
        }
    }

    fn create_instance(entry: &ash::Entry) -> ash::Instance {
        let debug_utils_messenger_create_info =
            Application::get_debug_utils_messenger_create_info();

        // Primarily here to test validation layers
        let application_info = vk::ApplicationInfo::default();

        // For debugging purposes
        let validation_layers = [CString::new("VK_LAYER_KHRONOS_validation").unwrap()];
        let enabled_layers: Vec<*const i8> = validation_layers
            .iter()
            .map(|layer_name| layer_name.as_ptr())
            .collect();

        // Instance-level extensions for windowing, surface creation, debugging, etc.
        let enabled_extensions = vec![
            Surface::name().as_ptr(),      // VK_KHR_surface
            Win32Surface::name().as_ptr(), // VK_KHR_win32_surface
            DebugUtils::name().as_ptr(),   // VK_EXT_debug_utils
        ];

        let mut instance_create_info = vk::InstanceCreateInfo::builder()
            .application_info(&application_info)
            .enabled_layer_names(&enabled_layers)
            .enabled_extension_names(&enabled_extensions)
            .build();
        instance_create_info.p_next = &debug_utils_messenger_create_info as *const vk::DebugUtilsMessengerCreateInfoEXT as *const c_void;

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

    fn create_surface(
        entry: &ash::Entry,
        instance: &ash::Instance,
        window: &winit::window::Window,
    ) -> vk::SurfaceKHR {
        unsafe {
            let hwnd = window.hwnd() as HWND;
            let hinstance = GetModuleHandleW(ptr::null()) as *const c_void;
            let win32_create_info = vk::Win32SurfaceCreateInfoKHR::builder()
                .hinstance(hinstance)
                .hwnd(hwnd as *const c_void)
                .build();
            let win32_surface_loader = Win32Surface::new(entry, instance);

            win32_surface_loader
                .create_win32_surface(&win32_create_info, None)
                .expect("Unable to create window surface")
        }
    }

    fn find_physical_device(instance: &ash::Instance) -> Option<PhysicalDeviceDetails> {
        unsafe {
            let physical_devices = instance
                .enumerate_physical_devices()
                .expect("Enumerate physical devices");

            if physical_devices.len() == 0 {
                return None;
            }
            println!("Found {} possible physical devices", physical_devices.len());

            for physical_device in physical_devices.iter() {
                let physical_device_properties =
                    instance.get_physical_device_properties(*physical_device);
                let physical_device_queue_family_properties =
                    instance.get_physical_device_queue_family_properties(*physical_device);

                // For now, return the first device with a discrete GPU
                if let vk::PhysicalDeviceType::DISCRETE_GPU = physical_device_properties.device_type
                {
                    println!("Physical device (discrete GPU) found");
                    println!(
                        "Physical device ID {}",
                        physical_device_properties.device_id
                    );
                    return Some(PhysicalDeviceDetails {
                        physical_device: *physical_device,
                        properties: physical_device_properties,
                        features: instance.get_physical_device_features(*physical_device),
                        memory_properties: instance.get_physical_device_memory_properties(*physical_device),
                        queue_family_properties: physical_device_queue_family_properties,
                    });
                }
            }
            None
        }
    }

    fn find_queue_family_index(
        instance: &ash::Instance,
        physical_device: &vk::PhysicalDevice,
    ) -> Option<u32> {
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

    fn create_device(
        instance: &ash::Instance,
        physical_device: &vk::PhysicalDevice,
    ) -> (ash::Device, vk::Queue) {
        unsafe {
            // Device-level extensions (just swapchain, for now)
            let extensions = [Swapchain::name().as_ptr()];

            // Information for constructing queues
            let queue_family_index =
                Application::find_queue_family_index(instance, physical_device)
                    .expect("No suitable queue family index found");
            let queue_priorities = [1.0];
            let device_queue_create_infos = [vk::DeviceQueueCreateInfo::builder()
                .queue_family_index(queue_family_index)
                .queue_priorities(&queue_priorities)
                .build()];

            // Enable certain features, like non-solid filling
            let mut physical_device_features = instance.get_physical_device_features(*physical_device);
            physical_device_features.fill_mode_non_solid = true as vk::Bool32;

            // Construct the device
            let device_create_info = vk::DeviceCreateInfo::builder()
                .enabled_extension_names(&extensions)
                .queue_create_infos(&device_queue_create_infos)
                .enabled_features(&physical_device_features)
                .build();

            let device = instance
                .create_device(*physical_device, &device_create_info, None)
                .unwrap();

            // Finally, grab the queue that we will submit drawing / presentation commands to
            let queue = device.get_device_queue(queue_family_index, 0);

            (device, queue)
        }
    }

    fn create_swapchain(
        entry: &ash::Entry,
        instance: &ash::Instance,
        device: &ash::Device,
        physical_device: &vk::PhysicalDevice,
        surface: &vk::SurfaceKHR,
    ) -> SwapchainDetails {
        unsafe {
            let surface_loader = Surface::new(entry, instance);

            // Check for surface support (the validation layers throw an error if we don't call this at
            // least once before creating a swapchain?)
            let queue_family_index =
                Application::find_queue_family_index(instance, physical_device).unwrap();
            let surface_support = surface_loader.get_physical_device_surface_support(
                *physical_device,
                queue_family_index,
                *surface,
            );

            let surface_formats = surface_loader
                .get_physical_device_surface_formats(*physical_device, *surface)
                .unwrap();
            let surface_format = surface_formats
                .iter()
                .map(|format| match format.format {
                    vk::Format::UNDEFINED => vk::SurfaceFormatKHR {
                        format: vk::Format::B8G8R8_UNORM,
                        color_space: format.color_space,
                    },
                    _ => *format,
                })
                .nth(0)
                .expect("Could not find a suitable surface format");
            println!(
                "Chose swapchain image format {:?} and color space {:?}",
                surface_format.format, surface_format.color_space
            );

            let surface_capabilities = surface_loader
                .get_physical_device_surface_capabilities(*physical_device, *surface)
                .unwrap();

            let mut desired_image_count = surface_capabilities.min_image_count + 1;
            if surface_capabilities.max_image_count > 0
                && desired_image_count > surface_capabilities.max_image_count
            {
                desired_image_count = surface_capabilities.max_image_count;
            }
            println!("{} images in swapchain", desired_image_count);

            // Choose the resolution of the swapchain images
            let surface_resolution = match surface_capabilities.current_extent.width {
                std::u32::MAX => vk::Extent2D {
                    width: WINDOW_WIDTH,
                    height: WINDOW_HEIGHT,
                },
                _ => surface_capabilities.current_extent,
            };
            println!(
                "Chose swapchain image resolution of {:?}",
                surface_resolution
            );

            // Choose the pre-transform
            let pre_transform = if surface_capabilities
                .supported_transforms
                .contains(vk::SurfaceTransformFlagsKHR::IDENTITY)
            {
                vk::SurfaceTransformFlagsKHR::IDENTITY
            } else {
                surface_capabilities.current_transform
            };

            // Choose the presentation mode (mailbox, if available)
            let present_modes = surface_loader
                .get_physical_device_surface_present_modes(*physical_device, *surface)
                .unwrap();
            let present_mode = present_modes
                .iter()
                .cloned()
                .find(|&mode| mode == vk::PresentModeKHR::MAILBOX)
                .unwrap_or(vk::PresentModeKHR::FIFO);
            println!("Using swapchain presentation mode {:?}", present_mode);

            let swapchain_loader = Swapchain::new(instance, device);

            let swapchain_create_info = vk::SwapchainCreateInfoKHR::builder()
                .surface(*surface)
                .min_image_count(desired_image_count)
                .image_color_space(surface_format.color_space)
                .image_format(surface_format.format)
                .image_extent(surface_resolution)
                .image_usage(vk::ImageUsageFlags::COLOR_ATTACHMENT)
                .image_sharing_mode(vk::SharingMode::EXCLUSIVE)
                .pre_transform(pre_transform)
                .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
                .present_mode(present_mode)
                .clipped(true)
                .image_array_layers(1);

            let swapchain = swapchain_loader
                .create_swapchain(&swapchain_create_info, None)
                .unwrap();

            let swapchain_images = swapchain_loader
                .get_swapchain_images(swapchain)
                .expect("Failed to retrieve swapchain images");

            let mut swapchain_image_views = vec![];
            for image in swapchain_images.iter() {
                let image_view_create_info = vk::ImageViewCreateInfo::builder()
                    .image(*image)
                    .view_type(vk::ImageViewType::TYPE_2D)
                    .format(surface_format.format)
                    .subresource_range(
                        vk::ImageSubresourceRange::builder()
                            .aspect_mask(vk::ImageAspectFlags::COLOR)
                            .base_mip_level(0)
                            .base_array_layer(0)
                            .level_count(1)
                            .layer_count(1)
                            .build(),
                    )
                    .build();
                swapchain_image_views.push(
                    device
                        .create_image_view(&image_view_create_info, None)
                        .expect("Failed to create swapchain image view"),
                );
            }
            SwapchainDetails {
                loader: swapchain_loader,
                swapchain,
                swapchain_image_format: surface_format.format,
                swapchain_image_extent: surface_resolution,
                swapchain_images,
                swapchain_image_views,
            }
        }
    }

    fn create_depth_image(
        device: &ash::Device,
        physical_device_memory_properties: &vk::PhysicalDeviceMemoryProperties,
    ) -> (vk::Image, vk::DeviceMemory, vk::ImageView) {
        unsafe {
            let (depth_image, depth_image_memory) = utils::create_image_2d(
                device,
                physical_device_memory_properties,
                vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT,
                WINDOW_WIDTH,
                WINDOW_HEIGHT,
                vk::Format::D32_SFLOAT,
                vk::ImageTiling::OPTIMAL,
                vk::MemoryPropertyFlags::DEVICE_LOCAL,
                vk::ImageLayout::UNDEFINED,
            );

            let image_view_create_info = vk::ImageViewCreateInfo::builder()
                .format(vk::Format::D32_SFLOAT)
                .subresource_range(
                    vk::ImageSubresourceRange::builder()
                        .layer_count(1)
                        .level_count(1)
                        .base_array_layer(0)
                        .base_mip_level(0)
                        .aspect_mask(vk::ImageAspectFlags::DEPTH)
                        .build(),
                )
                .view_type(vk::ImageViewType::TYPE_2D)
                .image(depth_image)
                .build();
            let depth_image_view = device
                .create_image_view(&image_view_create_info, None)
                .expect("Failed to create depth image view");

            (depth_image, depth_image_memory, depth_image_view)
        }
    }

    fn create_render_pass(
        device: &ash::Device,
        swapchain_details: &SwapchainDetails,
    ) -> vk::RenderPass {
        unsafe {
            let attachment_descriptions = [
                vk::AttachmentDescription {
                    format: swapchain_details.swapchain_image_format,
                    samples: vk::SampleCountFlags::TYPE_1,
                    load_op: vk::AttachmentLoadOp::CLEAR,
                    store_op: vk::AttachmentStoreOp::STORE,
                    initial_layout: vk::ImageLayout::UNDEFINED,
                    final_layout: vk::ImageLayout::PRESENT_SRC_KHR,
                    ..Default::default()
                },
                // Depth attachment yet
                vk::AttachmentDescription {
                    format: vk::Format::D32_SFLOAT,
                    samples: vk::SampleCountFlags::TYPE_1,
                    load_op: vk::AttachmentLoadOp::CLEAR,
                    store_op: vk::AttachmentStoreOp::DONT_CARE,
                    initial_layout: vk::ImageLayout::UNDEFINED,
                    final_layout: vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
                    ..Default::default()
                },
            ];

            let attachment_references = [
                vk::AttachmentReference {
                    attachment: 0,
                    layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
                },
                vk::AttachmentReference {
                    attachment: 1,
                    layout: vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
                },
            ];

            let subpass_dependencies = [vk::SubpassDependency {
                src_subpass: vk::SUBPASS_EXTERNAL,
                src_stage_mask: vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                dst_access_mask: vk::AccessFlags::COLOR_ATTACHMENT_READ
                    | vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
                dst_stage_mask: vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                ..Default::default()
            }];

            let subpass_descriptions = [vk::SubpassDescription::builder()
                .color_attachments(&[attachment_references[0]])
                .depth_stencil_attachment(&attachment_references[1])
                .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
                .build()];

            let render_pass_create_info = vk::RenderPassCreateInfo::builder()
                .attachments(&attachment_descriptions)
                .subpasses(&subpass_descriptions)
                .dependencies(&subpass_dependencies);

            let render_pass = device
                .create_render_pass(&render_pass_create_info, None)
                .expect("Failed to create render pass");
            render_pass
        }
    }

    fn create_framebuffers(
        device: &ash::Device,
        render_pass: &vk::RenderPass,
        swapchain_details: &SwapchainDetails,
        depth_image_view: &vk::ImageView,
    ) -> Vec<vk::Framebuffer> {
        unsafe {
            swapchain_details
                .swapchain_image_views
                .iter()
                .map(|&image_view| {
                    // For each image view in the swapchain, construct a new framebuffer
                    let attachments = [image_view, *depth_image_view];
                    let framebuffer_create_info = vk::FramebufferCreateInfo::builder()
                        .render_pass(*render_pass)
                        .attachments(&attachments)
                        .width(swapchain_details.swapchain_image_extent.width)
                        .height(swapchain_details.swapchain_image_extent.height)
                        .layers(1);

                    device
                        .create_framebuffer(&framebuffer_create_info, None)
                        .unwrap()
                })
                .collect()
        }
    }

    fn create_shader_module(device: &ash::Device, shader_code: Vec<u32>) -> vk::ShaderModule {
        unsafe {
            let shader_module_create_info = vk::ShaderModuleCreateInfo::builder()
                .code(&shader_code)
                .build();

            device
                .create_shader_module(&shader_module_create_info, None)
                .expect("Failed to create shader module")
        }
    }

    fn create_graphics_pipeline(
        device: &ash::Device,
        render_pass: &vk::RenderPass,
    ) -> (vk::Pipeline, vk::PipelineLayout) {
        unsafe {
            // First, load the shader modules
            let mut vert_file = std::fs::File::open("shaders/vert.spv").unwrap();
            let vert_shader_code = ash::util::read_spv(&mut vert_file).unwrap();
            let vert_shader_module = Application::create_shader_module(device, vert_shader_code);

            let mut frag_file = std::fs::File::open("shaders/frag.spv").unwrap();
            let frag_shader_code = ash::util::read_spv(&mut frag_file).unwrap();
            let frag_shader_module = Application::create_shader_module(device, frag_shader_code);

            // Create the pipeline layout: add descriptors, push constants, etc.
            println!(
                "Size of push constants: {}",
                std::mem::size_of::<PushConstants>() as u32
            );
            let push_constant_range = vk::PushConstantRange::builder()
                .size(std::mem::size_of::<PushConstants>() as u32)
                .offset(0)
                .stage_flags(vk::ShaderStageFlags::VERTEX)
                .build();

            let pipeline_layout_create_info = vk::PipelineLayoutCreateInfo::builder()
                .push_constant_ranges(&[push_constant_range])
                .build();
            let pipeline_layout = device
                .create_pipeline_layout(&pipeline_layout_create_info, None)
                .expect("Failed to create pipeline layout");

            // Assemble structs for graphics pipeline
            let entry = CString::new("main").unwrap();
            let shader_stage_create_infos = [
                vk::PipelineShaderStageCreateInfo::builder()
                    .stage(vk::ShaderStageFlags::VERTEX)
                    .module(vert_shader_module)
                    .name(&entry)
                    .build(),
                vk::PipelineShaderStageCreateInfo::builder()
                    .stage(vk::ShaderStageFlags::FRAGMENT)
                    .module(frag_shader_module)
                    .name(&entry)
                    .build(),
            ];

            // For now, we are hard-coding the vertices into the shader
            let vertex_input_binding_description = vk::VertexInputBindingDescription::builder()
                .binding(0)
                .input_rate(vk::VertexInputRate::VERTEX)
                .stride((std::mem::size_of::<f32>() * 3) as u32)
                .build();
            let vertex_input_attribute_description = vk::VertexInputAttributeDescription::builder()
                .binding(0) // Which buffer bindpoint will this attribute be sourced from
                .offset(0)
                .format(vk::Format::R32G32B32_SFLOAT)
                .location(0) // The location identifier (as seen in the vertex shader) of this attribute
                .build();
            let vertex_input_state_create_info = vk::PipelineVertexInputStateCreateInfo::builder()
                .vertex_binding_descriptions(&[vertex_input_binding_description])
                .vertex_attribute_descriptions(&[vertex_input_attribute_description])
                .build();

            // Triangles
            let input_assembly_state_create_info =
                vk::PipelineInputAssemblyStateCreateInfo::builder()
                    .topology(vk::PrimitiveTopology::TRIANGLE_LIST)
                    .build();

            // Set up viewport(s) and scissor(s)
            let viewports = [vk::Viewport::builder()
                .x(0.0)
                .y(0.0)
                .width(WINDOW_WIDTH as f32)
                .height(WINDOW_HEIGHT as f32)
                .min_depth(0.0)
                .max_depth(1.0)
                .build()];
            let scissors = [vk::Rect2D::builder()
                .offset(vk::Offset2D::default())
                .extent(vk::Extent2D {
                    width: WINDOW_WIDTH,
                    height: WINDOW_HEIGHT,
                })
                .build()];
            let viewport_state_create_info = vk::PipelineViewportStateCreateInfo::builder()
                .viewports(&viewports)
                .scissors(&scissors)
                .build();

            // Set up rasterization state
            let rasterization_state_create_info = vk::PipelineRasterizationStateCreateInfo {
                front_face: vk::FrontFace::CLOCKWISE,
                line_width: 1.0,
                polygon_mode: vk::PolygonMode::LINE,
                cull_mode: vk::CullModeFlags::NONE,
                ..Default::default()
            };

            // Set up multisample state
            let multisample_state_create_info = vk::PipelineMultisampleStateCreateInfo {
                rasterization_samples: vk::SampleCountFlags::TYPE_1,
                ..Default::default()
            };

            // Set up color blend state
            let color_blend_attachment_states = [vk::PipelineColorBlendAttachmentState {
                blend_enable: 0,
                src_color_blend_factor: vk::BlendFactor::SRC_COLOR,
                dst_color_blend_factor: vk::BlendFactor::ONE_MINUS_DST_COLOR,
                color_blend_op: vk::BlendOp::ADD,
                src_alpha_blend_factor: vk::BlendFactor::ZERO,
                dst_alpha_blend_factor: vk::BlendFactor::ZERO,
                alpha_blend_op: vk::BlendOp::ADD,
                color_write_mask: vk::ColorComponentFlags::all(),
            }];
            let color_blend_state_create_info = vk::PipelineColorBlendStateCreateInfo::builder()
                .logic_op(vk::LogicOp::CLEAR)
                .attachments(&color_blend_attachment_states)
                .build();

            let depth_stencil_state_create_info =
                vk::PipelineDepthStencilStateCreateInfo::builder()
                    .depth_test_enable(true)
                    .depth_write_enable(true)
                    .depth_compare_op(vk::CompareOp::LESS)
                    .build();

            let graphics_pipeline_create_info = vk::GraphicsPipelineCreateInfo::builder()
                .stages(&shader_stage_create_infos)
                .vertex_input_state(&vertex_input_state_create_info)
                .input_assembly_state(&input_assembly_state_create_info)
                .viewport_state(&viewport_state_create_info)
                .rasterization_state(&rasterization_state_create_info)
                .multisample_state(&multisample_state_create_info)
                .depth_stencil_state(&depth_stencil_state_create_info)
                .color_blend_state(&color_blend_state_create_info)
                .layout(pipeline_layout)
                .render_pass(*render_pass)
                .build();

            let pipelines = device
                .create_graphics_pipelines(
                    vk::PipelineCache::null(),
                    &[graphics_pipeline_create_info],
                    None,
                )
                .expect("Failed to create graphics pipeline");

            (pipelines[0], pipeline_layout)
        }
    }

    fn create_command_pool(device: &ash::Device, queue_family_index: u32) -> vk::CommandPool {
        unsafe {
            let command_pool_create_info = vk::CommandPoolCreateInfo::builder()
                .queue_family_index(queue_family_index)
                .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER)
                .build();

            device
                .create_command_pool(&command_pool_create_info, None)
                .expect("Failed to create command pool")
        }
    }

    fn create_command_buffers(
        device: &ash::Device,
        command_pool: &vk::CommandPool,
        count: u32,
    ) -> Vec<vk::CommandBuffer> {
        unsafe {
            let command_buffer_allocate_info = vk::CommandBufferAllocateInfo::builder()
                .command_buffer_count(count)
                .command_pool(*command_pool)
                .level(vk::CommandBufferLevel::PRIMARY)
                .build();

            device
                .allocate_command_buffers(&command_buffer_allocate_info)
                .expect("Failed to allocate command buffers")
        }
    }

    fn initialize_synchronization_primitives(
        device: &ash::Device,
        count: u32,
    ) -> ([vk::Semaphore; 2], Vec<vk::Fence>) {
        unsafe {
            let semaphores = [
                device
                    .create_semaphore(&vk::SemaphoreCreateInfo::default(), None)
                    .expect("Failed to create 1st semaphore"),
                device
                    .create_semaphore(&vk::SemaphoreCreateInfo::default(), None)
                    .expect("Failed to create 2nd semaphore"),
            ];

            let mut fences = vec![];
            for i in 0..count {
                let fence_create_info = vk::FenceCreateInfo::builder()
                    .flags(vk::FenceCreateFlags::SIGNALED)
                    .build();
                fences.push(
                    device
                        .create_fence(&fence_create_info, None)
                        .expect("Failed to create fence"),
                );
            }

            (semaphores, fences)
        }
    }

    fn record_command_buffer(&self, index: u32) {
        let clear_values = [
            vk::ClearValue {
                color: vk::ClearColorValue {
                    float32: [0.0, 0.0, 0.0, 1.0],
                },
            },
            vk::ClearValue {
                depth_stencil: vk::ClearDepthStencilValue {
                    depth: 1.0,
                    stencil: 0,
                },
            },
        ];

        let render_area = vk::Rect2D::builder()
            .extent(
                vk::Extent2D::builder()
                    .width(WINDOW_WIDTH)
                    .height(WINDOW_HEIGHT)
                    .build(),
            )
            .build();

        unsafe {
            // Start actual commands
            self.device.begin_command_buffer(
                self.command_buffers[index as usize],
                &vk::CommandBufferBeginInfo::builder()
                    .flags(vk::CommandBufferUsageFlags::SIMULTANEOUS_USE)
                    .build(),
            );
            self.device.cmd_begin_render_pass(
                self.command_buffers[index as usize],
                &vk::RenderPassBeginInfo::builder()
                    .render_pass(self.render_pass)
                    .framebuffer(self.framebuffers[index as usize])
                    .render_area(render_area)
                    .clear_values(&clear_values)
                    .build(),
                vk::SubpassContents::INLINE,
            );
            self.device.cmd_bind_pipeline(
                self.command_buffers[index as usize],
                vk::PipelineBindPoint::GRAPHICS,
                self.pipeline,
            );

            self.device.cmd_push_constants(
                self.command_buffers[index as usize],
                self.pipeline_layout,
                vk::ShaderStageFlags::VERTEX,
                0,
                utils::any_as_u8_slice(&self.push_constants),
            );

            // Draw all of the meshes
            for mesh in self.meshes.iter() {
                unsafe {
                    self.device.cmd_bind_vertex_buffers(
                        self.command_buffers[index as usize],
                        0,
                        &[mesh.vertex_buffer],
                        &[0],
                    );

                    self.device.cmd_bind_index_buffer(
                        self.command_buffers[index as usize],
                        mesh.index_buffer,
                        0,
                        vk::IndexType::UINT32,
                    );

                    self.device.cmd_draw_indexed(
                        self.command_buffers[index as usize],
                        mesh.indices.len() as u32,
                        1,
                        0,
                        0,
                        0,
                    );
                }
            }

            //self.device
            //   .cmd_draw(self.command_buffers[index as usize], 3, 1, 0, 0);
            self.device
                .cmd_end_render_pass(self.command_buffers[index as usize]);
            self.device
                .end_command_buffer(self.command_buffers[index as usize]);
        }
    }

    pub fn draw_frame(&mut self) {
        unsafe {
            // Grab the index of the next available swapchain image
            let (index, _) = self
                .swapchain_details
                .loader
                .acquire_next_image(
                    self.swapchain_details.swapchain,
                    std::u64::MAX,
                    self.semaphores[0],
                    vk::Fence::null(),
                )
                .unwrap();

            // Don't re-record this command buffer until it is done being used (from the previous set of frames)
            self.device
                .wait_for_fences(&[self.fences[index as usize]], true, std::u64::MAX);
            self.device.reset_fences(&[self.fences[index as usize]]);

            // Record draw commands, etc.
            self.record_command_buffer(index);

            // Submit the command buffer
            let wait_stages = [vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT];
            let submit_info = vk::SubmitInfo::builder()
                .command_buffers(&[self.command_buffers[index as usize]])
                .wait_semaphores(&[self.semaphores[0]])
                .wait_dst_stage_mask(&wait_stages)
                .signal_semaphores(&[self.semaphores[1]])
                .build();
            self.device
                .queue_submit(self.queue, &[submit_info], self.fences[index as usize]);

            // Present the newly rendered image to the screen
            let present_info = vk::PresentInfoKHR::builder()
                .wait_semaphores(&[self.semaphores[1]])
                .swapchains(&[self.swapchain_details.swapchain])
                .image_indices(&[index])
                .build();
            self.swapchain_details
                .loader
                .queue_present(self.queue, &present_info)
                .unwrap();
        }
    }

    pub fn main_loop(mut self, event_loop: EventLoop<()>) {}
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
