use ash::version::{DeviceV1_0, InstanceV1_0};
use ash::vk;

/// See: `https://stackoverflow.com/questions/28127165/how-to-convert-struct-to-u8`
pub unsafe fn any_as_u8_slice<T: Sized>(p: &T) -> &[u8] {
    ::std::slice::from_raw_parts((p as *const T) as *const u8, ::std::mem::size_of::<T>())
}

pub unsafe extern "system" fn debug_utils_callback(
    message_severity: vk::DebugUtilsMessageSeverityFlagsEXT,
    message_type: vk::DebugUtilsMessageTypeFlagsEXT,
    p_callback_data: *const vk::DebugUtilsMessengerCallbackDataEXT,
    _p_user_data: *mut std::os::raw::c_void,
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
    let message = std::ffi::CStr::from_ptr((*p_callback_data).p_message);
    println!("[Debug]{}{}{:?}", severity, types, message);

    vk::FALSE
}

pub fn get_memory_index(
    physical_device_memory_properties: &vk::PhysicalDeviceMemoryProperties,
    mut type_bits: u32,
    properties: vk::MemoryPropertyFlags,
) -> Option<u32> {
    for i in 0..physical_device_memory_properties.memory_type_count {
        if (type_bits & 1) == 1
            && (physical_device_memory_properties.memory_types[i as usize].property_flags
                & properties)
                == properties
        {
            return Some(i);
        }
        type_bits >>= 1;
    }
    // No memory type matching the specified memory property flags was found
    None
}

pub fn create_buffer(
    device: &ash::Device,
    physical_device_memory_properties: &vk::PhysicalDeviceMemoryProperties,
    buffer_usage_flags: vk::BufferUsageFlags,
    size: u64,
) -> (vk::Buffer, vk::DeviceMemory) {
    unsafe {
        // Create the buffer
        let buffer_create_info = vk::BufferCreateInfo::builder()
            .size(size)
            .usage(buffer_usage_flags)
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .build();
        let buffer = device
            .create_buffer(&buffer_create_info, None)
            .expect("Failed to create buffer");

        // Get the memory requirements of the buffer
        let memory_requirements = device.get_buffer_memory_requirements(buffer);
        let memory_allocate_info = vk::MemoryAllocateInfo::builder()
            .allocation_size(memory_requirements.size)
            .memory_type_index(
                get_memory_index(
                    physical_device_memory_properties,
                    memory_requirements.memory_type_bits,
                    vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
                )
                .expect("No suitable memory type found when attempting to allocate this buffer"),
            )
            .build();

        // Allocate device memory
        let device_memory = device
            .allocate_memory(&memory_allocate_info, None)
            .expect("Failed to allocate device memory");

        // Bind the device memory (with no offset)
        device.bind_buffer_memory(buffer, device_memory, 0);

        (buffer, device_memory)
    }
}

pub fn create_image_2d(
    device: &ash::Device,
    physical_device_memory_properties: &vk::PhysicalDeviceMemoryProperties,
    image_usage_flags: vk::ImageUsageFlags,
    width: u32,
    height: u32,
    format: vk::Format,
    tiling: vk::ImageTiling,
    memory_property_flags: vk::MemoryPropertyFlags,
    initial_layout: vk::ImageLayout,
) -> (vk::Image, vk::DeviceMemory) {
    unsafe {
        let image_create_info = vk::ImageCreateInfo::builder()
            .image_type(vk::ImageType::TYPE_2D)
            .format(format)
            .extent(
                vk::Extent3D::builder()
                    .width(width)
                    .height(height)
                    .depth(1)
                    .build(),
            )
            .array_layers(1)
            .mip_levels(1)
            .tiling(tiling)
            .usage(image_usage_flags)
            .samples(vk::SampleCountFlags::TYPE_1)
            .build();

        let image = device
            .create_image(&image_create_info, None)
            .expect("Failed to create image");

        // Get the memory requirements of the buffer
        let memory_requirements = device.get_image_memory_requirements(image);
        let memory_allocate_info = vk::MemoryAllocateInfo::builder()
            .allocation_size(memory_requirements.size)
            .memory_type_index(
                get_memory_index(
                    physical_device_memory_properties,
                    memory_requirements.memory_type_bits,
                    memory_property_flags,
                )
                .expect("No suitable memory type found when attempting to allocate this image"),
            )
            .build();

        // Allocate device memory
        let device_memory = device
            .allocate_memory(&memory_allocate_info, None)
            .expect("Failed to allocate device memory");

        // Bind the device memory (with no offset)
        device.bind_image_memory(image, device_memory, 0);

        (image, device_memory)
    }
}
