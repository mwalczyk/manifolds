#version 450

layout(location = 0) in vec3 i_position;

layout(location = 0) out vec3 o_color;

layout(push_constant) uniform PushConstants 
{
	float time;
	float padding;
	vec2 resolution;
	mat4 view_projection;
} push_constants;

mat4 rotation(vec3 axis, float angle)
{
    axis = normalize(axis);
    float s = sin(angle);
    float c = cos(angle);
    float oc = 1.0 - c;
    
    return mat4(oc * axis.x * axis.x + c,           oc * axis.x * axis.y - axis.z * s,  oc * axis.z * axis.x + axis.y * s,  0.0,
                oc * axis.x * axis.y + axis.z * s,  oc * axis.y * axis.y + c,           oc * axis.y * axis.z - axis.x * s,  0.0,
                oc * axis.z * axis.x - axis.y * s,  oc * axis.y * axis.z + axis.x * s,  oc * axis.z * axis.z + c,           0.0,
                0.0,                                0.0,                                0.0,                                1.0);
}

void main() 
{
	mat4 rot_x = rotation(vec3(1.0, 0.0, 0.0), push_constants.time * 0.25);
	mat4 rot_y = rotation(vec3(0.0, 1.0, 0.0), push_constants.time * 0.35);
    gl_Position = push_constants.view_projection * rot_x * rot_y * vec4(i_position, 1.0);

    o_color = i_position * 0.5 + 0.5;
}