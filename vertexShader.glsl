#version 330 core
layout (location = 0) in vec3 aPos;
layout (location = 1) in vec3 aNormal; // New: Input normal vector

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

out vec3 Normal;   // New: Output normal to fragment shader
out vec3 FragPos;  // New: Output fragment position to fragment shader

void main() {
    FragPos = vec3(model * vec4(aPos, 1.0)); // Calculate world position of fragment
    Normal = mat3(transpose(inverse(model))) * aNormal; // Transform normal to world space
    
    gl_Position = projection * view * model * vec4(aPos, 1.0);
}