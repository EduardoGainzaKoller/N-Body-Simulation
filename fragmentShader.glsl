#version 330 core
out vec4 FragColor;

in vec3 Normal;
in vec3 FragPos;

uniform vec3 objectColor; // Color of the object
uniform vec3 lightColor;  // Color of the light
uniform vec3 lightPos;    // Position of the light source
uniform vec3 viewPos;     // Camera position (needed for specular highlights)

void main() {
    // 1. Ambient lighting
    float ambientStrength = 0.1f;
    vec3 ambient = ambientStrength * lightColor;

    // 2. Diffuse lighting
    vec3 norm = normalize(Normal);
    vec3 lightDir = normalize(lightPos - FragPos); // Direction from fragment to light
    float diff = max(dot(norm, lightDir), 0.0);
    vec3 diffuse = diff * lightColor;

    // 3. Specular lighting
    float specularStrength = 0.5f;
    vec3 viewDir = normalize(viewPos - FragPos); // Direction from fragment to camera
    vec3 reflectDir = reflect(-lightDir, norm);  // Light reflection direction
    float spec = pow(max(dot(viewDir, reflectDir), 0.0), 32); // Shininess factor (32 here)
    vec3 specular = specularStrength * spec * lightColor;

    // Combined lighting
    vec3 result = (ambient + diffuse + specular) * objectColor;
    FragColor = vec4(result, 1.0);
}