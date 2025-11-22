#include "N-Body Simulation.h"
#include <iostream>
#include <vector>
#include <cmath>
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtc/constants.hpp> // For glm::pi<float>()
#include <fstream>
#include <sstream>

using namespace std;

// --- PHYSICS AND GRAPHICS CONSTANTS ---
// Using G=1.0 and lower DeltaTime for basic stability
const double G = 1.0;
const double DELTA_TIME = 0.05; // Time step for the simulation (reduced for stability)
const double SOFTENING = 1e-3; // Small factor to avoid singularities (r -> 0)

const unsigned int SCR_WIDTH = 1200;
const unsigned int SCR_HEIGHT = 800;

// --- CAMERA MOVEMENT CONSTANTS ---
const float CAMERA_SPEED = 50.0f; // Speed units per second
const float MOUSE_SENSITIVITY = 0.1f;   // Mouse sensitivity

// --- GLOBAL GRAPHICS VARIABLES ---
GLFWwindow* window;
GLuint shaderProgram; // Shader program ID
GLuint VAO;
GLuint sphereVBO, sphereEBO; // Buffers for sphere mesh data

// --- SPHERE DATA ---
std::vector<float> sphereVertices;
std::vector<float> sphereNormals;
std::vector<unsigned int> sphereIndices;
unsigned int sphereIndexCount; // Number of indices to draw

// --- CAMERA VECTORS AND TRACKING ---
glm::vec3 cameraPos = glm::vec3(0.0f, 0.0f, 300.0f); // Initial camera position
glm::vec3 cameraUp = glm::vec3(0.0f, 1.0f, 0.0f);   // World up vector

glm::vec3 cameraFront = glm::vec3(0.0f, 0.0f, -1.0f); // Direction the camera is currently looking
float lastX = SCR_WIDTH / 2.0f;
float lastY = SCR_HEIGHT / 2.0f;
float yaw = -90.0f; // Yaw (rotation around Y-axis) 
float pitch = 0.0f; // Pitch (rotation around X-axis)
bool firstMouse = true;

// --- LIGHTING ---
glm::vec3 lightPos = glm::vec3(0.0f, 500.0f, 300.0f); // Position of the light source
glm::vec3 lightColor = glm::vec3(1.0f, 1.0f, 1.0f);  // White light

// --- CLASSES AND PHYSICS ---

class Vector3D {
public:
    double x, y, z;

    Vector3D(double x = 0, double y = 0, double z = 0) : x(x), y(y), z(z) {}

    double magnitude() const {
        return sqrt(x * x + y * y + z * z);
    }

    Vector3D operator+(const Vector3D& other) const {
        return Vector3D(x + other.x, y + other.y, z + other.z);
    }

    Vector3D operator-(const Vector3D& other) const {
        return Vector3D(x - other.x, y - other.y, z - other.z);
    }

    Vector3D operator*(double scalar) const {
        return Vector3D(x * scalar, y * scalar, z * scalar);
    }

    Vector3D& operator+=(const Vector3D& other) {
        x += other.x; y += other.y; z += other.z;
        return *this;
    }
};

class Body {
public:
    Vector3D position;
    Vector3D velocity;
    Vector3D acceleration;
    double mass;

    Body(Vector3D pos, Vector3D vel, double m) : position(pos), velocity(vel), mass(m), acceleration(Vector3D(0, 0, 0)) {}

    void update(double dt) {
        velocity = velocity + acceleration * dt;
        position = position + velocity * dt;
        acceleration = Vector3D(0, 0, 0); // Reset acceleration for next timestep
    }
};

std::vector<Body> bodies;

void computeForces() {
    // Reset total acceleration
    for (auto& body : bodies) {
        body.acceleration = Vector3D(0, 0, 0);
    }

    // Optimized triangular loop (pair-wise interaction)
    for (size_t i = 0; i < bodies.size(); ++i) {
        for (size_t j = i + 1; j < bodies.size(); ++j) {
            Vector3D r_vec = bodies[j].position - bodies[i].position;

            double r_squared = r_vec.x * r_vec.x + r_vec.y * r_vec.y + r_vec.z * r_vec.z;
            double distance_squared_soft = r_squared + (SOFTENING * SOFTENING);
            double distance_soft = std::sqrt(distance_squared_soft);

            // Scalar factor G / r_soft^3
            double factor_r_cubed = G / (distance_soft * distance_squared_soft);

            // Acceleration of i due to j: a_i = G * m_j / r_soft^3 * r_vec
            Vector3D acc_i = r_vec * (bodies[j].mass * factor_r_cubed);

            // Acceleration of j due to i: a_j = -G * m_i / r_soft^3 * r_vec
            Vector3D acc_j = r_vec * (-bodies[i].mass * factor_r_cubed);

            // Apply accelerations to both bodies
            bodies[i].acceleration += acc_i;
            bodies[j].acceleration += acc_j;
        }
    }
}

// --- GRAPHICS FUNCTIONS ---

// Helper function to check for shader compilation/linking errors
void checkShaderErrors(GLuint shader, std::string type) {
    GLint success;
    GLchar infoLog[1024];
    if (type != "PROGRAM") {
        glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
        if (!success) {
            glGetShaderInfoLog(shader, 1024, NULL, infoLog);
            std::cerr << "ERROR::SHADER_COMPILATION_ERROR of type: " << type << "\n" << infoLog << std::endl;
        }
    }
    else {
        glGetProgramiv(shader, GL_LINK_STATUS, &success);
        if (!success) {
            glGetProgramInfoLog(shader, 1024, NULL, infoLog);
            std::cerr << "ERROR::PROGRAM_LINKING_ERROR of type: " << type << "\n" << infoLog << std::endl;
        }
    }
}

// Loads, compiles, and links the vertex and fragment shaders
GLuint loadShaders() {
    // Read shader code from files
    std::string vertexCode;
    std::string fragmentCode;
    std::ifstream vShaderFile;
    std::ifstream fShaderFile;

    // Ensure ifstream objects can throw exceptions
    vShaderFile.exceptions(std::ifstream::failbit | std::ifstream::badbit);
    fShaderFile.exceptions(std::ifstream::failbit | std::ifstream::badbit);

    try {
        // Use the absolute path (CHANGE THIS FOR PORTABILITY)
        vShaderFile.open("C:/Users/edani/source/repos/N-Body Simulation/vertexShader.glsl");
        fShaderFile.open("C:/Users/edani/source/repos/N-Body Simulation/fragmentShader.glsl");
        std::stringstream vShaderStream, fShaderStream;

        // Read file buffer into streams
        vShaderStream << vShaderFile.rdbuf();
        fShaderStream << fShaderFile.rdbuf();

        // Close file handlers
        vShaderFile.close();
        fShaderFile.close();

        // Convert stream to string
        vertexCode = vShaderStream.str();
        fragmentCode = fShaderStream.str();
    }
    catch (std::ifstream::failure& e) {
        std::cerr << "ERROR::SHADER::FILE_NOT_SUCCESFULLY_READ: " << e.what() << std::endl;
        return 0;
    }

    const char* vShaderCode = vertexCode.c_str();
    const char* fShaderCode = fragmentCode.c_str();

    // Compile shaders
    unsigned int vertex, fragment;

    // Vertex Shader
    vertex = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vertex, 1, &vShaderCode, NULL);
    glCompileShader(vertex);
    checkShaderErrors(vertex, "VERTEX");

    // Fragment Shader
    fragment = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fragment, 1, &fShaderCode, NULL);
    glCompileShader(fragment);
    checkShaderErrors(fragment, "FRAGMENT");

    // Link shaders into a program
    GLuint ID = glCreateProgram();
    glAttachShader(ID, vertex);
    glAttachShader(ID, fragment);
    glLinkProgram(ID);
    checkShaderErrors(ID, "PROGRAM");

    // Delete the shaders as they're linked into our program now
    glDeleteShader(vertex);
    glDeleteShader(fragment);

    return ID;
}

// Generates the vertex, normal, and index data for a sphere mesh
void generateSphere(float radius, int segments) {
    // Clear previous data
    sphereVertices.clear();
    sphereNormals.clear();
    sphereIndices.clear();

    for (int y = 0; y <= segments; y++) {
        for (int x = 0; x <= segments; x++) {
            float xSegment = (float)x / (float)segments;
            float ySegment = (float)y / (float)segments;
            float xPos = std::cos(xSegment * 2.0f * glm::pi<float>()) * std::sin(ySegment * glm::pi<float>());
            float yPos = std::cos(ySegment * glm::pi<float>());
            float zPos = std::sin(xSegment * 2.0f * glm::pi<float>()) * std::sin(ySegment * glm::pi<float>());

            sphereVertices.push_back(xPos * radius);
            sphereVertices.push_back(yPos * radius);
            sphereVertices.push_back(zPos * radius);

            // Normals are the same as positions for a unit sphere
            sphereNormals.push_back(xPos);
            sphereNormals.push_back(yPos);
            sphereNormals.push_back(zPos);
        }
    }

    bool oddRow = false;
    for (int y = 0; y < segments; y++) {
        if (oddRow) { // TriangleStrip goes right-to-left
            for (int x = segments; x >= 0; x--) {
                sphereIndices.push_back((y + 1) * (segments + 1) + x);
                sphereIndices.push_back(y * (segments + 1) + x);
            }
        }
        else { // TriangleStrip goes left-to-right
            for (int x = 0; x <= segments; x++) {
                sphereIndices.push_back(y * (segments + 1) + x);
                sphereIndices.push_back((y + 1) * (segments + 1) + x);
            }
        }
        oddRow = !oddRow;
    }
    sphereIndexCount = sphereIndices.size();
}

// Configures the buffers to hold the sphere mesh data
void setupBuffers() {
    // Generate the sphere geometry once
    generateSphere(1.0f, 24); // Radius 1.0, 24 segments 

    glGenVertexArrays(1, &VAO);
    glGenBuffers(1, &sphereVBO);
    glGenBuffers(1, &sphereEBO); // Element Buffer Object for indices

    glBindVertexArray(VAO);

    // VBO for vertices and normals
    glBindBuffer(GL_ARRAY_BUFFER, sphereVBO);
    // Allocate space for both vertices and normals
    glBufferData(GL_ARRAY_BUFFER, sphereVertices.size() * sizeof(float) + sphereNormals.size() * sizeof(float), NULL, GL_STATIC_DRAW);
    glBufferSubData(GL_ARRAY_BUFFER, 0, sphereVertices.size() * sizeof(float), sphereVertices.data());
    glBufferSubData(GL_ARRAY_BUFFER, sphereVertices.size() * sizeof(float), sphereNormals.size() * sizeof(float), sphereNormals.data());

    // EBO for indices
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, sphereEBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sphereIndices.size() * sizeof(unsigned int), sphereIndices.data(), GL_STATIC_DRAW);

    // Vertex Positions (layout location 0)
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);

    // Normal Vectors (layout location 1, offset by vertex data size)
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)(sphereVertices.size() * sizeof(float)));
    glEnableVertexAttribArray(1);

    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);
}

// Sends projection, view, model, and lighting uniforms to the shader
void setViewProjection() {
    // 1. Projection Matrix (Perspective)
    glm::mat4 projection = glm::perspective(
        glm::radians(45.0f),
        (float)SCR_WIDTH / (float)SCR_HEIGHT,
        0.1f, 1000.0f
    );
    // 2. View Matrix (Camera)
    glm::mat4 view = glm::lookAt(cameraPos, cameraPos + cameraFront, cameraUp);

    // Send view and projection matrices (Model is sent per-body in main loop)
    glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "projection"), 1, GL_FALSE, glm::value_ptr(projection));
    glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "view"), 1, GL_FALSE, glm::value_ptr(view));

    // Send lighting uniforms
    glUniform3fv(glGetUniformLocation(shaderProgram, "lightPos"), 1, glm::value_ptr(lightPos));
    glUniform3fv(glGetUniformLocation(shaderProgram, "lightColor"), 1, glm::value_ptr(lightColor));
    glUniform3fv(glGetUniformLocation(shaderProgram, "viewPos"), 1, glm::value_ptr(cameraPos));
}


bool initGraphics() {
    // 1. Initialize GLFW
    if (!glfwInit()) {
        std::cerr << "Failed to initialize GLFW" << std::endl;
        return false;
    }
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    // 2. Create GLFW window
    window = glfwCreateWindow(SCR_WIDTH, SCR_HEIGHT, "N-Body 3D Simulation", NULL, NULL);
    if (window == NULL) {
        std::cerr << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
        return false;
    }
    glfwMakeContextCurrent(window);

    // 3. Initialize GLEW
    glewExperimental = GL_TRUE;
    if (glewInit() != GLEW_OK) {
        std::cerr << "Failed to initialize GLEW" << std::endl;
        return false;
    }

    // 4. Initial OpenGL configuration
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_PROGRAM_POINT_SIZE); // Keep this in case you revert to GL_POINTS

    return true;
}

// Mouse callback function for camera rotation
void mouseCallback(GLFWwindow* window, double xpos, double ypos) {
    if (firstMouse) {
        lastX = xpos;
        lastY = ypos;
        firstMouse = false;
    }

    float xoffset = xpos - lastX;
    float yoffset = lastY - ypos;
    lastX = xpos;
    lastY = ypos;

    xoffset *= MOUSE_SENSITIVITY;
    yoffset *= MOUSE_SENSITIVITY;

    yaw += xoffset;
    pitch += yoffset;

    // Constrain pitch
    if (pitch > 89.0f)
        pitch = 89.0f;
    if (pitch < -89.0f)
        pitch = -89.0f;

    // Calculate the new cameraFront vector from Euler angles
    glm::vec3 front;
    front.x = cos(glm::radians(yaw)) * cos(glm::radians(pitch));
    front.y = sin(glm::radians(pitch));
    front.z = sin(glm::radians(yaw)) * cos(glm::radians(pitch));
    cameraFront = glm::normalize(front);
}

void processInput(float deltaTime) {
    // Close window on ESC key press
    if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
        glfwSetWindowShouldClose(window, true);

    float velocity = CAMERA_SPEED * deltaTime;

    // Movement controls (WASD)
    if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS)
        cameraPos += velocity * cameraFront; // Move forward
    if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS)
        cameraPos -= velocity * cameraFront; // Move backward
    if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS)
        cameraPos -= glm::normalize(glm::cross(cameraFront, cameraUp)) * velocity; // Move left
    if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS)
        cameraPos += glm::normalize(glm::cross(cameraFront, cameraUp)) * velocity; // Move right
}

// --- MAIN FUNCTION ---

int main()
{
    // 1. Graphics Initialization
    if (!initGraphics()) return -1;

    // Configure mouse for camera control
    glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
    glfwSetCursorPosCallback(window, mouseCallback);

    // 2. Shaders and Buffers Setup (Loads sphere mesh data)
    shaderProgram = loadShaders();
    setupBuffers();

    // 3. N-Body System Initialization 
    // Uses G=1.0 and calculated orbital velocities for starting stability
    double sun_mass = 10000.0;
    double r1 = 100.0;
    double r2 = 150.0;
    double v1 = sqrt(G * sun_mass / r1);
    double v2 = sqrt(G * sun_mass / r2);

    bodies.push_back(Body(Vector3D(0, 0, 0), Vector3D(0, 0, 0), sun_mass)); // Sun 
    bodies.push_back(Body(Vector3D(r1, 0, 0), Vector3D(0, v1, 0), 10.0)); // Planet 1 (perfect circle in XY)
    bodies.push_back(Body(Vector3D(0, r2, 50), Vector3D(-v2, 0, 0), 15.0)); // Planet 2 (elliptical orbit in YZ plane)

    float lastFrame = 0.0f;
    float currentFrame;
    float deltaTime;

    // Main simulation loop
    while (!glfwWindowShouldClose(window)) {
        // Calculate DeltaTime 
        currentFrame = glfwGetTime();
        deltaTime = currentFrame - lastFrame;
        lastFrame = currentFrame;

        // 1. User Input (GLFW)
        processInput(deltaTime);

        // 2. Physics Logic
        computeForces();
        for (auto& body : bodies) {
            body.update(DELTA_TIME); // Use fixed DELTA_TIME for physics
        }

        // 3. OpenGL Rendering
        glClearColor(0.0f, 0.0f, 0.0f, 1.0f); // Black background
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        glUseProgram(shaderProgram);
        setViewProjection();

        // Bind the VAO containing the sphere mesh data
        glBindVertexArray(VAO);

        // Draw each body as a sphere
        for (const auto& body : bodies) {
            // Calculate model matrix (Translation and Scaling)
            glm::mat4 model = glm::mat4(1.0f);
            model = glm::translate(model, glm::vec3(body.position.x, body.position.y, body.position.z));

            // Set scale based on mass (Sun is larger than planets)
            float scale_factor = (body.mass > 1000) ? 10.0f : 2.0f;
            model = glm::scale(model, glm::vec3(scale_factor));

            // Send Model matrix to the shader
            glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "model"), 1, GL_FALSE, glm::value_ptr(model));

            // Set object color (e.g., yellow for sun, blue for planets)
            glm::vec3 objectColor = (body.mass > 1000) ? glm::vec3(1.0f, 1.0f, 0.0f) : glm::vec3(0.3f, 0.7f, 1.0f);
            glUniform3fv(glGetUniformLocation(shaderProgram, "objectColor"), 1, glm::value_ptr(objectColor));

            // Draw the sphere using indices (GL_TRIANGLE_STRIP from EBO)
            glDrawElements(GL_TRIANGLE_STRIP, sphereIndexCount, GL_UNSIGNED_INT, 0);
        }
        glBindVertexArray(0);

        // 4. Display the frame
        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    // Cleanup
    glDeleteVertexArrays(1, &VAO);
    glDeleteBuffers(1, &sphereVBO);
    glDeleteBuffers(1, &sphereEBO);
    glDeleteProgram(shaderProgram);
    glfwTerminate();
    return 0;
}