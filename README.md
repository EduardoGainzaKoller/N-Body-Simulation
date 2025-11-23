# 🌌 N-Body Gravity Simulation

A real-time 3D simulation of gravitational interactions between celestial bodies using **C++** and **OpenGL**. This project implements an $O(N^2)$ N-Body algorithm where every object exerts a gravitational force on every other object, visualizing the results with dynamic lighting and camera controls.

![OpenGL](https://img.shields.io/badge/OpenGL-3.3-green) ![C++](https://img.shields.io/badge/C++-17-blue) ![License](https://img.shields.io/badge/License-MIT-lightgrey)

##  Features

* **N-Body Physics:** accurately calculates the gravitational force between all pairs of objects ($F = G \frac{m_1 m_2}{r^2}$).
* **3D Rendering:** Renders spheres with lighting using a custom shader pipeline.
* **Free Camera:** WASD movement and mouse look (First-person style) to explore the simulation.
* **Scalable:** Easy to add more bodies or change masses in the code.

##  Dependencies

To build and run this project, you need the following libraries installed and linked:

* **[GLFW](https://www.glfw.org/)**: For window management and input.
* **[GLEW](http://glew.sourceforge.net/)**: For managing OpenGL function pointers.
* **[GLM](https://github.com/g-truc/glm)**: For linear algebra (vectors and matrices).

##  Installation & Setup

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/your-username/n-body-simulation.git](https://github.com/your-username/n-body-simulation.git)
    ```

2.  **Configure Paths:**
    ⚠️ **Important:** Before running, open `main.cpp` and locate the `loadShaders()` function. Update the file paths for the shaders to match your local directory, or use relative paths:
    ```cpp
    // Change this absolute path to your local path
    vShaderFile.open("path/to/your/project/vertexShader.glsl");
    fShaderFile.open("path/to/your/project/fragmentShader.glsl");
    ```

3.  **Build:**
    * **Visual Studio:** Add the `.cpp` and `.h` files to your project. Link `opengl32.lib`, `glfw3.lib`, and `glew32.lib` in the project properties.
    * **GCC/MinGW:**
        ```bash
        g++ main.cpp -o simulation -lglfw3 -lglew32 -lopengl32 -lgdi32
        ```

##  Controls

| Key / Input | Action |
| :--- | :--- |
| **W, A, S, D** | Move the camera (Forward, Left, Back, Right) |
| **Mouse** | Look around (Yaw and Pitch) |
| **ESC** | Close the simulation |

##  Physics & Configuration

You can tweak the simulation stability and realism by modifying the constants at the top of `main.cpp`:

```cpp
// Physics Constants
const double G = 6.674e-11;  // Gravitational Constant (Real) or 1.0 (Simulation units)
const double DELTA_TIME = 0.01; // Time step per frame
```
