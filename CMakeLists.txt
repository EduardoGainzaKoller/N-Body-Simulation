
cmake_minimum_required(VERSION 3.31)

project ("N-Body Simulation")
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Incluye la toolchain de vcpkg
if(NOT DEFINED CMAKE_TOOLCHAIN_FILE)
    set(CMAKE_TOOLCHAIN_FILE "C:/tools/vcpkg/scripts/buildsystems/vcpkg.cmake" CACHE STRING "")
endif()

include("C:/tools/vcpkg/scripts/buildsystems/vcpkg.cmake")

# Busca las librerías instaladas por vcpkg
# Sin CONFIG, para que CMake use los archivos de vcpkg
find_package(glfw3 REQUIRED)
find_package(GLEW REQUIRED)
find_package(glm REQUIRED)

# Crea el ejecutable
add_executable (CMakeTarget "N-Body Simulation.cpp" "N-Body Simulation.h")

# Linkea las librerías usando los targets de vcpkg
target_link_libraries(CMakeTarget PRIVATE glfw GLEW::GLEW glm::glm)
