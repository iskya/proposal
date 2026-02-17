cmake_minimum_required(VERSION 3.16)
project(BeamScan)

# Find Geant4 (require vis and UI drivers for interactive mode)
find_package(Geant4 REQUIRED ui_all vis_all)
include(${Geant4_USE_FILE})

# Source files
file(GLOB sources ${PROJECT_SOURCE_DIR}/src/*.cc)
file(GLOB headers ${PROJECT_SOURCE_DIR}/include/*.hh)

# Include directories
include_directories(${PROJECT_SOURCE_DIR}/include)

# Build executable
add_executable(beamscan beamscan.cc ${sources} ${headers})
target_link_libraries(beamscan ${Geant4_LIBRARIES})

# Copy macros to build directory
file(GLOB MACRO_FILES ${PROJECT_SOURCE_DIR}/macros/*.mac)
foreach(_file ${MACRO_FILES})
    get_filename_component(_filename ${_file} NAME)
    configure_file(${_file} ${PROJECT_BINARY_DIR}/${_filename} COPYONLY)
endforeach()
