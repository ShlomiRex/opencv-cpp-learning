cmake_minimum_required(VERSION 3.0)

# specify the C++ standard
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# debug
set(CMAKE_BUILD_TYPE Debug)

project( DisplayImage )
find_package( OpenCV REQUIRED )
INCLUDE_DIRECTORIES( include/ )
add_executable( app src/main.cpp )
add_executable( gradients src/gradients.cpp )
target_link_libraries( app ${OpenCV_LIBS} )
target_link_libraries( gradients ${OpenCV_LIBS} )
