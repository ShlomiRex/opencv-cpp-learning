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
add_executable( hist_eq src/hist_eq.cpp )
add_executable( demo src/demo.cpp )
target_link_libraries( app ${OpenCV_LIBS} )
target_link_libraries( hist_eq ${OpenCV_LIBS} )
target_link_libraries( demo ${OpenCV_LIBS} )