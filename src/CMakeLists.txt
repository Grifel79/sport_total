cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(sport_total_app)

find_package(OpenCV REQUIRED)
find_package(Torch REQUIRED)
include_directories( ${OpenCV_INCLUDE_DIRS} )

add_executable(sport_total_app sport_total_app.cpp)
target_link_libraries(sport_total_app "${TORCH_LIBRARIES}")
target_link_libraries(sport_total_app ${OpenCV_LIBS})
set_property(TARGET sport_total_app PROPERTY CXX_STANDARD 14)