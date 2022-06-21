cmake_minimum_required(VERSION 3.5.1 FATAL_ERROR)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_BUILD_TYPE Release)

# FOR MAIN PROJECT
project(planeFitting)

#set(Boost_DEBUG ON)


find_package(OpenCV REQUIRED)
find_package(PCL REQUIRED)
find_package(Boost 1.73.0 REQUIRED)
find_package(Eigen3 3.3 REQUIRED NO_MODULE)
find_package(GLM REQUIRED)

if(NOT GLM_FOUND)
        message(Error "GLM not found")
endif(NOT GLM_FOUND)

IF (Boost_FOUND)
    INCLUDE_DIRECTORIES(${Boost_INCLUDE_DIR})
    ADD_DEFINITIONS( "-DHAS_BOOST" )
ENDIF()

variable_watch(CMAKE_BUILD_TYPE)
set(MAIN_SOURCE main.cpp plyReader.cpp plyReader.h)

message("CMAKE TYPE:  ${CMAKE_BUILD_TYPE}" )
include_directories(${PCL_INCLUDE_DIRS})
include_directories("C:/Program Files/PCL 1.11.0/3rdParty/Boost/include")
include_directories(${GLM_INCLUDE_DIRS})
link_directories(${GLM_LIBRARY_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
link_directories("C:/Program Files/PCL 1.11.0/3rdParty/Boost/lib")
add_definitions(${GLM_DEFINITIONS})
add_definitions(${PCL_DEFINITIONS})
add_executable(${PROJECT_NAME} ${MAIN_SOURCE})
target_link_libraries(${PROJECT_NAME} ${PCL_LIBRARIES} ${OpenCV_LIBS})
target_link_libraries(${PROJECT_NAME} Eigen3::Eigen)
message("=========================================")
message("Project: ${PROJECT_NAME} COMPILED WITH CMAKE " ${CMAKE_VERSION})
message("=========================================")
