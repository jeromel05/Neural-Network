cmake_minimum_required (VERSION 2.6)
project (projet_network)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Debug" CACHE STRING "" FORCE)
endif(NOT CMAKE_BUILD_TYPE)
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
SET(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -Wall -pedantic")
SET(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3")
option(test "Build tests." ON)

SET(SOURCE_FILES src/Network.cpp src/Utilities.cpp src/Network.hpp src/Utilities.hpp)

include_directories(${CMAKE_SOURCE_DIR}/extra/include)

add_executable(projet_network src/main.cpp ${SOURCE_FILES})

if (test)
  enable_testing()
  find_package(GTest)
  if (NOT GTEST_FOUND)
    link_directories(${CMAKE_SOURCE_DIR}/extra/lib)
    SET(GTEST_INCLUDE_DIRS ${CMAKE_SOURCE_DIR}/extra/include)
    SET(GTEST_BOTH_LIBRARIES libgtest.a libgtest_main.a)
  endif(NOT GTEST_FOUND)
  include_directories(${GTEST_INCLUDE_DIRS} ${CMAKE_SOURCE_DIR}/src)
  add_executable(network_unittest test/test_main.cpp ${SOURCE_FILES})
  target_link_libraries(network_unittest ${GTEST_BOTH_LIBRARIES} pthread)
  add_test(projet_network network_unittest)
endif(test)

find_package(Doxygen)
if (DOXYGEN_FOUND)
  add_custom_target(doc
    ${DOXYGEN_EXECUTABLE} ${CMAKE_SOURCE_DIR}/Doxyfile
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    COMMENT "Generating API documentation with Doxygen" VERBATIM)
endif(DOXYGEN_FOUND)