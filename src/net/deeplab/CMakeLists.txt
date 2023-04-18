target_include_directories(fgprs PUBLIC utils)
include_directories(utils)
add_subdirectory(utils)

target_sources(fgprs PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/deeplab.cpp)
target_sources(fgprs PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/deeplab_decoder.cpp)