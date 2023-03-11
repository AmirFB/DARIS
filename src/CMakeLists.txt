target_include_directories(fgprs PUBLIC dat)
include_directories(dat)
add_subdirectory(dat)

target_include_directories(fgprs PUBLIC net)
include_directories(net)
add_subdirectory(net)

target_include_directories(fgprs PUBLIC sch)
include_directories(sch)
add_subdirectory(sch)

target_include_directories(fgprs PUBLIC tests)
include_directories(tests)
add_subdirectory(tests)

target_include_directories(fgprs PUBLIC third_party)
include_directories(third_party)
add_subdirectory(third_party)