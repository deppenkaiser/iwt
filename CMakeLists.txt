cmake_minimum_required(VERSION 3.10)
project(iwt LANGUAGES C)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(PkgConfig REQUIRED)

add_compile_definitions(_POSIX_C_SOURCE=200809L)

set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -D_DEBUG")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

add_executable(iwt)
add_subdirectory(../../libraries/ocl ${CMAKE_BINARY_DIR}/ocl)
add_subdirectory(../../libraries/string ${CMAKE_BINARY_DIR}/string)
add_subdirectory(../../libraries/gui ${CMAKE_BINARY_DIR}/gui)
add_subdirectory(../../libraries/vector ${CMAKE_BINARY_DIR}/vector)

target_sources(iwt PRIVATE
	src/main.c
	src/gui.c
	src/iwt_kernel.c
	src/iwt_kernel_frozen.c
	src/iwt.c
	src/init.c
	src/dodecahedron.c
)

target_include_directories(iwt PRIVATE
	../../libraries
)

target_link_libraries(iwt
	ocl
	string
	gui
	vector
	m
)
