# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.2

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list

# Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /Applications/CMake.app/Contents/bin/cmake

# The command to remove a file.
RM = /Applications/CMake.app/Contents/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /Users/Terry/Desktop/OpenCVTests/RandomImage

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /Users/Terry/Desktop/OpenCVTests/RandomImage

# Include any dependencies generated for this target.
include CMakeFiles/RandomImage.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/RandomImage.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/RandomImage.dir/flags.make

CMakeFiles/RandomImage.dir/RandomImage.cpp.o: CMakeFiles/RandomImage.dir/flags.make
CMakeFiles/RandomImage.dir/RandomImage.cpp.o: RandomImage.cpp
	$(CMAKE_COMMAND) -E cmake_progress_report /Users/Terry/Desktop/OpenCVTests/RandomImage/CMakeFiles $(CMAKE_PROGRESS_1)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object CMakeFiles/RandomImage.dir/RandomImage.cpp.o"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/RandomImage.dir/RandomImage.cpp.o -c /Users/Terry/Desktop/OpenCVTests/RandomImage/RandomImage.cpp

CMakeFiles/RandomImage.dir/RandomImage.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/RandomImage.dir/RandomImage.cpp.i"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -E /Users/Terry/Desktop/OpenCVTests/RandomImage/RandomImage.cpp > CMakeFiles/RandomImage.dir/RandomImage.cpp.i

CMakeFiles/RandomImage.dir/RandomImage.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/RandomImage.dir/RandomImage.cpp.s"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -S /Users/Terry/Desktop/OpenCVTests/RandomImage/RandomImage.cpp -o CMakeFiles/RandomImage.dir/RandomImage.cpp.s

CMakeFiles/RandomImage.dir/RandomImage.cpp.o.requires:
.PHONY : CMakeFiles/RandomImage.dir/RandomImage.cpp.o.requires

CMakeFiles/RandomImage.dir/RandomImage.cpp.o.provides: CMakeFiles/RandomImage.dir/RandomImage.cpp.o.requires
	$(MAKE) -f CMakeFiles/RandomImage.dir/build.make CMakeFiles/RandomImage.dir/RandomImage.cpp.o.provides.build
.PHONY : CMakeFiles/RandomImage.dir/RandomImage.cpp.o.provides

CMakeFiles/RandomImage.dir/RandomImage.cpp.o.provides.build: CMakeFiles/RandomImage.dir/RandomImage.cpp.o

# Object files for target RandomImage
RandomImage_OBJECTS = \
"CMakeFiles/RandomImage.dir/RandomImage.cpp.o"

# External object files for target RandomImage
RandomImage_EXTERNAL_OBJECTS =

RandomImage: CMakeFiles/RandomImage.dir/RandomImage.cpp.o
RandomImage: CMakeFiles/RandomImage.dir/build.make
RandomImage: /usr/local/lib/libopencv_videostab.3.0.0.dylib
RandomImage: /usr/local/lib/libopencv_ts.a
RandomImage: /usr/local/lib/libopencv_superres.3.0.0.dylib
RandomImage: /usr/local/lib/libopencv_stitching.3.0.0.dylib
RandomImage: /usr/local/lib/libopencv_shape.3.0.0.dylib
RandomImage: /usr/local/lib/libopencv_photo.3.0.0.dylib
RandomImage: /usr/local/lib/libopencv_objdetect.3.0.0.dylib
RandomImage: /usr/local/lib/libopencv_calib3d.3.0.0.dylib
RandomImage: /usr/local/share/OpenCV/3rdparty/lib/libippicv.a
RandomImage: /usr/local/lib/libopencv_features2d.3.0.0.dylib
RandomImage: /usr/local/lib/libopencv_ml.3.0.0.dylib
RandomImage: /usr/local/lib/libopencv_highgui.3.0.0.dylib
RandomImage: /usr/local/lib/libopencv_videoio.3.0.0.dylib
RandomImage: /usr/local/lib/libopencv_imgcodecs.3.0.0.dylib
RandomImage: /usr/local/lib/libopencv_flann.3.0.0.dylib
RandomImage: /usr/local/lib/libopencv_video.3.0.0.dylib
RandomImage: /usr/local/lib/libopencv_imgproc.3.0.0.dylib
RandomImage: /usr/local/lib/libopencv_core.3.0.0.dylib
RandomImage: CMakeFiles/RandomImage.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --red --bold "Linking CXX executable RandomImage"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/RandomImage.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/RandomImage.dir/build: RandomImage
.PHONY : CMakeFiles/RandomImage.dir/build

CMakeFiles/RandomImage.dir/requires: CMakeFiles/RandomImage.dir/RandomImage.cpp.o.requires
.PHONY : CMakeFiles/RandomImage.dir/requires

CMakeFiles/RandomImage.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/RandomImage.dir/cmake_clean.cmake
.PHONY : CMakeFiles/RandomImage.dir/clean

CMakeFiles/RandomImage.dir/depend:
	cd /Users/Terry/Desktop/OpenCVTests/RandomImage && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /Users/Terry/Desktop/OpenCVTests/RandomImage /Users/Terry/Desktop/OpenCVTests/RandomImage /Users/Terry/Desktop/OpenCVTests/RandomImage /Users/Terry/Desktop/OpenCVTests/RandomImage /Users/Terry/Desktop/OpenCVTests/RandomImage/CMakeFiles/RandomImage.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/RandomImage.dir/depend
