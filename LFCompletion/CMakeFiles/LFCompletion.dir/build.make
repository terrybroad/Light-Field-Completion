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
CMAKE_SOURCE_DIR = /Users/Terry/Desktop/OpenCVTests/LFCompletion

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /Users/Terry/Desktop/OpenCVTests/LFCompletion

# Include any dependencies generated for this target.
include CMakeFiles/LFCompletion.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/LFCompletion.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/LFCompletion.dir/flags.make

CMakeFiles/LFCompletion.dir/LFCompletion.cpp.o: CMakeFiles/LFCompletion.dir/flags.make
CMakeFiles/LFCompletion.dir/LFCompletion.cpp.o: LFCompletion.cpp
	$(CMAKE_COMMAND) -E cmake_progress_report /Users/Terry/Desktop/OpenCVTests/LFCompletion/CMakeFiles $(CMAKE_PROGRESS_1)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object CMakeFiles/LFCompletion.dir/LFCompletion.cpp.o"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/LFCompletion.dir/LFCompletion.cpp.o -c /Users/Terry/Desktop/OpenCVTests/LFCompletion/LFCompletion.cpp

CMakeFiles/LFCompletion.dir/LFCompletion.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/LFCompletion.dir/LFCompletion.cpp.i"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -E /Users/Terry/Desktop/OpenCVTests/LFCompletion/LFCompletion.cpp > CMakeFiles/LFCompletion.dir/LFCompletion.cpp.i

CMakeFiles/LFCompletion.dir/LFCompletion.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/LFCompletion.dir/LFCompletion.cpp.s"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -S /Users/Terry/Desktop/OpenCVTests/LFCompletion/LFCompletion.cpp -o CMakeFiles/LFCompletion.dir/LFCompletion.cpp.s

CMakeFiles/LFCompletion.dir/LFCompletion.cpp.o.requires:
.PHONY : CMakeFiles/LFCompletion.dir/LFCompletion.cpp.o.requires

CMakeFiles/LFCompletion.dir/LFCompletion.cpp.o.provides: CMakeFiles/LFCompletion.dir/LFCompletion.cpp.o.requires
	$(MAKE) -f CMakeFiles/LFCompletion.dir/build.make CMakeFiles/LFCompletion.dir/LFCompletion.cpp.o.provides.build
.PHONY : CMakeFiles/LFCompletion.dir/LFCompletion.cpp.o.provides

CMakeFiles/LFCompletion.dir/LFCompletion.cpp.o.provides.build: CMakeFiles/LFCompletion.dir/LFCompletion.cpp.o

# Object files for target LFCompletion
LFCompletion_OBJECTS = \
"CMakeFiles/LFCompletion.dir/LFCompletion.cpp.o"

# External object files for target LFCompletion
LFCompletion_EXTERNAL_OBJECTS =

LFCompletion: CMakeFiles/LFCompletion.dir/LFCompletion.cpp.o
LFCompletion: CMakeFiles/LFCompletion.dir/build.make
LFCompletion: /usr/local/lib/libopencv_videostab.3.0.0.dylib
LFCompletion: /usr/local/lib/libopencv_ts.a
LFCompletion: /usr/local/lib/libopencv_superres.3.0.0.dylib
LFCompletion: /usr/local/lib/libopencv_stitching.3.0.0.dylib
LFCompletion: /usr/local/lib/libopencv_shape.3.0.0.dylib
LFCompletion: /usr/local/lib/libopencv_photo.3.0.0.dylib
LFCompletion: /usr/local/lib/libopencv_objdetect.3.0.0.dylib
LFCompletion: /usr/local/lib/libopencv_calib3d.3.0.0.dylib
LFCompletion: /usr/local/share/OpenCV/3rdparty/lib/libippicv.a
LFCompletion: /usr/local/lib/libopencv_features2d.3.0.0.dylib
LFCompletion: /usr/local/lib/libopencv_ml.3.0.0.dylib
LFCompletion: /usr/local/lib/libopencv_highgui.3.0.0.dylib
LFCompletion: /usr/local/lib/libopencv_videoio.3.0.0.dylib
LFCompletion: /usr/local/lib/libopencv_imgcodecs.3.0.0.dylib
LFCompletion: /usr/local/lib/libopencv_flann.3.0.0.dylib
LFCompletion: /usr/local/lib/libopencv_video.3.0.0.dylib
LFCompletion: /usr/local/lib/libopencv_imgproc.3.0.0.dylib
LFCompletion: /usr/local/lib/libopencv_core.3.0.0.dylib
LFCompletion: CMakeFiles/LFCompletion.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --red --bold "Linking CXX executable LFCompletion"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/LFCompletion.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/LFCompletion.dir/build: LFCompletion
.PHONY : CMakeFiles/LFCompletion.dir/build

CMakeFiles/LFCompletion.dir/requires: CMakeFiles/LFCompletion.dir/LFCompletion.cpp.o.requires
.PHONY : CMakeFiles/LFCompletion.dir/requires

CMakeFiles/LFCompletion.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/LFCompletion.dir/cmake_clean.cmake
.PHONY : CMakeFiles/LFCompletion.dir/clean

CMakeFiles/LFCompletion.dir/depend:
	cd /Users/Terry/Desktop/OpenCVTests/LFCompletion && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /Users/Terry/Desktop/OpenCVTests/LFCompletion /Users/Terry/Desktop/OpenCVTests/LFCompletion /Users/Terry/Desktop/OpenCVTests/LFCompletion /Users/Terry/Desktop/OpenCVTests/LFCompletion /Users/Terry/Desktop/OpenCVTests/LFCompletion/CMakeFiles/LFCompletion.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/LFCompletion.dir/depend

