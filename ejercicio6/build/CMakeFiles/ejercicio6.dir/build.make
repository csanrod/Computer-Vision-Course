# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.16

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


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
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /hdd/1cuatri/vision/vision_pr/ejercicio6

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /hdd/1cuatri/vision/vision_pr/ejercicio6/build

# Include any dependencies generated for this target.
include CMakeFiles/ejercicio6.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/ejercicio6.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/ejercicio6.dir/flags.make

CMakeFiles/ejercicio6.dir/view3D.cpp.o: CMakeFiles/ejercicio6.dir/flags.make
CMakeFiles/ejercicio6.dir/view3D.cpp.o: ../view3D.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/hdd/1cuatri/vision/vision_pr/ejercicio6/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/ejercicio6.dir/view3D.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/ejercicio6.dir/view3D.cpp.o -c /hdd/1cuatri/vision/vision_pr/ejercicio6/view3D.cpp

CMakeFiles/ejercicio6.dir/view3D.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/ejercicio6.dir/view3D.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /hdd/1cuatri/vision/vision_pr/ejercicio6/view3D.cpp > CMakeFiles/ejercicio6.dir/view3D.cpp.i

CMakeFiles/ejercicio6.dir/view3D.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/ejercicio6.dir/view3D.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /hdd/1cuatri/vision/vision_pr/ejercicio6/view3D.cpp -o CMakeFiles/ejercicio6.dir/view3D.cpp.s

# Object files for target ejercicio6
ejercicio6_OBJECTS = \
"CMakeFiles/ejercicio6.dir/view3D.cpp.o"

# External object files for target ejercicio6
ejercicio6_EXTERNAL_OBJECTS =

ejercicio6: CMakeFiles/ejercicio6.dir/view3D.cpp.o
ejercicio6: CMakeFiles/ejercicio6.dir/build.make
ejercicio6: /usr/lib/x86_64-linux-gnu/libpcl_visualization.so
ejercicio6: /usr/lib/x86_64-linux-gnu/libpcl_features.so
ejercicio6: /usr/lib/x86_64-linux-gnu/libpcl_filters.so
ejercicio6: /usr/lib/x86_64-linux-gnu/libboost_system.so
ejercicio6: /usr/lib/x86_64-linux-gnu/libboost_filesystem.so
ejercicio6: /usr/lib/x86_64-linux-gnu/libboost_date_time.so
ejercicio6: /usr/lib/x86_64-linux-gnu/libboost_iostreams.so
ejercicio6: /usr/lib/x86_64-linux-gnu/libboost_regex.so
ejercicio6: /usr/lib/x86_64-linux-gnu/libvtkChartsCore-7.1.so.7.1p.1
ejercicio6: /usr/lib/x86_64-linux-gnu/libvtkInfovisCore-7.1.so.7.1p.1
ejercicio6: /usr/lib/x86_64-linux-gnu/libfreetype.so
ejercicio6: /usr/lib/x86_64-linux-gnu/libz.so
ejercicio6: /usr/lib/x86_64-linux-gnu/libjpeg.so
ejercicio6: /usr/lib/x86_64-linux-gnu/libpng.so
ejercicio6: /usr/lib/x86_64-linux-gnu/libtiff.so
ejercicio6: /usr/lib/x86_64-linux-gnu/libexpat.so
ejercicio6: /usr/lib/x86_64-linux-gnu/libvtkIOGeometry-7.1.so.7.1p.1
ejercicio6: /usr/lib/x86_64-linux-gnu/libvtkIOLegacy-7.1.so.7.1p.1
ejercicio6: /usr/lib/x86_64-linux-gnu/libvtkIOPLY-7.1.so.7.1p.1
ejercicio6: /usr/lib/x86_64-linux-gnu/libvtkRenderingLOD-7.1.so.7.1p.1
ejercicio6: /usr/lib/x86_64-linux-gnu/libvtkViewsContext2D-7.1.so.7.1p.1
ejercicio6: /usr/lib/x86_64-linux-gnu/libvtkViewsCore-7.1.so.7.1p.1
ejercicio6: /usr/lib/x86_64-linux-gnu/libvtkRenderingContextOpenGL2-7.1.so.7.1p.1
ejercicio6: /usr/lib/x86_64-linux-gnu/libvtkRenderingOpenGL2-7.1.so.7.1p.1
ejercicio6: /usr/lib/x86_64-linux-gnu/libflann_cpp.so
ejercicio6: /usr/lib/x86_64-linux-gnu/libpcl_io.so
ejercicio6: /usr/lib/x86_64-linux-gnu/libpcl_sample_consensus.so
ejercicio6: /usr/lib/x86_64-linux-gnu/libpcl_search.so
ejercicio6: /usr/lib/x86_64-linux-gnu/libpcl_octree.so
ejercicio6: /usr/lib/x86_64-linux-gnu/libpcl_kdtree.so
ejercicio6: /usr/lib/x86_64-linux-gnu/libpcl_common.so
ejercicio6: /usr/lib/x86_64-linux-gnu/libvtkInteractionWidgets-7.1.so.7.1p.1
ejercicio6: /usr/lib/x86_64-linux-gnu/libvtkFiltersModeling-7.1.so.7.1p.1
ejercicio6: /usr/lib/x86_64-linux-gnu/libvtkInteractionStyle-7.1.so.7.1p.1
ejercicio6: /usr/lib/x86_64-linux-gnu/libvtkFiltersExtraction-7.1.so.7.1p.1
ejercicio6: /usr/lib/x86_64-linux-gnu/libvtkFiltersStatistics-7.1.so.7.1p.1
ejercicio6: /usr/lib/x86_64-linux-gnu/libvtkImagingFourier-7.1.so.7.1p.1
ejercicio6: /usr/lib/x86_64-linux-gnu/libvtkalglib-7.1.so.7.1p.1
ejercicio6: /usr/lib/x86_64-linux-gnu/libvtkFiltersHybrid-7.1.so.7.1p.1
ejercicio6: /usr/lib/x86_64-linux-gnu/libvtkImagingGeneral-7.1.so.7.1p.1
ejercicio6: /usr/lib/x86_64-linux-gnu/libvtkImagingSources-7.1.so.7.1p.1
ejercicio6: /usr/lib/x86_64-linux-gnu/libvtkImagingHybrid-7.1.so.7.1p.1
ejercicio6: /usr/lib/x86_64-linux-gnu/libvtkRenderingAnnotation-7.1.so.7.1p.1
ejercicio6: /usr/lib/x86_64-linux-gnu/libvtkImagingColor-7.1.so.7.1p.1
ejercicio6: /usr/lib/x86_64-linux-gnu/libvtkRenderingVolume-7.1.so.7.1p.1
ejercicio6: /usr/lib/x86_64-linux-gnu/libvtkIOXML-7.1.so.7.1p.1
ejercicio6: /usr/lib/x86_64-linux-gnu/libvtkIOXMLParser-7.1.so.7.1p.1
ejercicio6: /usr/lib/x86_64-linux-gnu/libvtkIOCore-7.1.so.7.1p.1
ejercicio6: /usr/lib/x86_64-linux-gnu/libvtkRenderingContext2D-7.1.so.7.1p.1
ejercicio6: /usr/lib/x86_64-linux-gnu/libvtkRenderingFreeType-7.1.so.7.1p.1
ejercicio6: /usr/lib/x86_64-linux-gnu/libfreetype.so
ejercicio6: /usr/lib/x86_64-linux-gnu/libvtkImagingCore-7.1.so.7.1p.1
ejercicio6: /usr/lib/x86_64-linux-gnu/libvtkRenderingCore-7.1.so.7.1p.1
ejercicio6: /usr/lib/x86_64-linux-gnu/libvtkCommonColor-7.1.so.7.1p.1
ejercicio6: /usr/lib/x86_64-linux-gnu/libvtkFiltersGeometry-7.1.so.7.1p.1
ejercicio6: /usr/lib/x86_64-linux-gnu/libvtkFiltersSources-7.1.so.7.1p.1
ejercicio6: /usr/lib/x86_64-linux-gnu/libvtkFiltersGeneral-7.1.so.7.1p.1
ejercicio6: /usr/lib/x86_64-linux-gnu/libvtkCommonComputationalGeometry-7.1.so.7.1p.1
ejercicio6: /usr/lib/x86_64-linux-gnu/libvtkFiltersCore-7.1.so.7.1p.1
ejercicio6: /usr/lib/x86_64-linux-gnu/libvtkIOImage-7.1.so.7.1p.1
ejercicio6: /usr/lib/x86_64-linux-gnu/libvtkCommonExecutionModel-7.1.so.7.1p.1
ejercicio6: /usr/lib/x86_64-linux-gnu/libvtkCommonDataModel-7.1.so.7.1p.1
ejercicio6: /usr/lib/x86_64-linux-gnu/libvtkCommonTransforms-7.1.so.7.1p.1
ejercicio6: /usr/lib/x86_64-linux-gnu/libvtkCommonMisc-7.1.so.7.1p.1
ejercicio6: /usr/lib/x86_64-linux-gnu/libvtkCommonMath-7.1.so.7.1p.1
ejercicio6: /usr/lib/x86_64-linux-gnu/libvtkCommonSystem-7.1.so.7.1p.1
ejercicio6: /usr/lib/x86_64-linux-gnu/libvtkCommonCore-7.1.so.7.1p.1
ejercicio6: /usr/lib/x86_64-linux-gnu/libvtksys-7.1.so.7.1p.1
ejercicio6: /usr/lib/x86_64-linux-gnu/libvtkDICOMParser-7.1.so.7.1p.1
ejercicio6: /usr/lib/x86_64-linux-gnu/libvtkmetaio-7.1.so.7.1p.1
ejercicio6: /usr/lib/x86_64-linux-gnu/libz.so
ejercicio6: /usr/lib/x86_64-linux-gnu/libGLEW.so
ejercicio6: /usr/lib/x86_64-linux-gnu/libSM.so
ejercicio6: /usr/lib/x86_64-linux-gnu/libICE.so
ejercicio6: /usr/lib/x86_64-linux-gnu/libX11.so
ejercicio6: /usr/lib/x86_64-linux-gnu/libXext.so
ejercicio6: /usr/lib/x86_64-linux-gnu/libXt.so
ejercicio6: CMakeFiles/ejercicio6.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/hdd/1cuatri/vision/vision_pr/ejercicio6/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable ejercicio6"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/ejercicio6.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/ejercicio6.dir/build: ejercicio6

.PHONY : CMakeFiles/ejercicio6.dir/build

CMakeFiles/ejercicio6.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/ejercicio6.dir/cmake_clean.cmake
.PHONY : CMakeFiles/ejercicio6.dir/clean

CMakeFiles/ejercicio6.dir/depend:
	cd /hdd/1cuatri/vision/vision_pr/ejercicio6/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /hdd/1cuatri/vision/vision_pr/ejercicio6 /hdd/1cuatri/vision/vision_pr/ejercicio6 /hdd/1cuatri/vision/vision_pr/ejercicio6/build /hdd/1cuatri/vision/vision_pr/ejercicio6/build /hdd/1cuatri/vision/vision_pr/ejercicio6/build/CMakeFiles/ejercicio6.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/ejercicio6.dir/depend

