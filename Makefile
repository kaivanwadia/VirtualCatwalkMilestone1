#############################################################################
# Makefile for building: catwalk
# Generated by qmake (2.01a) (Qt 4.8.1) on: Sun Nov 23 19:09:18 2014
# Project:  catwalk.pro
# Template: app
# Command: /usr/bin/qmake-qt4 -spec /usr/share/qt4/mkspecs/linux-g++ -o Makefile catwalk.pro
#############################################################################

####### Compiler, tools and options

CC            = gcc
CXX           = g++
DEFINES       = -DQT_NO_DEBUG -DQT_OPENGL_LIB -DQT_GUI_LIB -DQT_CORE_LIB -DQT_SHARED
CFLAGS        = -pipe -O2 -Wall -W -D_REENTRANT $(DEFINES)
CXXFLAGS      = -pipe -g -O2 -O2 -Wall -W -D_REENTRANT $(DEFINES)
INCPATH       = -I/usr/share/qt4/mkspecs/linux-g++ -I. -I/usr/include/qt4/QtCore -I/usr/include/qt4/QtGui -I/usr/include/qt4/QtOpenGL -I/usr/include/qt4 -Ieigen_3.2.2 -ISOIL/src -I/usr/X11R6/include -I. -I.
LINK          = g++
LFLAGS        = -Wl,-O1
LIBS          = $(SUBLIBS)  -L/usr/lib/x86_64-linux-gnu -L/usr/X11R6/lib -lGLU -L/v/filer4b/v20q001/kwadia/PhysicalSimCS395T/VCMilestone1/VirtualCatwalkMilestone1/SOIL/lib -lSOIL -lQtOpenGL -lQtGui -lQtCore -lGL -lpthread 
AR            = ar cqs
RANLIB        = 
QMAKE         = /usr/bin/qmake-qt4
TAR           = tar -cf
COMPRESS      = gzip -9f
COPY          = cp -f
SED           = sed
COPY_FILE     = $(COPY)
COPY_DIR      = $(COPY) -r
STRIP         = strip
INSTALL_FILE  = install -m 644 -p
INSTALL_DIR   = $(COPY_DIR)
INSTALL_PROGRAM = install -m 755 -p
DEL_FILE      = rm -f
SYMLINK       = ln -f -s
DEL_DIR       = rmdir
MOVE          = mv -f
CHK_DIR_EXISTS= test -d
MKDIR         = mkdir -p

####### Output directory

OBJECTS_DIR   = ./

####### Files

SOURCES       = main.cpp \
		mainwindow.cpp \
		glpanel.cpp \
		controller.cpp \
		simulation.cpp \
		simparameters.cpp \
		mesh.cpp \
		rigidbodytemplate.cpp \
		rigidbodyinstance.cpp \
		vectormath.cpp \
		camera.cpp \
		rotator.cpp \
		signeddistancefield.cpp \
		collisions/CTCD.cpp \
		cloth.cpp moc_mainwindow.cpp \
		moc_glpanel.cpp \
		moc_controller.cpp
OBJECTS       = main.o \
		mainwindow.o \
		glpanel.o \
		controller.o \
		simulation.o \
		simparameters.o \
		mesh.o \
		rigidbodytemplate.o \
		rigidbodyinstance.o \
		vectormath.o \
		camera.o \
		rotator.o \
		signeddistancefield.o \
		CTCD.o \
		cloth.o \
		moc_mainwindow.o \
		moc_glpanel.o \
		moc_controller.o
DIST          = /usr/share/qt4/mkspecs/common/unix.conf \
		/usr/share/qt4/mkspecs/common/linux.conf \
		/usr/share/qt4/mkspecs/common/gcc-base.conf \
		/usr/share/qt4/mkspecs/common/gcc-base-unix.conf \
		/usr/share/qt4/mkspecs/common/g++-base.conf \
		/usr/share/qt4/mkspecs/common/g++-unix.conf \
		/usr/share/qt4/mkspecs/qconfig.pri \
		/usr/share/qt4/mkspecs/features/qt_functions.prf \
		/usr/share/qt4/mkspecs/features/qt_config.prf \
		/usr/share/qt4/mkspecs/features/exclusive_builds.prf \
		/usr/share/qt4/mkspecs/features/default_pre.prf \
		/usr/share/qt4/mkspecs/features/release.prf \
		/usr/share/qt4/mkspecs/features/default_post.prf \
		/usr/share/qt4/mkspecs/features/unix/gdb_dwarf_index.prf \
		/usr/share/qt4/mkspecs/features/warn_on.prf \
		/usr/share/qt4/mkspecs/features/qt.prf \
		/usr/share/qt4/mkspecs/features/unix/opengl.prf \
		/usr/share/qt4/mkspecs/features/unix/thread.prf \
		/usr/share/qt4/mkspecs/features/moc.prf \
		/usr/share/qt4/mkspecs/features/resources.prf \
		/usr/share/qt4/mkspecs/features/uic.prf \
		/usr/share/qt4/mkspecs/features/yacc.prf \
		/usr/share/qt4/mkspecs/features/lex.prf \
		/usr/share/qt4/mkspecs/features/include_source_dir.prf \
		catwalk.pro
QMAKE_TARGET  = catwalk
DESTDIR       = 
TARGET        = catwalk

first: all
####### Implicit rules

.SUFFIXES: .o .c .cpp .cc .cxx .C

.cpp.o:
	$(CXX) -c $(CXXFLAGS) $(INCPATH) -o "$@" "$<"

.cc.o:
	$(CXX) -c $(CXXFLAGS) $(INCPATH) -o "$@" "$<"

.cxx.o:
	$(CXX) -c $(CXXFLAGS) $(INCPATH) -o "$@" "$<"

.C.o:
	$(CXX) -c $(CXXFLAGS) $(INCPATH) -o "$@" "$<"

.c.o:
	$(CC) -c $(CFLAGS) $(INCPATH) -o "$@" "$<"

####### Build rules

all: Makefile $(TARGET)

$(TARGET): libSOIL ui_mainwindow.h $(OBJECTS)  
	$(LINK) $(LFLAGS) -o $(TARGET) $(OBJECTS) $(OBJCOMP) $(LIBS)

Makefile: catwalk.pro  /usr/share/qt4/mkspecs/linux-g++/qmake.conf /usr/share/qt4/mkspecs/common/unix.conf \
		/usr/share/qt4/mkspecs/common/linux.conf \
		/usr/share/qt4/mkspecs/common/gcc-base.conf \
		/usr/share/qt4/mkspecs/common/gcc-base-unix.conf \
		/usr/share/qt4/mkspecs/common/g++-base.conf \
		/usr/share/qt4/mkspecs/common/g++-unix.conf \
		/usr/share/qt4/mkspecs/qconfig.pri \
		/usr/share/qt4/mkspecs/features/qt_functions.prf \
		/usr/share/qt4/mkspecs/features/qt_config.prf \
		/usr/share/qt4/mkspecs/features/exclusive_builds.prf \
		/usr/share/qt4/mkspecs/features/default_pre.prf \
		/usr/share/qt4/mkspecs/features/release.prf \
		/usr/share/qt4/mkspecs/features/default_post.prf \
		/usr/share/qt4/mkspecs/features/unix/gdb_dwarf_index.prf \
		/usr/share/qt4/mkspecs/features/warn_on.prf \
		/usr/share/qt4/mkspecs/features/qt.prf \
		/usr/share/qt4/mkspecs/features/unix/opengl.prf \
		/usr/share/qt4/mkspecs/features/unix/thread.prf \
		/usr/share/qt4/mkspecs/features/moc.prf \
		/usr/share/qt4/mkspecs/features/resources.prf \
		/usr/share/qt4/mkspecs/features/uic.prf \
		/usr/share/qt4/mkspecs/features/yacc.prf \
		/usr/share/qt4/mkspecs/features/lex.prf \
		/usr/share/qt4/mkspecs/features/include_source_dir.prf \
		/usr/lib/x86_64-linux-gnu/libQtOpenGL.prl \
		/usr/lib/x86_64-linux-gnu/libQtGui.prl \
		/usr/lib/x86_64-linux-gnu/libQtCore.prl
	$(QMAKE) -spec /usr/share/qt4/mkspecs/linux-g++ -o Makefile catwalk.pro
/usr/share/qt4/mkspecs/common/unix.conf:
/usr/share/qt4/mkspecs/common/linux.conf:
/usr/share/qt4/mkspecs/common/gcc-base.conf:
/usr/share/qt4/mkspecs/common/gcc-base-unix.conf:
/usr/share/qt4/mkspecs/common/g++-base.conf:
/usr/share/qt4/mkspecs/common/g++-unix.conf:
/usr/share/qt4/mkspecs/qconfig.pri:
/usr/share/qt4/mkspecs/features/qt_functions.prf:
/usr/share/qt4/mkspecs/features/qt_config.prf:
/usr/share/qt4/mkspecs/features/exclusive_builds.prf:
/usr/share/qt4/mkspecs/features/default_pre.prf:
/usr/share/qt4/mkspecs/features/release.prf:
/usr/share/qt4/mkspecs/features/default_post.prf:
/usr/share/qt4/mkspecs/features/unix/gdb_dwarf_index.prf:
/usr/share/qt4/mkspecs/features/warn_on.prf:
/usr/share/qt4/mkspecs/features/qt.prf:
/usr/share/qt4/mkspecs/features/unix/opengl.prf:
/usr/share/qt4/mkspecs/features/unix/thread.prf:
/usr/share/qt4/mkspecs/features/moc.prf:
/usr/share/qt4/mkspecs/features/resources.prf:
/usr/share/qt4/mkspecs/features/uic.prf:
/usr/share/qt4/mkspecs/features/yacc.prf:
/usr/share/qt4/mkspecs/features/lex.prf:
/usr/share/qt4/mkspecs/features/include_source_dir.prf:
/usr/lib/x86_64-linux-gnu/libQtOpenGL.prl:
/usr/lib/x86_64-linux-gnu/libQtGui.prl:
/usr/lib/x86_64-linux-gnu/libQtCore.prl:
qmake:  FORCE
	@$(QMAKE) -spec /usr/share/qt4/mkspecs/linux-g++ -o Makefile catwalk.pro

dist: 
	@$(CHK_DIR_EXISTS) .tmp/catwalk1.0.0 || $(MKDIR) .tmp/catwalk1.0.0 
	$(COPY_FILE) --parents $(SOURCES) $(DIST) .tmp/catwalk1.0.0/ && $(COPY_FILE) --parents mainwindow.h glpanel.h controller.h simulation.h simparameters.h mesh.h rigidbodytemplate.h rigidbodyinstance.h vectormath.h camera.h rotator.h signeddistancefield.h collisions/Distance.h collisions/CTCD.h cloth.h .tmp/catwalk1.0.0/ && $(COPY_FILE) --parents main.cpp mainwindow.cpp glpanel.cpp controller.cpp simulation.cpp simparameters.cpp mesh.cpp rigidbodytemplate.cpp rigidbodyinstance.cpp vectormath.cpp camera.cpp rotator.cpp signeddistancefield.cpp collisions/CTCD.cpp cloth.cpp .tmp/catwalk1.0.0/ && $(COPY_FILE) --parents mainwindow.ui .tmp/catwalk1.0.0/ && (cd `dirname .tmp/catwalk1.0.0` && $(TAR) catwalk1.0.0.tar catwalk1.0.0 && $(COMPRESS) catwalk1.0.0.tar) && $(MOVE) `dirname .tmp/catwalk1.0.0`/catwalk1.0.0.tar.gz . && $(DEL_FILE) -r .tmp/catwalk1.0.0


clean:compiler_clean 
	-$(DEL_FILE) $(OBJECTS)
	-$(DEL_FILE) *~ core *.core


####### Sub-libraries

distclean: clean
	-$(DEL_FILE) $(TARGET) 
	-$(DEL_FILE) Makefile


libSOIL:
	cd /v/filer4b/v20q001/kwadia/PhysicalSimCS395T/VCMilestone1/VirtualCatwalkMilestone1/SOIL/projects/makefile/ && make

check: first

mocclean: compiler_moc_header_clean compiler_moc_source_clean

mocables: compiler_moc_header_make_all compiler_moc_source_make_all

compiler_moc_header_make_all: moc_mainwindow.cpp moc_glpanel.cpp moc_controller.cpp
compiler_moc_header_clean:
	-$(DEL_FILE) moc_mainwindow.cpp moc_glpanel.cpp moc_controller.cpp
moc_mainwindow.cpp: mainwindow.h
	/usr/bin/moc-qt4 $(DEFINES) $(INCPATH) mainwindow.h -o moc_mainwindow.cpp

moc_glpanel.cpp: camera.h \
		rotator.h \
		glpanel.h
	/usr/bin/moc-qt4 $(DEFINES) $(INCPATH) glpanel.h -o moc_glpanel.cpp

moc_controller.cpp: simparameters.h \
		controller.h
	/usr/bin/moc-qt4 $(DEFINES) $(INCPATH) controller.h -o moc_controller.cpp

compiler_rcc_make_all:
compiler_rcc_clean:
compiler_image_collection_make_all: qmake_image_collection.cpp
compiler_image_collection_clean:
	-$(DEL_FILE) qmake_image_collection.cpp
compiler_moc_source_make_all:
compiler_moc_source_clean:
compiler_uic_make_all: ui_mainwindow.h
compiler_uic_clean:
	-$(DEL_FILE) ui_mainwindow.h
ui_mainwindow.h: mainwindow.ui \
		glpanel.h \
		camera.h \
		rotator.h
	/usr/bin/uic-qt4 mainwindow.ui -o ui_mainwindow.h

compiler_yacc_decl_make_all:
compiler_yacc_decl_clean:
compiler_yacc_impl_make_all:
compiler_yacc_impl_clean:
compiler_lex_make_all:
compiler_lex_clean:
compiler_clean: compiler_moc_header_clean compiler_uic_clean 

####### Compile

main.o: main.cpp mainwindow.h \
		controller.h \
		simparameters.h
	$(CXX) -c $(CXXFLAGS) $(INCPATH) -o main.o main.cpp

mainwindow.o: mainwindow.cpp mainwindow.h \
		ui_mainwindow.h \
		glpanel.h \
		camera.h \
		rotator.h \
		simparameters.h \
		controller.h
	$(CXX) -c $(CXXFLAGS) $(INCPATH) -o mainwindow.o mainwindow.cpp

glpanel.o: glpanel.cpp glpanel.h \
		camera.h \
		rotator.h \
		controller.h \
		simparameters.h
	$(CXX) -c $(CXXFLAGS) $(INCPATH) -o glpanel.o glpanel.cpp

controller.o: controller.cpp controller.h \
		simparameters.h \
		mainwindow.h \
		simulation.h
	$(CXX) -c $(CXXFLAGS) $(INCPATH) -o controller.o controller.cpp

simulation.o: simulation.cpp simparameters.h \
		rigidbodytemplate.h \
		rigidbodyinstance.h \
		cloth.h \
		mesh.h \
		vectormath.h \
		signeddistancefield.h
	$(CXX) -c $(CXXFLAGS) $(INCPATH) -o simulation.o simulation.cpp

simparameters.o: simparameters.cpp simparameters.h
	$(CXX) -c $(CXXFLAGS) $(INCPATH) -o simparameters.o simparameters.cpp

mesh.o: mesh.cpp mesh.h
	$(CXX) -c $(CXXFLAGS) $(INCPATH) -o mesh.o mesh.cpp

rigidbodytemplate.o: rigidbodytemplate.cpp rigidbodytemplate.h \
		mesh.h \
		signeddistancefield.h
	$(CXX) -c $(CXXFLAGS) $(INCPATH) -o rigidbodytemplate.o rigidbodytemplate.cpp

rigidbodyinstance.o: rigidbodyinstance.cpp rigidbodyinstance.h \
		vectormath.h \
		rigidbodytemplate.h \
		mesh.h
	$(CXX) -c $(CXXFLAGS) $(INCPATH) -o rigidbodyinstance.o rigidbodyinstance.cpp

vectormath.o: vectormath.cpp vectormath.h
	$(CXX) -c $(CXXFLAGS) $(INCPATH) -o vectormath.o vectormath.cpp

camera.o: camera.cpp camera.h
	$(CXX) -c $(CXXFLAGS) $(INCPATH) -o camera.o camera.cpp

rotator.o: rotator.cpp rotator.h \
		camera.h
	$(CXX) -c $(CXXFLAGS) $(INCPATH) -o rotator.o rotator.cpp

signeddistancefield.o: signeddistancefield.cpp signeddistancefield.h \
		rigidbodytemplate.h \
		mesh.h \
		collisions/Distance.h \
		collisions/CTCD.h \
		vectormath.h
	$(CXX) -c $(CXXFLAGS) $(INCPATH) -o signeddistancefield.o signeddistancefield.cpp

CTCD.o: collisions/CTCD.cpp collisions/CTCD.h \
		collisions/rpoly.h
	$(CXX) -c $(CXXFLAGS) $(INCPATH) -o CTCD.o collisions/CTCD.cpp

cloth.o: cloth.cpp cloth.h \
		mesh.h \
		vectormath.h \
		simulation.h \
		simparameters.h
	$(CXX) -c $(CXXFLAGS) $(INCPATH) -o cloth.o cloth.cpp

moc_mainwindow.o: moc_mainwindow.cpp 
	$(CXX) -c $(CXXFLAGS) $(INCPATH) -o moc_mainwindow.o moc_mainwindow.cpp

moc_glpanel.o: moc_glpanel.cpp 
	$(CXX) -c $(CXXFLAGS) $(INCPATH) -o moc_glpanel.o moc_glpanel.cpp

moc_controller.o: moc_controller.cpp 
	$(CXX) -c $(CXXFLAGS) $(INCPATH) -o moc_controller.o moc_controller.cpp

####### Install

install:   FORCE

uninstall:   FORCE

FORCE:

