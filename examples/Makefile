TF_DIR=../../tensorflow
include ../common.mk


COMPILED_FILES_SELECTION ?= reference
CXXFLAGS += -I $(COMPILED_FILES_SELECTION)

EXAMPLES=hello_world mobilenet hello_world_compiled cifar10 cifar10_compiled hello_world_5 hello_world_5_compiled
BINARIES= $(foreach e,  $(EXAMPLES), $(e)$(EXE_SUFFIX))

$(info $(BINARIES))

all: $(BINARIES)

run_all: all
	for f in $(BINARIES) ; do echo $$f ; ./$$f ; done

clean:
	$(RM) *.o $(BINARIES)
	rm -rf generated

mobilenet$(EXE_SUFFIX): mobilenet.o mobilenet_model.o gnu.o
	$(CXX) -o $@ $^ $(LDOPTS) $(LIBS)

mobilenet_compiled$(EXE_SUFFIX): mobilenet2.o $(COMPILED_FILES_SELECTION)/compiled_mobilenet.o gnu.o
	$(CXX) -o $@ $^ $(LDOPTS) $(LIBS)

hello_world$(EXE_SUFFIX): hello_world.o hello_world_model.o
	$(CXX) -o $@ $^ $(LDOPTS) $(LIBS)

hello_world_compiled$(EXE_SUFFIX): hello_world2.o $(COMPILED_FILES_SELECTION)/compiled_hello_world.o
	$(CXX) -o $@ $^ $(LDOPTS)  $(LIBS)


hello_world_5$(EXE_SUFFIX): hello_world_5.o hello_world_5_model.o
	$(CXX) -o $@ $^ $(LDOPTS) $(LIBS)

hello_world_5_compiled$(EXE_SUFFIX): hello_world2_5.o $(COMPILED_FILES_SELECTION)/compiled_hello_world_5.o
	$(CXX) -o $@ $^ $(LDOPTS)  $(LIBS)


cifar10$(EXE_SUFFIX): cifar10_model.o truck.o cifar10.o
	$(CXX) -o $@ $^ $(LDOPTS)  $(LIBS)

cifar10_compiled$(EXE_SUFFIX): truck.o $(COMPILED_FILES_SELECTION)/compiled_cifar10.o cifar10_run_comp.o
	$(CXX) -o $@ $^ $(LDOPTS) $(LIBS)

regenerate: ../compiler$(EXE_SUFFIX)
	mkdir -p generated
	../compiler hello_world.tflite generated/compiled_hello_world.cc hello_world_
	../compiler hello_world_5.tflite generated/compiled_hello_world_5.cc hello_world_5_
	../compiler mobilenet.tflite generated/compiled_mobilenet.cc mobilenet_
	../compiler cifar10.tflite generated/compiled_cifar10.cc cifar10_
