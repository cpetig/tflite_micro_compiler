TF_DIR=../../tensorflow
include ../common.mk

EXAMPLES=hello_world mobilenet hello_world_compiled cifar10 cifar10_compiled hello_world_5 hello_world_5_compiled
BINARIES= $(foreach e,  $(EXAMPLES), $(e)$(EXE_SUFFIX))

$(info $(BINARIES))

all: $(BINARIES)

run_all: all
	for f in $(BINARIES) ; do echo $$f ; ./$$f ; done

clean:
	$(RM) *.o $(BINARIES)

mobilenet$(EXE_SUFFIX): mobilenet.o mobilenet_v1_0_25_160_quantized.o gnu.o
	$(CXX) -o $@ $^ $(LDOPTS) $(LIBS)

mobilenet_compiled$(EXE_SUFFIX): mobilenet2.o compiled_mobilenet.o gnu.o
	$(CXX) -o $@ $^ $(LDOPTS) $(LIBS)

hello_world$(EXE_SUFFIX): hello_world.o hello_world_model.o
	$(CXX) -o $@ $^ $(LDOPTS) $(LIBS)

hello_world_compiled$(EXE_SUFFIX): hello_world2.o compiled_hello_world.o
	$(CXX) -o $@ $^ $(LDOPTS)  $(LIBS)


hello_world_5$(EXE_SUFFIX): hello_world_5.o hello_world_model_5.o
	$(CXX) -o $@ $^ $(LDOPTS) $(LIBS)

hello_world_5_compiled$(EXE_SUFFIX): hello_world2_5.o compiled_hello_world_5.o
	$(CXX) -o $@ $^ $(LDOPTS)  $(LIBS)


cifar10$(EXE_SUFFIX): cifar10_model.o truck.o cifar10.o
	$(CXX) -o $@ $^ $(LDOPTS)  $(LIBS)

cifar10_compiled$(EXE_SUFFIX): truck.o compiled_cifar10.o cifar10_run_comp.o
	$(CXX) -o $@ $^ $(LDOPTS) $(LIBS)

regenerate: ../compiler$(EXE_SUFFIX)
	../compiler hello_world.tflite compiled_hello_world.cc hello_ 
	../compiler hello_world_5.tflite compiled_hello_world_5.cc hello_ 
	../compiler mobilenet_v1_0_25_160_quantized.tflite compiled_mobilenet.cc mobilenet_
	../compiler cifar10.tflite compiled_cifar10.cc cifar_
