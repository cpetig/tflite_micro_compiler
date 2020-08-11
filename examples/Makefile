TF_DIR=../../tensorflow
include ../common.mk


all: hello_world  hello_world_compiled  mobilenet mobilenet_compiled cifar10 cifar10_compiled

clean:
	$(RM) *.o hello_world hello_world_compiled  mobilnet mobilnet_compiled cifar10 cifar10_compiled

mobilenet: mobilenet.o mobilenet_v1_0_25_160_quantized.o gnu.o
	$(CXX) -o $@ $^ $(LDOPTS) $(LIBS)

mobilenet_compiled: mobilenet2.o compiled_mobilenet.o gnu.o
	$(CXX) -o $@ $^ $(LDOPTS) $(LIBS)

hello_world: hello_world.o hello_world_model.o
	$(CXX) -o $@ $^ $(LDOPTS) $(LIBS)

hello_world_compiled: hello_world2.o compiled_hello_world.o
	$(CXX) -o $@ $^ $(LDOPTS)  $(LIBS)

hello_world_model.o: hello_world_model.cc
	$(CXX) -o $@ -c $^ $(CXXFLAGS)

cifar10: cifar10_model.o truck.o cifar10.o
	$(CXX) -o $@ $^ $(LDOPTS)  $(LIBS)

cifar10_compiled: truck.o compiled_cifar10.o cifar10_run_comp.o
	$(CXX) -o $@ $^ $(LDOPTS) $(LIBS)

regenerate: ../compiler
	../compiler hello_world.tflite compiled_hello_world.cc hello_ 
	../compiler mobilenet_v1_0_25_160_quantized.tflite compiled_mobilenet.cc mobilenet_
	../compiler cifar10.tflite compiled_cifar10.cc cifar_
