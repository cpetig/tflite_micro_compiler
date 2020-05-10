TF_DIR=../tensorflow
CXXFLAGS= -std=c++11 -DTF_LITE_STATIC_MEMORY -DNDEBUG -O3 -DTF_LITE_DISABLE_X86_NEON \
	-I${TF_DIR} -I${TF_DIR}/tensorflow/lite/micro/tools/make/downloads/ \
	-I${TF_DIR}/tensorflow/lite/micro/tools/make/downloads/gemmlowp \
	-I${TF_DIR}/tensorflow/lite/micro/tools/make/downloads/flatbuffers/include \
	-I${TF_DIR}/tensorflow/lite/micro/tools/make/downloads/ruy \
	-I${TF_DIR}/tensorflow/lite/micro/tools/make/downloads/kissfft
LIBS=-L${TF_DIR}/tensorflow/lite/micro/tools/make/gen/linux_x86_64/lib/ \
	-ltensorflow-microlite

all: hello_world mobilnet

mobilnet: mobilnet.o mobilenet_v1_0_25_160_quantized.o tflu_dump.o compiled_mobilnet.o
	$(CXX) -o $@ $^ ${LIBS}

hello_world: hello_world.o hello_world_model.o tflu_dump.o compiled_hello.o
	$(CXX) -o $@ $^ ${LIBS}

hello_world_model.o: ${TF_DIR}/tensorflow/lite/micro/examples/hello_world/model.cc
	$(CXX) -o $@ -c $^ $(CXXFLAGS)
