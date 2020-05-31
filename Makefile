TF_DIR=../tensorflow
CXXFLAGS=-g -std=c++14 -DTF_LITE_STATIC_MEMORY -DNDEBUG -O3 -DTF_LITE_DISABLE_X86_NEON \
	-I${TF_DIR} -I${TF_DIR}/tensorflow/lite/micro/tools/make/downloads/ \
	-I${TF_DIR}/tensorflow/lite/micro/tools/make/downloads/gemmlowp \
	-I${TF_DIR}/tensorflow/lite/micro/tools/make/downloads/flatbuffers/include \
	-I${TF_DIR}/tensorflow/lite/micro/tools/make/downloads/ruy \
	-I${TF_DIR}/tensorflow/lite/micro/tools/make/downloads/kissfft
LIBS=-L${TF_DIR}/tensorflow/lite/micro/tools/make/gen/linux_x86_64/lib/ \
	-ltensorflow-microlite

all: compiler

clean:
	make -C examples clean
	rm *.o compiler

compiler: src/main.o src/Compiler.o src/CodeWriter.o src/TypeToString.o
	$(CXX) -o $@ $^ ${LIBS}
