TF_DIR=../tensorflow
CXXFLAGS=-g -std=c++14 -DTF_LITE_STATIC_MEMORY -DNDEBUG -O3 -DTF_LITE_DISABLE_X86_NEON -DSUFFICIENT_ARENA_SIZE=128\*1024\*1024 \
	-I${TF_DIR} -I${TF_DIR}/tensorflow/lite/micro/tools/make/downloads/ \
	-I${TF_DIR}/tensorflow/lite/micro/tools/make/downloads/gemmlowp \
	-I${TF_DIR}/tensorflow/lite/micro/tools/make/downloads/flatbuffers/include \
	-I${TF_DIR}/tensorflow/lite/micro/tools/make/downloads/ruy \
	-I${TF_DIR}/tensorflow/lite/micro/tools/make/downloads/kissfft
LIBS=-L${TF_DIR}/tensorflow/lite/micro/tools/make/gen/linux_x86_64/lib/ \
	-ltensorflow-microlite -ldl

all: compiler examples

compiler: src/main.o src/Compiler.o src/CodeWriter.o src/TypeToString.o src/RecordAllocations.o src/MemMap.o
	$(CXX) -o $@ $^ ${LIBS}

clean: clean-compiler clean-examples

.PHONY: examples clean-examples clean-compiler
examples:
	cd examples && $(MAKE)

clean-examples:
	$(MAKE) -C examples clean

clean-compiler:
	$(RM) src/*.o compiler
