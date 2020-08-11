TF_DIR=../tensorflow
include common.mk

.PHONY: tflite all

all: compiler examples

tflite:
	$(MAKE) -C $(TF_DIR) -f tensorflow/lite/micro/tools/make/Makefile microlite

COMPILER_OBJS = src/main.o src/Compiler.o src/CodeWriter.o src/TypeToString.o src/RecordAllocations.o src/MemMap.o src/CustomOperators.o

compiler: $(COMPILER_OBJS) tflite
	$(CXX) $(LDOPTS) -o $@ $(COMPILER_OBJS) $(LIBS)

clean: clean-compiler clean-examples
	$(MAKE) -C $(TF_DIR) -f tensorflow/lite/micro/tools/make/makefile clean

FORMAT_FILES := $(shell find src -regex '.*\(h\|cpp\)')

format: 
	clang-format -i $(FORMAT_FILES)

.PHONY: examples clean-examples clean-compiler
examples:
	cd examples && $(MAKE)

clean-examples:
	$(MAKE) -C examples clean

clean-compiler:
	$(RM) src/*.o compiler
	
