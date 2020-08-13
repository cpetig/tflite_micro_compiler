TF_DIR=../tensorflow
include common.mk

.PHONY: tflite all 

all: compiler$(EXE_SUFFIX) examples

$(TF_MICROLITE_LIB): tflite

tflite:
	$(MAKE) -C $(TF_DIR) -f tensorflow/lite/micro/tools/make/Makefile microlite

COMPILER_OBJS = src/main.o src/Compiler.o src/CodeWriter.o src/TypeToString.o src/RecordAllocations.o src/MemMap.o src/CustomOperators.o

compiler$(EXE_SUFFIX): $(COMPILER_OBJS) $(TF_MICROLITE_LIB)
	$(CXX) $(CXXFLAGS) $(LDOPTS) -o $@ $(COMPILER_OBJS) $(LIBS)

clean: clean-compiler clean-examples
	$(MAKE) -C $(TF_DIR) -f tensorflow/lite/micro/tools/make/makefile clean

FORMAT_FILES := $(shell find src -regex '.*\(h\|cpp\)')

format: 
	clang-format -i $(FORMAT_FILES)

.PHONY: examples clean-examples clean-compiler

examples: tflite
	$(MAKE) -C examples all

run_examples: tflite
	$(MAKE) -C examples run_all
 
regenerate: compiler$(EXE_SUFFIX)
	$(MAKE) -C examples regenerate

clean-examples:
	$(MAKE) -C examples clean

clean-compiler:
	$(RM) src/*.o compiler$(EXE_SUFFIX)
	
