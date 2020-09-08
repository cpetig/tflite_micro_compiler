$(info Adding tflite-micro compiler)
TFLITE_U_COMPILER_SRCS := \
  src/CodeWriter.cc  src/CustomOperators.cc src/MemMap.cc src/TypeToString.cc \
  src/Compiler.cc  src/main.cc src/RecordAllocations.cc

TFLITE_U_COMPILER_HDRS := \
  src/CodeWriter.h src/Compiler.h src/CustomOperators.h src/MemMap.h src/RecordAllocations.h src/TypeToString.h




# Builds a standalone binary.
$(eval $(call microlite_test,compiler,\
  $(TFLITE_U_COMPILER_SRCS),$(TFLITE_U_COMPILER_HDRS)))
