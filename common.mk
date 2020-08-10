CXXFLAGS=-g -std=c++14 -DTF_LITE_STATIC_MEMORY -DNDEBUG -O3 -DTF_LITE_DISABLE_X86_NEON -DSUFFICIENT_ARENA_SIZE=128\*1024\*1024 \
	-I$(TF_DIR) -I$(TF_DIR)/tensorflow/lite/micro/tools/make/downloads/ \
	-I$(TF_DIR)/tensorflow/lite/micro/tools/make/downloads/gemmlowp \
	-I$(TF_DIR)/tensorflow/lite/micro/tools/make/downloads/flatbuffers/include \
	-I$(TF_DIR)/tensorflow/lite/micro/tools/make/downloads/ruy \
	-I$(TF_DIR)/tensorflow/lite/micro/tools/make/downloads/kissfft

LDOPTS=-L $(TF_DIR)/tensorflow/lite/micro/tools/make/gen/$(HOST_OS_BUILD)/lib


ifeq ($(OS),Windows_NT)
  LIBS=-ltensorflow-microlite 
  HOST_OS_BUILD=windows_x86_64
else
  LIBS=-ltensorflow-microlite -ldl
  HOST_OS_BUILD=linux_x86_64
endif
