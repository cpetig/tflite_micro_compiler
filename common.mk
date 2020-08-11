CXXFLAGS=-g -std=c++14 -DTF_LITE_STATIC_MEMORY -DDEBUG -O1 -DTF_LITE_DISABLE_X86_NEON -DSUFFICIENT_ARENA_SIZE=128\*1024\*1024 \
	-I$(TF_DIR) -I$(TF_DIR)/tensorflow/lite/micro/tools/make/downloads/ \
	-I$(TF_DIR)/tensorflow/lite/micro/tools/make/downloads/gemmlowp \
	-I$(TF_DIR)/tensorflow/lite/micro/tools/make/downloads/flatbuffers/include \
	-I$(TF_DIR)/tensorflow/lite/micro/tools/make/downloads/ruy \
	-I$(TF_DIR)/tensorflow/lite/micro/tools/make/downloads/kissfft 

ifeq ($(BUILD_TYPE),debug)
  HOST_OS_BUILD:=$(HOST_OS_BUILD)_debug
endif
TF_MICROLITE_LIBDIR=$(TF_DIR)/tensorflow/lite/micro/tools/make/gen/$(HOST_OS_BUILD)/lib
TF_MICROLITE_LIB=$(TF_MICROLITE_LIBDIR)/libtensorflow-microlite.a

ifeq ($(OS),Windows_NT)
  LIBS=$(TF_MICROLITE_LIB)
  HOST_OS_BUILD := windows_x86_64
  EXE_SUFFIX := .exe
else
  LIBS=$(TF_MICROLITE_LIB) -ldl
  HOST_OS_BUILD := linux_x86_64
   EXE_SUFFIX :=
endif

