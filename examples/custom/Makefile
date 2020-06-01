TF_DIR=../../../tensorflow
CXXFLAGS=-fPIC -g -std=c++11 -DTF_LITE_STATIC_MEMORY -DNDEBUG -O3 -DTF_LITE_DISABLE_X86_NEON \
	-I${TF_DIR} -I${TF_DIR}/tensorflow/lite/micro/tools/make/downloads/ \
	-I${TF_DIR}/tensorflow/lite/micro/tools/make/downloads/gemmlowp \
	-I${TF_DIR}/tensorflow/lite/micro/tools/make/downloads/flatbuffers/include \
	-I${TF_DIR}/tensorflow/lite/micro/tools/make/downloads/ruy \
	-I${TF_DIR}/tensorflow/lite/micro/tools/make/downloads/kissfft

libtflite_micro_custom.so: registration.o fake_implementations.o custom_implementations.o
	$(CXX) --shared -o $@ $^
