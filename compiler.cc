
#include <fstream>
#include <iostream>  // for check output

#include "tensorflow/lite/micro/kernels/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/simple_memory_allocator.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"

extern void dump_data(char const* prefix, tflite::MicroInterpreter* interpreter,
                      uint8_t const* tflite_array, uint8_t const* tflite_end,
                      uint8_t const* tensor_arena, uint8_t const* arena_end);

void register_addons(tflite::ops::micro::AllOpsResolver*);

int main(int argc, char** argv) {
  if (argc < 2 || argc > 4) {
    std::cerr << "USAGE: " << argv[0]
              << " input.tflite arena_size prefix >output.cpp" << std::endl;
    return 1;
  }
  int tensor_arena_size = 6 * 1024;
  if (argc >= 3) tensor_arena_size = atoi(argv[2]);
  char const* prefix = "model_";
  if (argc >= 4) prefix = argv[3];
  std::ifstream f(argv[1], std::ios::binary | std::ios::ate);
  if (!f) {
    std::cerr << "Could not open input file\n";
    return 2;
  }
  size_t model_len = f.tellg();
  f.seekg(0, std::ios::beg);
  std::vector<uint8_t> model_buf(model_len);
  if (!f.read((char*)model_buf.data(), model_len)) {
    std::cerr << "Could not read input file\n";
    return 3;
  }
  std::vector<uint8_t> arena_buf(tensor_arena_size);

  tflite::MicroErrorReporter micro_error_reporter;
  tflite::ErrorReporter* error_reporter = &micro_error_reporter;

  // Map the model into a usable data structure. This doesn't involve any
  // copying or parsing, it's a very lightweight operation.
  const tflite::Model* model = ::tflite::GetModel(model_buf.data());
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    TF_LITE_REPORT_ERROR(error_reporter,
                         "Model provided is schema version %d not equal "
                         "to supported version %d.\n",
                         model->version(), TFLITE_SCHEMA_VERSION);
    return 4;
  }
  tflite::ops::micro::AllOpsResolver resolver;
  register_addons(&resolver);

  // Build an interpreter to run the model with.
  tflite::MicroInterpreter interpreter(model, resolver, arena_buf.data(),
                                       tensor_arena_size, error_reporter);
  TfLiteStatus allocate_status = interpreter.AllocateTensors();
  if (allocate_status != kTfLiteOk) {
    TF_LITE_REPORT_ERROR(error_reporter, "AllocateTensors() failed");
    return 5;
  }

  dump_data(prefix, &interpreter, model_buf.data(),
            model_buf.data() + model_len, arena_buf.data(),
            arena_buf.data() + tensor_arena_size);

  return 0;
}
