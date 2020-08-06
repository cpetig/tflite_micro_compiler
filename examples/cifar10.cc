
#include <iostream>  // for check output

#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/simple_memory_allocator.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"

// Create an area of memory to use for input, output, and intermediate arrays.
// The size of this will depend on the model you're using, and may need to be
// determined by experimentation.
static const int tensor_arena_size = 150 * 1000;
static uint8_t tensor_arena[tensor_arena_size];

extern "C" const unsigned char cifar10_tflite[];
extern "C" const unsigned char truck[];
extern "C" const int cifar10_tflite_len;

// Set up logging.
static tflite::ErrorReporter* error_reporter = nullptr;
// This pulls in all the operation implementations we need.
static tflite::AllOpsResolver* resolver = nullptr;
static const tflite::Model* model = nullptr;
static tflite::MicroInterpreter* interpreter = nullptr;

void init(void) {
  static tflite::MicroErrorReporter micro_error_reporter;
  error_reporter = &micro_error_reporter;

  // Map the model into a usable data structure. This doesn't involve any
  // copying or parsing, it's a very lightweight operation.
  model = ::tflite::GetModel(cifar10_tflite);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    TF_LITE_REPORT_ERROR(error_reporter,
                         "Model provided is schema version %d not equal "
                         "to supported version %d.\n",
                         model->version(), TFLITE_SCHEMA_VERSION);
    return;
  }
  static tflite::AllOpsResolver local_resolver;
  resolver = &local_resolver;

  // Build an interpreter to run the model with.
  static tflite::MicroInterpreter static_interpreter(
      model, *resolver, tensor_arena, tensor_arena_size, error_reporter);
  interpreter = &static_interpreter;
  TfLiteStatus allocate_status = interpreter->AllocateTensors();
  if (allocate_status != kTfLiteOk) {
    TF_LITE_REPORT_ERROR(error_reporter, "AllocateTensors() failed");
    return;
  }
}

void run() {
  TfLiteTensor* model_input = interpreter->input(0);
  for (uint32_t i = 0; i < 32 * 32 * 3; ++i)
    model_input->data.f[i] = truck[i] / 255.0f;

  TfLiteStatus invoke_status = interpreter->Invoke();
  if (invoke_status != kTfLiteOk) {
    TF_LITE_REPORT_ERROR(error_reporter, "Invoke failed");
  }
  TfLiteTensor* model_output = interpreter->output(0);
  for (int i = 0; i < model_output->dims->data[1]; ++i)
    std::cerr << model_output->data.f[i] << ", ";
  std::cerr << std::endl;
}

int main(int argc, char** argv) {
  init();
  run();
  return 0;
}
