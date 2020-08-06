
#include <stdio.h>  // for check output

#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/simple_memory_allocator.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"

// Create an area of memory to use for input, output, and intermediate arrays.
// The size of this will depend on the model you're using, and may need to be
// determined by experimentation.
static const int tensor_arena_size = 10 * 1024 * 1024;
static uint8_t tensor_arena[tensor_arena_size];

extern "C" const unsigned char __1_tflite[];
// extern "C" const unsigned int __1_tflite_len;
extern "C" const unsigned char gnu_ppm[];

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
  model = ::tflite::GetModel(__1_tflite);
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

// strictly this is no longer necessary at all
void exit(void) {
  if (interpreter) {
    interpreter = 0;
  }
  if (resolver) {
    resolver = 0;
  }
  if (error_reporter) {
    error_reporter = 0;
  }
  model = 0;
}

void run() {
  TfLiteTensor* model_input = interpreter->input(0);
  memcpy(model_input->data.uint8, gnu_ppm, 160*160*3);

  TfLiteStatus invoke_status = interpreter->Invoke();
  if (invoke_status != kTfLiteOk) {
    TF_LITE_REPORT_ERROR(error_reporter, "Invoke failed");
  }
  TfLiteTensor* model_output = interpreter->output(0);
  uint32_t best=0;
  uint32_t bestval=model_output->data.uint8[0];
  for (uint32_t i=1;i<1001;++i) {
    if (model_output->data.uint8[i]>bestval) {
      bestval= model_output->data.uint8[i];
      best=i;
    }
  }
  printf("Best match is %u with %d%%\n", best, (int)(bestval * 100 / 255));
}

int main(int argc, char** argv) {
  init();
  run();
  exit();
  return 0;
}
