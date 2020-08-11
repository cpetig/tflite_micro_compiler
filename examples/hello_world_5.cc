
#include <iostream>  // for check output

#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/simple_memory_allocator.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"
#include "tensorflow/lite/micro/testing/test_utils.h"

// Create an area of memory to use for input, output, and intermediate arrays.
// The size of this will depend on the model you're using, and may need to be
// determined by experimentation.
static const int tensor_arena_size = 6 * 1024;
static uint8_t tensor_arena[tensor_arena_size];

extern const uint8_t hello_world_packed_5_data[];
// extern const int g_model_len;

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
  model = ::tflite::GetModel(hello_world_packed_5_data);
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
    
    using tflite::testing::F2Q;
    using tflite::testing::Q2F;
  
    TfLiteTensor* model_input = interpreter->input(0);
    // Provide an input value
    auto in_q = F2Q(1.57f, model_input); // roughly PI/2
	tflite::GetTensorData<uint8_t>(model_input)[0]= in_q;
 
    TfLiteStatus invoke_status = interpreter->Invoke();
    if (invoke_status != kTfLiteOk) {
        TF_LITE_REPORT_ERROR(error_reporter, "Invoke failed");
    }
    TfLiteTensor* model_output = interpreter->output(0);

    auto out_q = tflite::GetTensorData<uint8_t>(model_output)[0];
    float out = Q2F((int32_t)out_q, model_output);
    std::cerr << "result " << out << std::endl;
}

int main(int argc, char** argv) {
  init();
  run();
  return 0;
}
