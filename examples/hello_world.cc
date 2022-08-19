#ifndef  TF_LITE_MICRO_FOOTPRINT_ONLY
#include <iostream>  // for check output
#endif
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/simple_memory_allocator.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"

// Create an area of memory to use for input, output, and intermediate arrays.
// The size of this will depend on the model you're using, and may need to be
// determined by experimentation.
static const int tensor_arena_size = 6 * 1024;
static uint8_t tensor_arena[tensor_arena_size];


typedef tflite::MicroMutableOpResolver<1> OpResolver;

extern const uint8_t hello_world_data[];

// Set up logging.
static tflite::ErrorReporter* reporter = nullptr;
// This pulls in all the operation implementations we need.
static OpResolver* resolver = nullptr;
static const tflite::Model* model = nullptr;
static tflite::MicroInterpreter* interpreter = nullptr;

void init(void) {
  static tflite::MicroErrorReporter micro_reporter;
  reporter = &micro_reporter;

  // Map the model into a usable data structure. This doesn't involve any
  // copying or parsing, it's a very lightweight operation.
  model = ::tflite::GetModel(hello_world_data);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    TF_LITE_REPORT_ERROR(reporter,
                         "Model provided is schema version %d not equal "
                         "to supported version %d.\n",
                         model->version(), TFLITE_SCHEMA_VERSION);
    return;
  }

  static OpResolver local_resolver(reporter);
  if (local_resolver.AddFullyConnected() != kTfLiteOk) {
    return;
  }
  resolver = &local_resolver;

  // Build an interpreter to run the model with.
  static tflite::MicroInterpreter static_interpreter(
      model, *resolver, tensor_arena, tensor_arena_size, reporter);
  interpreter = &static_interpreter;
  TfLiteStatus allocate_status = interpreter->AllocateTensors();
  if (allocate_status != kTfLiteOk) {
    TF_LITE_REPORT_ERROR(reporter, "AllocateTensors() failed");
    return;
  }
}


#ifndef  TF_LITE_MICRO_FOOTPRINT_ONLY

uint8_t F2Q(float value, const TfLiteTensor *tensor)
{
  int32_t result = tensor->params.zero_point +
                   (value / tensor->params.scale) + 0.5f;
  if (result < std::numeric_limits<uint8_t>::min()) {
    result = std::numeric_limits<uint8_t>::min();
  }
  if (result > std::numeric_limits<uint8_t>::max()) {
    result = std::numeric_limits<uint8_t>::max();
  }
  return result;
} 

// Converts a quantized value to coded float for quantization
// params of specified tensor
float Q2F(int32_t code, const TfLiteTensor *tensor)
{
    return (code - (int32_t)tensor->params.zero_point) * tensor->params.scale;
}
#endif

void run() {
 #ifndef  TF_LITE_MICRO_FOOTPRINT_ONLY
 
    TfLiteTensor* model_input = interpreter->input(0);
    // Provide an input value
    auto in_q = F2Q(1.57f, model_input); // roughly PI/2
	tflite::GetTensorData<uint8_t>(model_input)[0]= in_q;
#endif

    TfLiteStatus invoke_status = interpreter->Invoke();
    if (invoke_status != kTfLiteOk) {
        TF_LITE_REPORT_ERROR(reporter, "Invoke failed");
    }
#ifndef  TF_LITE_MICRO_FOOTPRINT_ONLY
    TfLiteTensor* model_output = interpreter->output(0);

    auto out_q = tflite::GetTensorData<uint8_t>(model_output)[0];
    float out = Q2F((int32_t)out_q, model_output);
    std::cerr << "result " << out << std::endl;
#endif
}

int main(int argc, char** argv) {
  init();
  run();
  return 0;
}
