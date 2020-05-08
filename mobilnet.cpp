
#include "tensorflow/lite/micro/kernels/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/simple_memory_allocator.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"
#include <stdio.h> // for check output

// Create an area of memory to use for input, output, and intermediate arrays.
// The size of this will depend on the model you're using, and may need to be
// determined by experimentation.
static const int tensor_arena_size = 10 * 1024 * 1024;
static uint8_t tensor_arena[tensor_arena_size];

extern "C" const unsigned char __1_tflite[];
extern "C" const unsigned int __1_tflite_len;

// Set up logging.
static tflite::MicroErrorReporter *micro_error_reporter;
// This pulls in all the operation implementations we need.
static tflite::ops::micro::AllOpsResolver *resolver;
static const tflite::Model* model;
static tflite::MicroInterpreter *interpreter;

extern void dump_data(char const* prefix, tflite::MicroInterpreter *interpreter,
	uint8_t const*tflite_array, uint8_t const* tflite_end,
	uint8_t const*tensor_arena, uint8_t const* arena_end);

void init(void)
{
	micro_error_reporter = new tflite::MicroErrorReporter;
	tflite::ErrorReporter* error_reporter = micro_error_reporter;

	// Map the model into a usable data structure. This doesn't involve any
	// copying or parsing, it's a very lightweight operation.
	model = ::tflite::GetModel(__1_tflite);
	if (model->version() != TFLITE_SCHEMA_VERSION) {
		error_reporter->Report(
			"Model provided is schema version %d not equal "
			"to supported version %d.\n",
			model->version(), TFLITE_SCHEMA_VERSION);
		return;
	}
	resolver = new tflite::ops::micro::AllOpsResolver();

	// Build an interpreter to run the model with.
	interpreter = new tflite::MicroInterpreter(model, *resolver, tensor_arena, tensor_arena_size, error_reporter);
	interpreter->AllocateTensors();

	dump_data("mobilnet_", interpreter, __1_tflite, __1_tflite + __1_tflite_len, tensor_arena, tensor_arena + tensor_arena_size);
}

void exit(void)
{
	if (interpreter) { delete interpreter; interpreter = 0; }
	if (resolver) { delete resolver; resolver = 0; }
	if (micro_error_reporter) { delete micro_error_reporter; micro_error_reporter = 0; }
	model = 0;
}

void run()
{
	TfLiteTensor* model_input = interpreter->input(0);
	for (uint32_t j = 0; j < model_input->dims->data[1]; ++j)
		model_input->data.f[j] = 0;

	TfLiteStatus invoke_status = interpreter->Invoke();
	if (invoke_status != kTfLiteOk) {
		((tflite::ErrorReporter*)micro_error_reporter)->Report("Invoke failed");
	}
}

int main(int argc, char** argv) {
	init();
	run();
	exit();
	return 0;
}

uint64_t GetTicks(void) { return 0; }
TfLiteStatus FullyConnectedEval(TfLiteContext* context, TfLiteNode* node) { return TfLiteStatus(); }
