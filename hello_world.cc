
#include "tensorflow/lite/micro/kernels/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/simple_memory_allocator.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"
#include <iostream> // for check output

// Create an area of memory to use for input, output, and intermediate arrays.
// The size of this will depend on the model you're using, and may need to be
// determined by experimentation.
static const int tensor_arena_size = 6 * 1024;
static uint8_t tensor_arena[tensor_arena_size];

extern const unsigned char g_model[];
extern const int g_model_len;

// Set up logging.
static tflite::ErrorReporter *error_reporter = nullptr;
// This pulls in all the operation implementations we need.
static tflite::ops::micro::AllOpsResolver *resolver = nullptr;
static const tflite::Model* model = nullptr;
static tflite::MicroInterpreter *interpreter = nullptr;

extern void dump_data(char const* prefix, tflite::MicroInterpreter *interpreter,
	uint8_t const*tflite_array, uint8_t const* tflite_end,
	uint8_t const*tensor_arena, uint8_t const* arena_end);

void init(void)
{
	static tflite::MicroErrorReporter micro_error_reporter;
	error_reporter = &micro_error_reporter;

	// Map the model into a usable data structure. This doesn't involve any
	// copying or parsing, it's a very lightweight operation.
	model = ::tflite::GetModel(g_model);
	if (model->version() != TFLITE_SCHEMA_VERSION) {
		TF_LITE_REPORT_ERROR(error_reporter,
			"Model provided is schema version %d not equal "
			"to supported version %d.\n",
			model->version(), TFLITE_SCHEMA_VERSION);
		return;
	}
	static tflite::ops::micro::AllOpsResolver local_resolver;
	resolver= &local_resolver;

	// Build an interpreter to run the model with.
	static tflite::MicroInterpreter static_interpreter(model, *resolver, tensor_arena, tensor_arena_size, error_reporter);
	interpreter= &static_interpreter;
	TfLiteStatus allocate_status = interpreter->AllocateTensors();
	if (allocate_status != kTfLiteOk) {
		TF_LITE_REPORT_ERROR(error_reporter, "AllocateTensors() failed");
		return;
	}

	dump_data("hello_", interpreter, g_model, g_model + g_model_len, tensor_arena, tensor_arena + tensor_arena_size);
}

void run()
{
	TfLiteTensor* model_input = interpreter->input(0);
    model_input->data.f[0] = 1.57f; // roughly PI/2

	TfLiteStatus invoke_status = interpreter->Invoke();
	if (invoke_status != kTfLiteOk) {
		TF_LITE_REPORT_ERROR(error_reporter, "Invoke failed");
	}
	TfLiteTensor* model_output = interpreter->output(0);
    std::cerr << "result " << model_output->data.f[0] << std::endl;
}

extern void hello_init(uint8_t const*tflite_array, uint8_t const*tensor_arena);
extern void hello_invoke(void const* (inputs[1]), void * (outputs[1]));

void test_compiled(void) {
	float in = 1.57f, out = 0.0f;
	void const* in_array[1]= {&in};
	void* out_array[1]= {&out};
	hello_init(g_model, tensor_arena);
	hello_invoke(in_array, out_array);
	std::cerr << "compiled result " << out << std::endl;
}

int main(int argc, char** argv) {
	test_compiled();
	init();
	run();
	return 0;
}
