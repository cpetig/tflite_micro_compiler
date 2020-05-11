
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
static int tensor_arena_size = 6 * 1024;
static uint8_t* tensor_arena=nullptr;
static uint8_t* g_model=nullptr;
static int g_model_len=0;
static char const *prefix="model_";

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

	dump_data(prefix, interpreter, g_model, g_model + g_model_len, tensor_arena, tensor_arena + tensor_arena_size);
}

int main(int argc, char** argv) {
	if (argc<2 || argc>4) {
		std::cerr << "USAGE: " << argv[0] << " input.tflite arena_size prefix >output.cpp" << std::endl;
		return 1;
	}
	if (argc>=3) tensor_arena_size= atoi(argv[2]);
	if (argc>=4) prefix=argv[3];
	FILE *f = fopen(argv[1], "rb");
	if (!f) {
		perror(argv[1]);
		return 2;
	}
	fseek(f, 0L, SEEK_END); 
    g_model_len = ftell(f); 
	fseek(f, 0L, SEEK_SET);
	g_model= (uint8_t*) malloc(g_model_len);
	if (!g_model) { std::cerr << "allocation failed" << std::endl; fclose(f); return 3; }
	fread(g_model, 1, g_model_len, f);
	fclose(f);
	tensor_arena= (uint8_t *)malloc(tensor_arena_size);
	init();
	free(tensor_arena);
	free(g_model);
	return 0;
}
