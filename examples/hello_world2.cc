/* Copyright 2020 Christof Petig. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include <iostream>

static const int tensor_arena_size = 6 * 1024;
static uint8_t tensor_arena[tensor_arena_size];

// extern const unsigned char g_model[];
// extern const int g_model_len;

extern void hello_init(/*uint8_t const*tflite_array,*/ uint8_t *tensor_arena);
extern void hello_invoke(float const* input, float * output);

void test_compiled(void) {
	float in = 1.57f, out = 0.0f;
	// void const* in_array[1]= {&in};
	// void* out_array[1]= {&out};
	hello_init(/*g_model,*/ tensor_arena);
	hello_invoke(&in, &out);
	std::cerr << "compiled result " << out << std::endl;
}

int main(int argc, char** argv) {
	test_compiled();
	return 0;
}
