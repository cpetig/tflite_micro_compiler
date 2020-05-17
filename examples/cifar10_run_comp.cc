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

static const int tensor_arena_size = 150 * 1024;
static uint8_t tensor_arena[tensor_arena_size];

// extern "C" const unsigned char cifar10_tflite[];
// extern "C" const int cifar10_tflite_len;
extern "C" const unsigned char truck[];

extern void cifar_init(/*uint8_t const*tflite_array,*/ uint8_t *tensor_arena);
extern void cifar_invoke(void const* (inputs[1]), void * (outputs[1]));

void test_compiled(void) {
	float in[32*32*3];
	for (uint32_t i=0;i<32*32*3;++i) in[i]=truck[i]/255.0f;
	float out[10];
	void const* in_array[1]= {&in};
	void* out_array[1]= {&out};
	cifar_init(/*cifar10_tflite,*/ tensor_arena);
	cifar_invoke(in_array, out_array);
	for (uint32_t i=0;i<10;++i)
		std::cerr << out[i] << ", ";
	std::cerr << std::endl;
}

int main(int argc, char** argv) {
	test_compiled();
	return 0;
}
