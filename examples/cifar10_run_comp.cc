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
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"

extern "C" const unsigned char truck[];

extern void cifar_init();
extern void cifar_invoke();
extern TfLiteTensor* cifar_input(int index=0);
extern TfLiteTensor* cifar_output(int index=0);

void test_compiled(void) {
	float *in = cifar_input()->data.f;
	for (uint32_t i=0;i<32*32*3;++i) in[i]=truck[i]/255.0f;
	float *out = cifar_output()->data.f;
	cifar_invoke();
	for (uint32_t i=0;i<10;++i)
		std::cerr << out[i] << ", ";
	std::cerr << std::endl;
}

int main(int argc, char** argv) {
	cifar_init();
	test_compiled();
	return 0;
}
