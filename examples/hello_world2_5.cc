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
#include "compiled_hello_world_5.cc.h"
#include "tensorflow/lite/micro/testing/test_utils.h"

void test_compiled(void) {
	hello_world_5_init();
    
    using tflite::testing::F2Q;
    using tflite::testing::Q2F;
  
  
    // Provide an input value
    auto in_q = F2Q(1.57f, hello_world_5_input(0));
	tflite::GetTensorData<uint8_t>(hello_world_5_input(0))[0]= in_q;
	hello_world_5_invoke();
	auto out_q = tflite::GetTensorData<uint8_t>(hello_world_5_output(0))[0];
    float out = Q2F((int32_t)out_q, hello_world_5_output(0));
	std::cerr << "result " << out << std::endl;
}

int main(int argc, char** argv) {
	test_compiled();
	return 0;
}
