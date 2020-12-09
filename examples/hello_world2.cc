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

#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"

#include "compiled_hello_world.cc.h"
#include "tensorflow/lite/micro/testing/test_utils.h"


int test_compiled(void) {
	hello_world_init();
  using tflite::testing::F2Q;
  using tflite::testing::Q2F;
  
  // Provide an input value
  // auto in_q = F2Q(0.52f, hello_world_input(0)); // Output of this is 21 -> see next line
  uint8_t in_q = 21;
  // std::cerr << static_cast<int>(in_q) << std::endl;
	tflite::GetTensorData<uint8_t>(hello_world_input(0))[0]= in_q;
	hello_world_invoke();
	auto out_q = tflite::GetTensorData<uint8_t>(hello_world_output(0))[0];
	if (out_q == 187) {
	  return 0;       // Correct result
	} else return 1;  // Wrong result
  // float out = Q2F((int32_t)out_q, hello_world_output(0));
	// std::cerr << "result " << out << std::endl;
}

int main(int argc, char** argv) {
	int success = test_compiled();
	return success;
}
