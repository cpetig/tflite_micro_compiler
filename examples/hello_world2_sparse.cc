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

#ifndef  TF_LITE_MICRO_FOOTPRINT_ONLY
#include <iostream>
#endif
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "compiled_hello_world_pruned_8.cc.h"
#ifndef  TF_LITE_MICRO_FOOTPRINT_ONLY
#include "tensorflow/lite/micro/testing/test_utils.h"
#endif

int test_compiled(void) {
	hello_world_pruned_8_init();
	const uint8_t out_ref_value = 185;
  const uint8_t in_q = 20;
	tflite::GetTensorData<uint8_t>(hello_world_pruned_8_input(0))[0]= in_q;
 #ifndef  TF_LITE_MICRO_FOOTPRINT_ONLY
    using tflite::testing::Q2F;
    // Decode implied input value
    std::cerr << "in " << (int)in_q << " coding " 
		          <<  Q2F((int32_t)in_q, hello_world_pruned_8_input(0)) << std::endl;
#endif

	hello_world_pruned_8_invoke();
	auto out_q = tflite::GetTensorData<uint8_t>(hello_world_pruned_8_output(0))[0];
#ifndef  TF_LITE_MICRO_FOOTPRINT_ONLY
	float out = Q2F((int32_t)out_q, hello_world_pruned_8_output(0));
	std::cerr << "result " << (int)out_q << " coding " << out << std::endl;
#endif

  return out_q == out_ref_value ? 0 : 1;
}

int main(int argc, char** argv) {
	return 	test_compiled();}
