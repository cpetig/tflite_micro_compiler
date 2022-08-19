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

#include <stdio.h>
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "lstm_compiled.cc.h"
#include <math.h>

static float state_h[6], state_c[6];

static const float amplitude=0.8;
static const float wavelength=16;
static const float phase = -3.141593f/2; // roughly -90deg

float calculate_sine(uint32_t index) {
    return amplitude*sinf(index*(6.283185f/wavelength) + phase);
}

void test_compiled(void) {
    lstm_input(1)->data.f = state_h;
    lstm_input(2)->data.f = state_c;
    lstm_output(1)->data.f = state_h; // feed back to state
    lstm_output(2)->data.f = state_c;
    for (uint32_t i=0;i<30;++i)
    {
        float in=calculate_sine(i);
    	tflite::GetTensorData<float>(lstm_input(0))[0]= in;
        lstm_invoke();
        printf("input %.3f output %.3f\n", in, tflite::GetTensorData<float>(lstm_output(0))[0]);
    }
}

int main(int argc, char** argv) {
	lstm_init();
	test_compiled();
	return 0;
}
