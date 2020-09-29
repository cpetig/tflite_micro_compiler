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
#include <stdio.h>
#endif
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "compiled_mobilenet.cc.h"

extern "C" const unsigned char gnu_ppm[];

int run() {
#ifndef  TF_LITE_MICRO_FOOTPRINT_ONLY
  TfLiteTensor* model_input = mobilenet_input(0);
  memcpy(model_input->data.uint8, gnu_ppm, 160*160*3);
#endif
  TfLiteStatus invoke_status = mobilenet_invoke();
  if (invoke_status != kTfLiteOk) {
#ifndef  TF_LITE_MICRO_FOOTPRINT_ONLY
    fprintf(stderr, "Invoke failed\n");
#endif
    return 1;
  }
#ifndef  TF_LITE_MICRO_FOOTPRINT_ONLY
  TfLiteTensor* model_output = mobilenet_output(0);
  int best=0;
  int bestval=model_output->data.uint8[0];
  for (uint32_t i=1;i<1001;++i) {
    if (model_output->data.uint8[i]>bestval) {
      bestval= model_output->data.uint8[i];
      best=i;
    }
  }
  printf("Best match is %d with %d%%\n", best, bestval * 100/256);
#endif
  return 0;
}

int main(int argc, char** argv) {
  mobilenet_init();
  return run();
}
