// This file is generated. Do not edit.
// Generated on: 01.09.2020 23:27:38

#ifndef hello_GEN_H
#define hello_GEN_H

#include "tensorflow/lite/c/common.h"

namespace hello_ {
// Sets up the model with init and prepare steps.
TfLiteStatus init();
// Returns the input tensor with the given index.
TfLiteTensor *input(int index);
// Returns the output tensor with the given index.
TfLiteTensor *output(int index);
// Runs inference for the model.
TfLiteStatus invoke();

// Returns the number of input tensors.
inline size_t inputs() {
  return 1;
}
// Returns the number of output tensors.
inline size_t outputs() {
  return 1;
}

#if 0 // enable only if you need these shortcuts
inline void *input_ptr(int index) {
  return input(index)->data.data;
}
inline size_t input_size(int index) {
  return input(index)->bytes;
}
inline int input_dims_len(int index) {
  return input(index)->dims->data[0];
}
inline int *input_dims(int index) {
  return &input(index)->dims->data[1];
}

inline void *output_ptr(int index) {
  return output(index)->data.data;
}
inline size_t output_size(int index) {
  return output(index)->bytes;
}
inline int output_dims_len(int index) {
  return output(index)->dims->data[0];
}
inline int *output_dims(int index) {
  return &output(index)->dims->data[1];
}
#endif
} // namespace hello_

#endif
