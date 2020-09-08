// This file is generated. Do not edit.
// Generated on: 11.08.2020 11:26:36

#ifndef hello_GEN_H
#define hello_GEN_H

#include "tensorflow/lite/c/common.h"

// Sets up the model with init and prepare steps.
TfLiteStatus hello_init();
// Returns the input tensor with the given index.
TfLiteTensor *hello_input(int index);
// Returns the output tensor with the given index.
TfLiteTensor *hello_output(int index);
// Runs inference for the model.
TfLiteStatus hello_invoke();

// Returns the number of input tensors.
inline size_t hello_inputs() {
  return 1;
}
// Returns the number of output tensors.
inline size_t hello_outputs() {
  return 1;
}

inline void *hello_input_ptr(int index) {
  return hello_input(index)->data.data;
}
inline size_t hello_input_size(int index) {
  return hello_input(index)->bytes;
}
inline int hello_input_dims_len(int index) {
  return hello_input(index)->dims->data[0];
}
inline int *hello_input_dims(int index) {
  return &hello_input(index)->dims->data[1];
}

inline void *hello_output_ptr(int index) {
  return hello_output(index)->data.data;
}
inline size_t hello_output_size(int index) {
  return hello_output(index)->bytes;
}
inline int hello_output_dims_len(int index) {
  return hello_output(index)->dims->data[0];
}
inline int *hello_output_dims(int index) {
  return &hello_output(index)->dims->data[1];
}

#endif
