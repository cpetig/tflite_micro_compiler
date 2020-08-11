// This file is generated. Do not edit.
// Generated on: 11.08.2020 11:26:36

#ifndef mobilenet_GEN_H
#define mobilenet_GEN_H

#include "tensorflow/lite/c/common.h"

// Sets up the model with init and prepare steps.
TfLiteStatus mobilenet_init();
// Returns the input tensor with the given index.
TfLiteTensor *mobilenet_input(int index);
// Returns the output tensor with the given index.
TfLiteTensor *mobilenet_output(int index);
// Runs inference for the model.
TfLiteStatus mobilenet_invoke();

// Returns the number of input tensors.
inline size_t mobilenet_inputs() {
  return 1;
}
// Returns the number of output tensors.
inline size_t mobilenet_outputs() {
  return 1;
}

inline void *mobilenet_input_ptr(int index) {
  return mobilenet_input(index)->data.data;
}
inline size_t mobilenet_input_size(int index) {
  return mobilenet_input(index)->bytes;
}
inline int mobilenet_input_dims_len(int index) {
  return mobilenet_input(index)->dims->data[0];
}
inline int *mobilenet_input_dims(int index) {
  return &mobilenet_input(index)->dims->data[1];
}

inline void *mobilenet_output_ptr(int index) {
  return mobilenet_output(index)->data.data;
}
inline size_t mobilenet_output_size(int index) {
  return mobilenet_output(index)->bytes;
}
inline int mobilenet_output_dims_len(int index) {
  return mobilenet_output(index)->dims->data[0];
}
inline int *mobilenet_output_dims(int index) {
  return &mobilenet_output(index)->dims->data[1];
}

#endif
