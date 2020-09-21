// This file is generated. Do not edit.
// Generated on: 21.09.2020 11:16:55

#ifndef cifar10_GEN_H
#define cifar10_GEN_H

#include "tensorflow/lite/c/common.h"

// Sets up the model with init and prepare steps.
TfLiteStatus cifar10_init();
// Returns the input tensor with the given index.
TfLiteTensor *cifar10_input(int index);
// Returns the output tensor with the given index.
TfLiteTensor *cifar10_output(int index);
// Runs inference for the model.
TfLiteStatus cifar10_invoke();

// Returns the number of input tensors.
inline size_t cifar10_inputs() {
  return 1;
}
// Returns the number of output tensors.
inline size_t cifar10_outputs() {
  return 1;
}

inline void *cifar10_input_ptr(int index) {
  return cifar10_input(index)->data.data;
}
inline size_t cifar10_input_size(int index) {
  return cifar10_input(index)->bytes;
}
inline int cifar10_input_dims_len(int index) {
  return cifar10_input(index)->dims->data[0];
}
inline int *cifar10_input_dims(int index) {
  return &cifar10_input(index)->dims->data[1];
}

inline void *cifar10_output_ptr(int index) {
  return cifar10_output(index)->data.data;
}
inline size_t cifar10_output_size(int index) {
  return cifar10_output(index)->bytes;
}
inline int cifar10_output_dims_len(int index) {
  return cifar10_output(index)->dims->data[0];
}
inline int *cifar10_output_dims(int index) {
  return &cifar10_output(index)->dims->data[1];
}

#endif
