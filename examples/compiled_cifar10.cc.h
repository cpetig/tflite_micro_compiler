// This file is generated. Do not edit.
// Generated on: 11.08.2020 11:26:36

#ifndef cifar_GEN_H
#define cifar_GEN_H

#include "tensorflow/lite/c/common.h"

// Sets up the model with init and prepare steps.
TfLiteStatus cifar_init();
// Returns the input tensor with the given index.
TfLiteTensor *cifar_input(int index);
// Returns the output tensor with the given index.
TfLiteTensor *cifar_output(int index);
// Runs inference for the model.
TfLiteStatus cifar_invoke();

// Returns the number of input tensors.
inline size_t cifar_inputs() {
  return 1;
}
// Returns the number of output tensors.
inline size_t cifar_outputs() {
  return 1;
}

inline void *cifar_input_ptr(int index) {
  return cifar_input(index)->data.data;
}
inline size_t cifar_input_size(int index) {
  return cifar_input(index)->bytes;
}
inline int cifar_input_dims_len(int index) {
  return cifar_input(index)->dims->data[0];
}
inline int *cifar_input_dims(int index) {
  return &cifar_input(index)->dims->data[1];
}

inline void *cifar_output_ptr(int index) {
  return cifar_output(index)->data.data;
}
inline size_t cifar_output_size(int index) {
  return cifar_output(index)->bytes;
}
inline int cifar_output_dims_len(int index) {
  return cifar_output(index)->dims->data[0];
}
inline int *cifar_output_dims(int index) {
  return &cifar_output(index)->dims->data[1];
}

#endif
