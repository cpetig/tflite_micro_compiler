// This file is generated. Do not edit.
// Generated on: 09.09.2020 16:20:04

#ifndef hello_world_5_GEN_H
#define hello_world_5_GEN_H

#include "tensorflow/lite/c/common.h"

// Sets up the model with init and prepare steps.
TfLiteStatus hello_world_5_init();
// Returns the input tensor with the given index.
TfLiteTensor *hello_world_5_input(int index);
// Returns the output tensor with the given index.
TfLiteTensor *hello_world_5_output(int index);
// Runs inference for the model.
TfLiteStatus hello_world_5_invoke();

// Returns the number of input tensors.
inline size_t hello_world_5_inputs() {
  return 1;
}
// Returns the number of output tensors.
inline size_t hello_world_5_outputs() {
  return 1;
}

inline void *hello_world_5_input_ptr(int index) {
  return hello_world_5_input(index)->data.data;
}
inline size_t hello_world_5_input_size(int index) {
  return hello_world_5_input(index)->bytes;
}
inline int hello_world_5_input_dims_len(int index) {
  return hello_world_5_input(index)->dims->data[0];
}
inline int *hello_world_5_input_dims(int index) {
  return &hello_world_5_input(index)->dims->data[1];
}

inline void *hello_world_5_output_ptr(int index) {
  return hello_world_5_output(index)->data.data;
}
inline size_t hello_world_5_output_size(int index) {
  return hello_world_5_output(index)->bytes;
}
inline int hello_world_5_output_dims_len(int index) {
  return hello_world_5_output(index)->dims->data[0];
}
inline int *hello_world_5_output_dims(int index) {
  return &hello_world_5_output(index)->dims->data[1];
}

#endif
