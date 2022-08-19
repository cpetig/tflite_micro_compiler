// This file is generated. Do not edit.
// Generated on: 12.08.2020 18:54:29

#ifndef lstm_GEN_H
#define lstm_GEN_H

#include "tensorflow/lite/c/common.h"

// Sets up the model with init and prepare steps.
TfLiteStatus lstm_init();
// Returns the input tensor with the given index.
TfLiteTensor *lstm_input(int index);
// Returns the output tensor with the given index.
TfLiteTensor *lstm_output(int index);
// Runs inference for the model.
TfLiteStatus lstm_invoke();

// Returns the number of input tensors.
inline size_t lstm_inputs() {
  return 3;
}
// Returns the number of output tensors.
inline size_t lstm_outputs() {
  return 3;
}

inline void *lstm_input_ptr(int index) {
  return lstm_input(index)->data.data;
}
inline size_t lstm_input_size(int index) {
  return lstm_input(index)->bytes;
}
inline int lstm_input_dims_len(int index) {
  return lstm_input(index)->dims->data[0];
}
inline int *lstm_input_dims(int index) {
  return &lstm_input(index)->dims->data[1];
}

inline void *lstm_output_ptr(int index) {
  return lstm_output(index)->data.data;
}
inline size_t lstm_output_size(int index) {
  return lstm_output(index)->bytes;
}
inline int lstm_output_dims_len(int index) {
  return lstm_output(index)->dims->data[0];
}
inline int *lstm_output_dims(int index) {
  return &lstm_output(index)->dims->data[1];
}

#endif
