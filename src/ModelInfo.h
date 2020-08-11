#ifndef TFLMCOMPILER_MODELINFO_H
#define TFLMCOMPILER_MODELINFO_H

#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace tflmc {

struct TensorInfo {
  TensorInfo(const TfLiteTensor *tensor_ptr) :
    tensor(tensor_ptr)
  {}
  const TfLiteTensor *tensor = nullptr;
};
struct RegistrationInfo {
  const TfLiteRegistration *reg = nullptr;
  tflite::BuiltinOperator code;
  std::string custom_name;
  bool operator==(const RegistrationInfo &other) {
    if (code != other.code) return false;
    if (code == tflite::BuiltinOperator_CUSTOM) {
      return custom_name == other.custom_name;
    } else
      return true;
  }
};
struct NodeInfo {
  NodeInfo() {}
  NodeInfo(TfLiteNode tfl_node, ptrdiff_t reg_index) :
    node(tfl_node),
    regIndex(reg_index)
  {}
  TfLiteNode node;
  ptrdiff_t regIndex = -1;
};
template <class T>
struct Option {
  bool None = true;
  T Some = T();
  void operator=(T const &val) {
    None = false;
    Some = val;
  }
  void clear() {
    Some = T();
    None = true;
  }
};

}  // namespace tflmc

#endif  // TFLMCOMPILER_MODELINFO_H
