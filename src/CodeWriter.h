#ifndef TFLMCOMPILER_CODEWRITER_H
#define TFLMCOMPILER_CODEWRITER_H

#include <iostream>

#include "tensorflow/lite/micro/micro_interpreter.h"

namespace tflmc {

// Helper functions for top-level code generation.
class CodeWriter {
 public:
  CodeWriter(std::ostream &out, const tflite::SubGraph *subgraph);

  void writeBuiltin(tflite::BuiltinOperator op, const void *data,
                    const std::string &name);

  // Write IntArray with variable declaration.
  void writeIntArray(const TfLiteIntArray &arr, const std::string &name);
  // Write only the comma separated contents of an IntArray.
  void writeIntArrayData(const TfLiteIntArray &arr);

  void writeTensor(const TfLiteTensor &t, const std::string &name);

  void writeQuantization(const TfLiteQuantization &q, const std::string &name);

#if TF_LITE_PACKED_QUANTIZED_DATA_VERSION == 100
  void writeQuantizationDetails(const TfLiteQuantization &q,
                                const std::string &name);
#endif

  template <typename T>
  CodeWriter &operator<<(T &&value) {
    out_ << std::forward<T>(value);
    return *this;
  }

 private:
  std::ostream &out_;
  const tflite::SubGraph *subgraph_ = nullptr;
};

}  // namespace tflmc

#endif
