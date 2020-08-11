#ifndef TFLMCOMPILER_CODEWRITER_H
#define TFLMCOMPILER_CODEWRITER_H

#include <iostream>
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/core/api/error_reporter.h"
#include "tensorflow/lite/version.h"

namespace tflmc {

// Helper functions for top-level code generation.
class CodeWriter {
 public:
  CodeWriter(std::ostream &out, const tflite::SubGraph *subgraph,
             tflite::ErrorReporter &errReporter);

  void writeBuiltin(tflite::BuiltinOperator op, const void *data,
                    const std::string &name);
                    
  void writeCustom(uint8_t const *opdata, size_t node_i, size_t opdata_size);

  std::pair<std::string, std::string> getBuiltinStrings(tflite::BuiltinOperator op,
                                                        const void* data);

  // Write IntArray with variable declaration.
  void writeIntArray(const TfLiteIntArray &arr, const std::string &name);
  // Write only the comma separated contents of an IntArray.
  void writeIntArrayData(const TfLiteIntArray &arr);

  void writeTensor(const TfLiteTensor &t, const std::string &name);

  void writeQuantization(const TfLiteQuantization &q, const std::string &name);

  void writeTensorArena(size_t tensor_arena_size);

template<class Container>
void writeArray(const Container &container, size_t elt_size, bool is_const,
   const char *decl, const char *name ) {

  out_ << decl << ' ' << name << R"([] = {
)";
  size_t elts = 0;
  for (auto &e : container) {
    out_ << std::to_string(e) << ",";
    ++elts;
    if (elts % 10 == 0) {
      out_ << "\n";
    } else { 
      out_ << " ";
    }
  }
  // To suppress warnings add dummy element if no scratch bufs
  if (container.empty()) {
    out_ << "0 // dummy to avoid empty vector";
  }
  out_ << R"(
};  
)";

  size_t footprint = elt_size * container.size();
  if (is_const) {
    const_data_usage_ += footprint;
  } else {
    init_data_usage_ += footprint;
  }
}



  template <typename T>
  CodeWriter &operator<<(T &&value) {
    out_ << std::forward<T>(value);
    return *this;
  }

  inline size_t initDataUsage() const { return init_data_usage_; }

  inline size_t uninitDataUsage() const { return uninit_data_usage_; }

  inline size_t constDataUsage() const { return const_data_usage_; }

 private:
  std::ostream &out_;
  const tflite::SubGraph *subgraph_ = nullptr;
  tflite::ErrorReporter &err_reporter_;

  size_t  init_data_usage_;
  size_t  uninit_data_usage_;
  size_t  const_data_usage_;
};

}  // namespace tflmc

#endif
