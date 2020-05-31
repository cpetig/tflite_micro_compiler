#ifndef TFLMCOMPILER_COMPILER_H
#define TFLMCOMPILER_COMPILER_H

#include <iostream>

#include "tensorflow/lite/micro/kernels/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace tflmc {

bool CompileFile(const std::string &modelFileName,
                 const std::string &outFileName,
                 const std::string &prefix = "model_");

class Compiler {
 public:
  // modelData: Flatbuffer binary data.
  // prefix: This string is prepended to every global name.
  Compiler(const void *modelData, const std::string &prefix = "model_");

  void writeSource(std::ostream &out);
  void writeHeader(std::ostream &out);

 private:
  bool init(const void *modelData);
  tflite::ErrorReporter &errReporter() { return microErrReporter_; }

 private:
  struct TensorInfo {
    const TfLiteTensor *tensor = nullptr;
  };
  struct RegistrationInfo {
    const TfLiteRegistration *reg = nullptr;
    tflite::BuiltinOperator code;
    bool operator==(const RegistrationInfo &other) {
      return code == other.code;
    }
  };
  struct NodeInfo {
    TfLiteNode node;
    ptrdiff_t regIndex = -1;
  };

 private:
  std::string prefix_;
  tflite::MicroErrorReporter microErrReporter_;
  const tflite::Model *model_ = nullptr;
  const tflite::SubGraph *subgraph_ = nullptr;
  tflite::ops::micro::AllOpsResolver resolver_;
  std::vector<uint8_t> arena_buf_;
  std::unique_ptr<tflite::MicroInterpreter> interpreter_;

  size_t arenaBufferSize_ = 0;
  std::vector<TensorInfo> tensors_;
  std::vector<RegistrationInfo> registrations_;
  std::vector<NodeInfo> nodes_;
  std::vector<int32_t> inputTensorIndices_;
  std::vector<int32_t> outputTensorIndices_;
};

}  // namespace tflmc

#endif
