#ifndef TFLMCOMPILER_COMPILER_H
#define TFLMCOMPILER_COMPILER_H

#include <iostream>

#include "MemMap.h"
#include "tensorflow/lite/core/api/error_reporter.h"
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_allocator.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace tflmc {

class CodeWriter;

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

  // Returns a name that describes a tensors relation to network layers.
  std::string getTensorName(int tensorIndex) const;

  bool noErrorsReported() const;

 private:
  bool init(const void *modelData);
  tflite::ErrorReporter &errReporter() { return microErrReporter_; }

  void writeCustomRegistrationsSource(CodeWriter &wr);

  void writeTflNodesSource(CodeWriter &wr);

  void writeTensorDataSource(CodeWriter &wr);

  void writeTypesAndWorkingArraysSource(CodeWriter &wr);

  void writeNodeDataSource(CodeWriter &wr);

  void writeScratchBufferOffsets(CodeWriter &wr);

  void writeContextAllocationHandlersSource(CodeWriter &wr);

  void writeInitSource(CodeWriter &wr);

  void writeTensorAccessorsSource(CodeWriter &wr);

  void writeInvokeSource(CodeWriter &wr);

 private:
  struct TensorInfo {
    TensorInfo(const TfLiteTensor *tensor_ptr) : tensor(tensor_ptr) {}
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
    NodeInfo(TfLiteNode tfl_node, ptrdiff_t reg_index)
        : node(tfl_node), regIndex(reg_index) {}
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

 private:
  /**
   * @brief Error reporter that tracks if Error was reported.
   *
   */
  class TrackingErrorReporter : public tflite::ErrorReporter {
   public:
    ~TrackingErrorReporter() {}
    int Report(const char *format, va_list args) override;

    bool getErrorReported() const { return error_reported_; }

   private:
    bool error_reported_ = false;
  };

  std::string prefix_;
  TrackingErrorReporter microErrReporter_;
  const tflite::Model *model_ = nullptr;
  const tflite::SubGraph *subgraph_ = nullptr;
  tflite::AllOpsResolver resolver_;
  SufficientArena arena_;
  uint8_t *aligned_arena_start_;
  size_t arena_size_;
  std::unique_ptr<tflite::MicroInterpreter> interpreter_;

  // static tflite::MicroAllocator* allocator_;
  MemMap memMap_;

  size_t arenaBufferSize_ = 0;
  size_t scratchBuffersAllocated_ = 0;
  std::vector<TensorInfo> tensors_;
  std::vector<RegistrationInfo> registrations_;
  std::vector<NodeInfo> nodes_;
  std::vector<int32_t> inputTensorIndices_;
  std::vector<int32_t> outputTensorIndices_;

  bool has_custom_ops = false;
  bool has_quantization = false;
  Option<TfLiteType> common_tensor_type;
  Option<bool> common_tensor_is_variable;
};

}  // namespace tflmc

#endif
