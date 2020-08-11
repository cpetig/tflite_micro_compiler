#include "BuiltinAllocations.h"

#include <string>
#include <sstream>

#include "TypeToString.h"
#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/core/api/error_reporter.h"

namespace {

class AllocatorToGetLastAllocSize : public tflite::BuiltinDataAllocator {
 public:

  void* Allocate(size_t size, size_t alignment_hint) override {
    lastAllocSize = size;
    allocated_blocks.push_back(std::make_unique<uint8_t []>(size));
    return reinterpret_cast<void *>(allocated_blocks.back().get());
  }

  void Deallocate(void* data) override {
  }


  size_t GetLastAllocSize() { return lastAllocSize; }

 private:
  std::vector<std::unique_ptr<uint8_t []>>  allocated_blocks;
  size_t lastAllocSize = 0;
};

}  // namespace

namespace tflmc {
namespace BuiltinAllocations {

size_t GetBuiltinDataSize(tflite::BuiltinOperator opType,
                          const tflite::SubGraph* subgraph,
                          tflite::ErrorReporter &errReporter) {
  // There seems to be no simple query function for this, so tickle the
  // information out of the parse function.
  auto dummyOp = subgraph->operators()->Get(0);
  AllocatorToGetLastAllocSize allocator;
  void* outData = nullptr;
  if (tflite::ParseOpData(dummyOp, opType, &errReporter, &allocator,
                          &outData) != kTfLiteOk) {
    throw std::runtime_error("ERROR: Unable to use tflite::ParseOpData to extract the BuiltinDataSize!\n"
		             "tensorflow/lite/core/api/flatbuffer_conversions.cc needs a patch to support this feature...");
  }

  return allocator.GetLastAllocSize();
}

std::pair<std::string, std::string> getBuiltinStrings(tflite::BuiltinOperator op,
                                                      const void* data) {
  using namespace tflmc;
  std::stringstream builtinOptionsName, builtinOptionsStruct;
  switch (op) {
    case tflite::BuiltinOperator_CONV_2D: {
      builtinOptionsName << "TfLiteConvParams";
      TfLiteConvParams const* p = (TfLiteConvParams const*)data;
      builtinOptionsStruct << "{ " << to_string(p->padding) << ", " << p->stride_width << ","
                           << p->stride_height << ", " << to_string(p->activation) << ", "
                           << p->dilation_width_factor << "," << p->dilation_height_factor
                           << " }";
    } break;
    case tflite::BuiltinOperator_DEPTHWISE_CONV_2D: {
      builtinOptionsName << "TfLiteDepthwiseConvParams";
      TfLiteDepthwiseConvParams const* p =
          (TfLiteDepthwiseConvParams const*)data;
      builtinOptionsStruct << "{ " << to_string(p->padding) << ", " << p->stride_width << ","
                           << p->stride_height << ", " << p->depth_multiplier << ", "
                           << to_string(p->activation) << ", " << p->dilation_width_factor
                           << "," << p->dilation_height_factor << " }";
    } break;
    case tflite::BuiltinOperator_FULLY_CONNECTED: {
      builtinOptionsName << "TfLiteFullyConnectedParams";
      TfLiteFullyConnectedParams const* p =
          (TfLiteFullyConnectedParams const*)data;
      builtinOptionsStruct << "{ " << to_string(p->activation) << ", " << to_string(p->weights_format)
                           << ", " << p->keep_num_dims << ", " << p->asymmetric_quantize_inputs
                           << " }";
    } break;
    case tflite::BuiltinOperator_MAX_POOL_2D:
    case tflite::BuiltinOperator_AVERAGE_POOL_2D: {
      builtinOptionsName << "TfLitePoolParams";
      TfLitePoolParams const* p = (TfLitePoolParams const*)data;
      builtinOptionsStruct << "{ " << to_string(p->padding) << ", " << p->stride_width << ","
                           << p->stride_height << ", " << p->filter_width << ","
                           << p->filter_height << ", " << to_string(p->activation) << ", { "
                           << to_string(p->computed.padding) << " } }";
    } break;
    case tflite::BuiltinOperator_RESHAPE: {
      builtinOptionsName << "TfLiteReshapeParams";
      builtinOptionsStruct << "{ {";
      TfLiteReshapeParams const* p = (TfLiteReshapeParams const*)data;
      for (uint32_t i = 0; i < TFLITE_RESHAPE_PARAMS_MAX_DIMENSION_COUNT; ++i)
        builtinOptionsStruct << p->shape[i] << ", ";
      builtinOptionsStruct << "}, " << p->num_dimensions << " }";
    } break;
    case tflite::BuiltinOperator_SOFTMAX: {
      builtinOptionsName << "TfLiteSoftmaxParams";
      TfLiteSoftmaxParams const* p = (TfLiteSoftmaxParams const*)data;
      builtinOptionsStruct << "{ " << p->beta << " }";
    } break;
    case tflite::BuiltinOperator_ADD: {
      builtinOptionsName << "TfLiteAddParams";
      TfLiteAddParams const* p = (TfLiteAddParams const*)data;
      builtinOptionsStruct << "{ " << to_string(p->activation) << ", "
                                   << p->pot_scale_int16 << " }";
    } break;
    case tflite::BuiltinOperator_MUL: {
      builtinOptionsName << "TfLiteMulParams";
      TfLiteMulParams const* p = (TfLiteMulParams const*)data;
      builtinOptionsStruct << "{ " << to_string(p->activation) << " }";
    } break;
    case tflite::BuiltinOperator_SUB: {
      builtinOptionsName << "TfLiteSubParams";
      TfLiteSubParams const* p = (TfLiteSubParams const*)data;
      builtinOptionsStruct << "{ " << to_string(p->activation) << ", "
                                   << p->pot_scale_int16 << " }";
    } break;
    case tflite::BuiltinOperator_CONCATENATION: {
      builtinOptionsName << "TfLiteConcatenationParams";
      TfLiteConcatenationParams const* p =
          (TfLiteConcatenationParams const*)data;
      builtinOptionsStruct << "{ " << p->axis << ", " << to_string(p->activation) << " }";
    } break;
    default: {
    } break;
  }
  return std::make_pair(builtinOptionsName.str(), builtinOptionsStruct.str());
}

}  // namespace BuiltinAllocations
}  // namespace tflmc
