#include "CodeWriter.h"

#include <ctime>
#include <iomanip>

#include "TypeToString.h"
#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"

namespace {

class AllocatorToGetLastAllocSize : public tflite::BuiltinDataAllocator {
 public:
  void* Allocate(size_t size, size_t alignment_hint) override {
    lastAllocSize = size;
    return malloc(size);
  }
  void Deallocate(void* data) override { free(data); }
  size_t GetLastAllocSize() { return lastAllocSize; }

 private:
  size_t lastAllocSize = 0;
};
size_t GetBuiltinDataSize(tflite::BuiltinOperator opType,
                          const tflite::SubGraph* subgraph) {
  // There seems to be no simple query function for this, so tickle the
  // information out of the parse function.
  auto dummyOp = subgraph->operators()->Get(0);
  tflite::MicroErrorReporter errReporter;
  AllocatorToGetLastAllocSize allocator;
  void* outData = nullptr;
  if (tflite::ParseOpData(dummyOp, opType, &errReporter, &allocator,
                          &outData) == kTfLiteOk)
    free(outData);
  return allocator.GetLastAllocSize();
}

}  // namespace

tflmc::CodeWriter::CodeWriter(std::ostream& out,
                              const tflite::SubGraph* subgraph)
    : out_(out), subgraph_(subgraph) {
  // Setup stream: Print booleans as string:
  out_ << std::boolalpha;
  // Print floats with precision that is sufficient for exact back-conversion:
  out_ << std::setprecision(std::numeric_limits<double>::max_digits10);

  out_ << "// This file is generated. Do not edit.\n";
  {
    auto t = std::time(nullptr);
    auto tm = *std::localtime(&t);
    out_ << "// Generated on: " << std::put_time(&tm, "%d.%m.%Y %H:%M:%S")
         << "\n";
  }
}

void tflmc::CodeWriter::writeBuiltin(tflite::BuiltinOperator op,
                                     const void* data,
                                     const std::string& name) {
  using namespace tflmc;
  if (!data) {
    return;
  }
  out_ << "const ";
  switch (op) {
    case tflite::BuiltinOperator_CONV_2D: {
      out_ << "TfLiteConvParams " << name << " = { ";
      TfLiteConvParams const* p = (TfLiteConvParams const*)data;
      out_ << to_string(p->padding) << ", " << p->stride_width << ","
           << p->stride_height << ", " << to_string(p->activation) << ", "
           << p->dilation_width_factor << "," << p->dilation_height_factor
           << " };";
    } break;
    case tflite::BuiltinOperator_DEPTHWISE_CONV_2D: {
      out_ << "TfLiteDepthwiseConvParams " << name << " = { ";
      TfLiteDepthwiseConvParams const* p =
          (TfLiteDepthwiseConvParams const*)data;
      out_ << to_string(p->padding) << ", " << p->stride_width << ","
           << p->stride_height << ", " << p->depth_multiplier << ", "
           << to_string(p->activation) << ", " << p->dilation_width_factor
           << "," << p->dilation_height_factor << " };";
    } break;
    case tflite::BuiltinOperator_FULLY_CONNECTED: {
      out_ << "TfLiteFullyConnectedParams " << name << " = { ";
      TfLiteFullyConnectedParams const* p =
          (TfLiteFullyConnectedParams const*)data;
      out_ << to_string(p->activation) << ", " << to_string(p->weights_format)
           << ", " << p->keep_num_dims << ", " << p->asymmetric_quantize_inputs
           << " };";
    } break;
    case tflite::BuiltinOperator_MAX_POOL_2D:
    case tflite::BuiltinOperator_AVERAGE_POOL_2D: {
      out_ << "TfLitePoolParams " << name << " = { ";
      TfLitePoolParams const* p = (TfLitePoolParams const*)data;
      out_ << to_string(p->padding) << ", " << p->stride_width << ","
           << p->stride_height << ", " << p->filter_width << ","
           << p->filter_height << ", " << to_string(p->activation) << ", { "
           << to_string(p->computed.padding) << " } };";
    } break;
    case tflite::BuiltinOperator_RESHAPE: {
      out_ << "TfLiteReshapeParams " << name << " = { { ";
      TfLiteReshapeParams const* p = (TfLiteReshapeParams const*)data;
      for (uint32_t i = 0; i < TFLITE_RESHAPE_PARAMS_MAX_DIMENSION_COUNT; ++i)
        out_ << p->shape[i] << ", ";
      out_ << "}, " << p->num_dimensions << " };";
    } break;
    case tflite::BuiltinOperator_SOFTMAX: {
      out_ << "TfLiteSoftmaxParams " << name << " = { ";
      TfLiteSoftmaxParams const* p = (TfLiteSoftmaxParams const*)data;
      out_ << p->beta << " };";
    } break;
    case tflite::BuiltinOperator_ADD: {
      out_ << "TfLiteAddParams " << name << " = { ";
      TfLiteAddParams const* p = (TfLiteAddParams const*)data;
      out_ << to_string(p->activation) << " };";
    } break;
    case tflite::BuiltinOperator_MEAN: {
      out_ << "TfLiteReducerParams " << name << " = { ";
      TfLiteReducerParams const* p = (TfLiteReducerParams const*)data;
      out_ << p->keep_dims << " };";
    } break;
    case tflite::BuiltinOperator_MUL: {
      out_ << "TfLiteMulParams " << name << " = { ";
      TfLiteMulParams const* p = (TfLiteMulParams const*)data;
      out_ << to_string(p->activation) << " };";
    } break;
    case tflite::BuiltinOperator_PACK: {
      out_ << "TfLitePackParams " << name << " = { ";
      TfLitePackParams const* p = (TfLitePackParams const*)data;
      out_ << p->values_count << ", " << p->axis << " };";
    } break;
    case tflite::BuiltinOperator_SHAPE: {
      out_ << "TfLiteShapeParams " << name << " = { "
           << " };";
    } break;
    case tflite::BuiltinOperator_SUB: {
      out_ << "TfLiteSubParams " << name << " = { ";
      TfLiteSubParams const* p = (TfLiteSubParams const*)data;
      out_ << to_string(p->activation) << " };";
    } break;
    case tflite::BuiltinOperator_CONCATENATION: {
      out_ << "TfLiteConcatenationParams " << name << " = { ";
      TfLiteConcatenationParams const* p =
          (TfLiteConcatenationParams const*)data;
      out_ << p->axis << ", " << to_string(p->activation) << " };";
    } break;
    case tflite::BuiltinOperator_STRIDED_SLICE: {
      out_ << "TfLiteStridedSliceParams " << name << " = { ";
      TfLiteStridedSliceParams const* p = (TfLiteStridedSliceParams const*)data;
      out_ << p->begin_mask << ", " << p->end_mask << ", " << p->ellipsis_mask
           << ", " << p->new_axis_mask << ", " << p->shrink_axis_mask << " };";
    } break;
    case tflite::BuiltinOperator_TRANSPOSE_CONV: {
      out_ << "TfLiteTransposeConvParams " << name << " = { ";
      TfLiteTransposeConvParams const* p =
          (TfLiteTransposeConvParams const*)data;
      out_ << to_string(p->padding) << ", " << p->stride_width << ", "
           << p->stride_height << " };";
    } break;
    default: {
      size_t datalen = GetBuiltinDataSize(op, subgraph_);
      uint32_t alignment = datalen >= 4 ? 4 : datalen >= 2 ? 2 : 1;
      out_ << "ALIGN(" << alignment << ") uint8_t " << name << "[" << datalen
           << "] = { ";
      for (uint32_t i = 0; i < datalen; ++i)
        out_ << int(((uint8_t const*)data)[i]) << ", ";
      out_ << " }; /* op type " << int(op) << "="
           << tflite::EnumNameBuiltinOperator(op) << " */";
    } break;
  }
  out_ << '\n';
}

void tflmc::CodeWriter::writeIntArray(const TfLiteIntArray& arr,
                                      const std::string& name) {
  if (arr.size == 0) {
    out_ << "const int " << name << " = 0; /* empty TfLiteIntArray */\n";
  } else {
    out_ << "const TfArray<" << arr.size << ", int> " << name << " = { "
         << arr.size << ", { ";
    writeIntArrayData(arr);
    out_ << " } };\n";
  }
}

void tflmc::CodeWriter::writeIntArrayData(const TfLiteIntArray& arr) {
  if (arr.size > 0) {
    out_ << arr.data[0];
    for (int i = 1; i < arr.size; i++) {
      out_ << ',' << arr.data[i];
    }
  }
}

// outputting int8_t as a character is not what we intend here, we want to see
// the value, so we introduce printT
template <class T, class printT>
static void dump_tensor_contents(std::ostream& out_, const TfLiteTensor& t,
                                 const std::string& tname,
                                 const std::string& name) {
  if (t.dims->size == 0) {  // special case 0 dimensions, we output an array to
                            // avoid distinction from >0 dimension at every use
    out_ << "const " << tname << " " << name << "[1] = { "
         << (printT)(tflite::GetTensorData<T>(&t)[0]) << " };\n";
    return;
  }

  uint32_t alignment = t.bytes >= 8 ? 8 : t.bytes >= 4 ? 4 : 2;

  // For packed formats the numer of serialized data items may not
  // necessarily match the nominal dimensions of the tensor.
  // We need to ensure this case is handled correctly.
  size_t nominal_elts = 1;
  for (int i = 0; i < t.dims->size; ++i) {
    nominal_elts *= t.dims->data[i];
  }

  size_t serialized_elts = t.bytes / sizeof(T);

  out_ << "const ALIGN(" << alignment << ") " << tname << " " << name << "[";

  if (serialized_elts != nominal_elts) {
    out_ << serialized_elts << " /* PACKED ";
  }

  out_ << t.dims->data[0];
  for (int i = 1; i < t.dims->size; ++i) out_ << '*' << t.dims->data[i];
  if (serialized_elts != nominal_elts) {
    out_ << " */";
  }
  out_ << "] = { ";
  if (t.dims->size == 1 || serialized_elts != nominal_elts) {
    // one dimension/packed: 10 per line of data
    for (int i = 0; i < serialized_elts; ++i) {
      if (i % 10 == 0) out_ << "\n    ";
      out_ << (printT)(tflite::GetTensorData<T>(&t)[i]) << ", ";
    }
    out_ << "\n};\n";
  } else if (t.dims->size == 2) {
    // two dimensions: Inner dimension is one line
    for (int i = 0; i < t.dims->data[0]; ++i) {
      out_ << "\n  ";
      for (int j = 0; j < t.dims->data[1]; ++j)
        out_ << (printT)(tflite::GetTensorData<T>(&t)[i * t.dims->data[1] + j])
             << ", ";
    }
    out_ << "\n};\n";
  } else {
    // More dimensions: Inner two dimensions per line (space between two
    // middle elements)
    int outer_dim = t.dims->data[0];
    int middle_dim = t.dims->data[t.dims->size - 2];
    int inner_dim = t.dims->data[t.dims->size - 1];
    for (int i = 1; i < t.dims->size - 2; ++i) outer_dim *= t.dims->data[i];
    for (int i = 0; i < outer_dim; ++i) {
      // out_ << "\n  ";
      // uint32_t outer_index = inner_dim * middle_dim;
      // output a meaningful index for this line
      uint32_t idx = i;
      std::string indexstr = "[][]";
      for (int32_t j = t.dims->size - 3; j >= 0; --j) {
        uint32_t idx_j = idx % t.dims->data[j];
        indexstr = "[" + std::to_string(idx_j) + "]" + indexstr;
        idx /= t.dims->data[j];
        // if (j>0)
        // 	outer_index *= t.dims->data[j];
      }
      out_ << "\n  /* " << indexstr << " */ ";
      for (int j = 0; j < middle_dim; ++j) {
        for (int k = 0; k < inner_dim; ++k)
          out_ << (printT)(tflite::GetTensorData<T>(
                      &t)[(i * middle_dim + j) * inner_dim + k])
               << ",";
        out_ << " ";  // separator between middle indices
      }
    }
    out_ << "\n};\n";
  }
}

#define DUMP_TENSOR2(TfType, CType, PrintType)                     \
  case TfType:                                                     \
    dump_tensor_contents<CType, PrintType>(out_, t, #CType, name); \
    break

void tflmc::CodeWriter::writeTensor(const TfLiteTensor& t,
                                    const std::string& name) {
  switch (t.type) {
    DUMP_TENSOR2(kTfLiteFloat32, float, float);
    DUMP_TENSOR2(kTfLiteInt32, int32_t, int32_t);
    DUMP_TENSOR2(kTfLiteUInt8, uint8_t, int32_t);
    DUMP_TENSOR2(kTfLiteInt64, int64_t, int64_t);
    // DUMP_TENSOR2(kTfLiteString);
    // DUMP_TENSOR2(kTfLiteBool, bool);
    DUMP_TENSOR2(kTfLiteInt16, int16_t, int16_t);
    // DUMP_TENSOR2(kTfLiteComplex64);
    DUMP_TENSOR2(kTfLiteInt8, int8_t, int32_t);
    // DUMP_TENSOR2(kTfLiteFloat16);
    DUMP_TENSOR2(kTfLiteFloat64, double, double);
    default: {
      out_ << "const ALIGN(4) uint8_t " << name << "[" << t.bytes << "] = { ";
      for (size_t i = 0; i < t.bytes; i++)
        out_ << int((uint8_t)t.data.raw_const[i]) << ",";
      out_ << " };\n";
    } break;
  }
}

void tflmc::CodeWriter::writeQuantization(const TfLiteQuantization& q,
                                          const std::string& name) {
  if (q.type == kTfLiteAffineQuantization) {
    auto aq = (TfLiteAffineQuantization const*)q.params;
    out_ << "const TfArray<" << aq->scale->size << ", float> " << name
         << "_scale = { " << aq->scale->size << ", { ";
    for (int i = 0; i < aq->scale->size; i++) {
      out_ << aq->scale->data[i] << ", ";
    }
    out_ << "} };\n";
    out_ << "const TfArray<" << aq->zero_point->size << ", int> " << name
         << "_zero = { " << aq->zero_point->size << ", { ";
    writeIntArrayData(*aq->zero_point);
    out_ << " } };\n";
    out_ << "const TfLiteAffineQuantization " << name << " = { "
         << "(TfLiteFloatArray*)&" << name << "_scale, "
         << "(TfLiteIntArray*)&" << name << "_zero, " << aq->quantized_dimension
         << " };\n";
  }
}

#if TF_LITE_PACKED_QUANTIZED_DATA_VERSION == 100
void tflmc::CodeWriter::writeQuantizationDetails(const TfLiteQuantization& q,
                                                 const std::string& name) {
  if (q.details.type == kTfLiteSub8BitPackedUniformDetail) {
    out_ << "const TfLiteCustomSub8BitPackingDetails " << name << " = { ";
    auto sub8_details = q.details.data.custom_sub8bit_packing;
    out_ << static_cast<unsigned>(sub8_details->bits_per_item) << ", ";
    out_ << static_cast<unsigned>(sub8_details->container_bits) << ", ";
    out_ << static_cast<unsigned>(sub8_details->packed_minor_dims) << ", ";
    out_ << "{}";
    out_ << "};\n";
  }
}
#endif
