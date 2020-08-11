#include "CodeWriter.h"

#include <ctime>
#include <iomanip>

#include "BuiltinAllocations.h"
#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/core/api/error_reporter.h"

tflmc::CodeWriter::CodeWriter(std::ostream& out,
                              const tflite::SubGraph* subgraph,
                              tflite::ErrorReporter &err_reporter
                              )
    : out_(out), subgraph_(subgraph)
    , err_reporter_(err_reporter)
    , init_data_usage_(0)
    , uninit_data_usage_(0)
    , const_data_usage_(0)
{
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
  auto builtin_strings = BuiltinAllocations::getBuiltinStrings(op, data);
  if (!builtin_strings.first.empty() && !builtin_strings.first.empty()) {
    out_ << builtin_strings.first << " " << name << " = "
         << builtin_strings.second << ";";
  } else {
    size_t datalen = BuiltinAllocations::GetBuiltinDataSize(op, subgraph_, err_reporter_);
    uint32_t alignment = datalen >= 4 ? 4 : datalen >= 2 ? 2 : 1;
    out_ << "ALIGN(" << alignment << ") uint8_t " << name << "[" << datalen
         << "] = { ";
    for (uint32_t i = 0; i < datalen; ++i)
      out_ << int(((uint8_t const*)data)[i]) << ", ";
    out_ << " }; /* op type " << int(op) << "="
         << tflite::EnumNameBuiltinOperator(op) << " */";
  }
  out_ << '\n';
}


                    
void tflmc::CodeWriter::writeCustom(uint8_t const *opdata, size_t node_i, size_t opdata_size) {
    out_ << "uint8_t ALIGN(4) opdata" + std::to_string(node_i) << "["
        << opdata_size << "] = { ";
    for (size_t j = 0; j < opdata_size; ++j)
      out_ << int(opdata[j]) << ", ";
    out_ << " }; /* custom_initial_data */\n";
    const_data_usage_ += opdata_size;
    init_data_usage_ += opdata_size;
}

template<class TFArray>
size_t writeTfArray( std::ostream &os, const TFArray *tfarray, const std::string &name, const char * suffix, const char *data_type_id)
{
    os << "const TfArray<" 
          << tfarray->size << ", " 
          << data_type_id << "> " 
       << name << suffix
       << " = { " << tfarray->size << ", { ";
    for (int i = 0; i < tfarray->size; i++) {
      os << tfarray->data[i] << ", ";
    }
    os << "} };\n";
    return tfarray->size+1;
}

void tflmc::CodeWriter::writeIntArray(const TfLiteIntArray& arr,
                                      const std::string& name) {
  if (arr.size == 0) {
    out_ << "const int " << name << " = 0; /* empty TfLiteIntArray */\n";
    const_data_usage_ += sizeof(int);
  } else {
    auto arr_size = writeTfArray(out_, &arr, name, "", "int");
    const_data_usage_ += sizeof(int)*arr_size;
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
static size_t dump_tensor_contents(std::ostream& out_, const TfLiteTensor& t,
                                 const std::string& tname,
                                 const std::string& name) {

  size_t mem_size; 
  if (t.dims->size == 0) {  // special case 0 dimensions, we output an array to
                            // avoid distinction from >0 dimension at every use
    out_ << "const " << tname << " " << name << "[1] = { "
         << (printT)(tflite::GetTensorData<T>(&t)[0]) << " };\n";
    return sizeof(T);
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
  if (t.dims->size == 1 || serialized_elts != nominal_elts)  // one dimension/packed: 10 per line of data
  {
    for (size_t i = 0; i < serialized_elts; ++i) {
        if (i%10 == 0)
          out_ << "\n    ";
      out_ << (printT)(tflite::GetTensorData<T>(&t)[i]) << ", "; 
    }
    out_ << "\n};\n";
    mem_size = serialized_elts*sizeof(T);
  } else if (t.dims->size == 2) {
    // two dimensions: Inner dimension is one line
    for (int i = 0; i < t.dims->data[0]; ++i) {
      out_ << "\n  ";
      for (int j = 0; j < t.dims->data[1]; ++j)
        out_ << (printT)(tflite::GetTensorData<T>(&t)[i * t.dims->data[1] + j])
             << ", ";
    }
    out_ << "\n};\n";
    mem_size = nominal_elts*sizeof(T);
  } else {
    // More dimensions: Inner two dimensions per line (space between two
    // middle elements)
    int outer_dim = t.dims->data[0];
    int middle_dim = t.dims->data[t.dims->size - 2];
    int inner_dim = t.dims->data[t.dims->size - 1];
    for (int i = 1; i < t.dims->size - 2; ++i)
      outer_dim *= t.dims->data[i];
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
    mem_size = nominal_elts*sizeof(T);
  }
  return mem_size;
}

#define DUMP_TENSOR2(TfType, CType, PrintType)                     \
  case TfType:                                                     \
    const_data_usage_ += dump_tensor_contents<CType, PrintType>(out_, t, #CType, name); \
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
      const_data_usage_ += t.bytes;
    } break;
  }
}


static void writeAffineQuantizationFields(std::ostream &out, const std::string& name, TfLiteAffineQuantization const *aq) {

  out << "{ "
         << "(TfLiteFloatArray*)&" << name << "_scale, "
         << "(TfLiteIntArray*)&" << name << "_zero, " << aq->quantized_dimension
         << " }";
}


#if SUPPORT_CUSTOM_QUANT
static void writeQuantizationDetails(
    std::ostream& out, const TfLiteCustomSub8BitPackingDetails* sub8_details,
    const std::string& name) {
    out << "const TfLiteCustomSub8BitPackingDetails " << name << " = { ";
    out << static_cast<unsigned>(sub8_details->bits_per_item) << ", ";
    out << static_cast<unsigned>(sub8_details->container_bits) << ", ";
    out << static_cast<unsigned>(sub8_details->packed_minor_dims) << ", ";
    out << static_cast<unsigned>(sub8_details->sparsity_coding) << ", ";
    out << "{}";
    out << "};\n";
}
#endif  // SUPPORT_CUSTOM_QUANT

void tflmc::CodeWriter::writeQuantization(const TfLiteQuantization& q,
                                          const std::string& name) {
  if (q.type == kTfLiteAffineQuantization) {
    auto aq = (TfLiteAffineQuantization const*)q.params;
    auto scale_size = writeTfArray(out_, aq->scale, name, "_scale", "float");
    auto zp_size = writeTfArray(out_,  aq->zero_point, name, "_zero", "int");
    const_data_usage_ += scale_size * sizeof(float)  + zp_size*sizeof(int);
    out_ << "const TfLiteAffineQuantization " << name << " = ";
    writeAffineQuantizationFields(out_, name, aq);
    out_ << ";\n";
    const_data_usage_ += sizeof(TfLiteAffineQuantization);
#if SUPPORT_CUSTOM_QUANT
  } else if (q.type == kTfLitePackedAffineQuantization) {
    auto paq = (TfLitePackedAffineQuantization const*)q.params;
    writeQuantizationDetails(out_, paq->custom_sub8bit_packing, name + "_packing");
    const_data_usage_ += sizeof(TfLiteCustomSub8BitPackingDetails);
    auto aq = &paq->affine;
    auto scale_size = writeTfArray(out_, aq->scale, name, "_scale", "float");
    auto zp_size = writeTfArray(out_,  aq->zero_point, name, "_zero", "int");
    const_data_usage_ += scale_size * sizeof(float)  + zp_size*sizeof(int);
    out_ << "const TfLitePackedAffineQuantization " << name << " = { ";
    writeAffineQuantizationFields(out_, name, aq);
    out_ << ", &" <<  name + "_packing" << "};\n";
    const_data_usage_ += sizeof(kTfLitePackedAffineQuantization);
#endif  // SUPPORT_CUSTOM_QUANT
  }
}

void tflmc::CodeWriter::writeTensorArena(size_t tensor_arena_size)
{
  out_ << R"(
constexpr int kTensorArenaSize = )"
     << tensor_arena_size << R"(;
uint8_t tensor_arena[kTensorArenaSize] ALIGN(16);
)";
  uninit_data_usage_ += tensor_arena_size;
}


  