
#include "Compiler.h"

#include <fstream>
#include <vector>

#include "CodeWriter.h"
#include "CustomOperators.h"
#include "RecordAllocations.h"
#include "TypeToString.h"
#include "tensorflow/lite/version.h"

bool tflmc::CompileFile(const std::string &modelFileName,
                        const std::string &outFileName,
                        const std::string &prefix) {
  // Load model flatbuffer.
  std::ifstream model_file(modelFileName, std::ios::binary | std::ios::ate);
  auto sz = model_file.tellg();
  model_file.seekg(0, std::ios::beg);
  std::vector<char> model_data(sz);
  if (!model_file.read(model_data.data(), sz)) {
    std::cerr << "Failed to read model file\n";
    return false;
  }

  std::ofstream outFile(outFileName);
  if (!outFile) {
    std::cerr << "Failed to create output file\n";
    return false;
  }

  try {
    Compiler compiler(model_data.data(), prefix);
    compiler.writeSource(outFile);
    return true;
  } catch (const std::exception &e) {
    std::cerr << e.what() << "\n";
  } catch (...) {
    std::cerr << "Unknown exception\n";
  }

  return false;
}

tflmc::Compiler::Compiler(const void *modelData, const std::string &prefix)
    : prefix_(prefix) {
  if (!init(modelData)) {
    throw std::runtime_error("Could not set up compiler");
  }
}

bool tflmc::Compiler::init(const void *modelData) {
  model_ = tflite::GetModel(modelData);
  if (model_->version() != TFLITE_SCHEMA_VERSION) {
    errReporter().Report(
        "Model provided is schema version %d not equal "
        "to supported version %d.",
        model_->version(), TFLITE_SCHEMA_VERSION);
    return false;
  }

  auto subgraphs = model_->subgraphs();
  if (subgraphs->size() != 1) {
    std::cerr << "Model needs to have exactly one subgraph as expected by TF "
                 "Lite for Micro\n";
    return false;
  }
  subgraph_ = (*subgraphs)[0];
  auto tensors = subgraph_->tensors();
  auto operators = subgraph_->operators();
  if (subgraph_->inputs()->size() == 0 || subgraph_->outputs()->size() == 0) {
    std::cerr << "No inputs or no outputs found in model\n";
    return false;
  }
  for (auto inIndex : *subgraph_->inputs()) {
    inputTensorIndices_.push_back(inIndex);
  }
  for (auto outIndex : *subgraph_->outputs()) {
    outputTensorIndices_.push_back(outIndex);
  }
  tflmc::custom_operator_handle custom = tflmc::LoadCustom(&resolver_);

  // Build an interpreter to run the model with.
  arena_buf_.resize(SUFFICIENT_ARENA_SIZE);
  interpreter_ = std::make_unique<tflite::MicroInterpreter>(
      model_, resolver_, arena_buf_.data(), arena_buf_.size(),
      &microErrReporter_);

  // Allocate memory from the tensor_arena for the model's tensors.
  TfLiteStatus allocate_status = interpreter_->AllocateTensors();
  if (allocate_status != kTfLiteOk) {
    errReporter().Report("AllocateTensors() failed");
    return false;
  }

  ptrdiff_t ramTensorBufferSize = 0;
  ptrdiff_t romOffset = 0;
  for (size_t i = 0; i < interpreter_->tensors_size(); i++) {
    auto tensor = interpreter_->tensor(i);
    tensors_.push_back({tensor});
    if (tensor->allocation_type == kTfLiteMmapRo) {
      memMap_.recordROM(romOffset, tensor->bytes, getTensorName(i));
      romOffset += tensor->bytes;
    } else {
      ptrdiff_t offset = (uint8_t *)tensor->data.data - arena_buf_.data();
      ptrdiff_t highSize = offset + tensor->bytes;
      ramTensorBufferSize = std::max(ramTensorBufferSize, highSize);
      memMap_.recordRAM(offset, tensor->bytes, getTensorName(i));
    }
  }

  for (size_t i = 0; i < interpreter_->operators_size(); i++) {
    auto nodeAndReg = interpreter_->node_and_registration(i);
    auto node = &nodeAndReg.node;
    auto reg = nodeAndReg.registration;
    auto code = tflite::EnumValuesBuiltinOperator()[reg->builtin_code];

    printf("operation %lu: %s\n", i, tflite::EnumNamesBuiltinOperator()[code]);

    RegistrationInfo regInfo{reg, code};
    if (code == tflite::BuiltinOperator_CUSTOM) {
      regInfo.custom_name = reg->custom_name;
      has_custom_ops = true;
    }
    auto itOp =
        std::find(registrations_.begin(), registrations_.end(), regInfo);
    if (itOp == registrations_.end()) {
      itOp = registrations_.insert(registrations_.end(), regInfo);
    }

    // There doesn't seem to be a way to get the node pointer, so copy it.
    nodes_.push_back({*node, itOp - registrations_.begin()});
  }

  auto runtimeAllocations = tflmc::RecordAllocations(model_);
  ptrdiff_t minRuntimeOffset = 0;  // These are negative so zero start is fine.
  for (const auto &alloc : runtimeAllocations) {
    minRuntimeOffset = std::min(minRuntimeOffset, alloc.offset);
  }
  size_t totalRuntimeAllocSize = 0;
  for (const auto &alloc : runtimeAllocations) {
    // TODO: This drops the alignment between buffers. Is this fine?
    totalRuntimeAllocSize += alloc.len;
    ptrdiff_t offset = alloc.offset - minRuntimeOffset + ramTensorBufferSize;
    memMap_.recordRAM(offset, alloc.len,
                      "PersistentBuf" + std::to_string(alloc.nodeIndex));
  }

  // This includes:
  // - Tensors
  // - Scratch buffers
  // - Persistent buffers
  // tensor metadata is not included, since we declare them outside the arena
  arenaBufferSize_ = ramTensorBufferSize + totalRuntimeAllocSize;

  // TODO: This is overestimating by quite a bit because of ABI differences.
  size_t tensorMetaSize = tensors_.size() * sizeof(TfLiteTensor);
  size_t nodeMetaSize = nodes_.size() * sizeof(TfLiteNode);
  memMap_.recordRAM(arenaBufferSize_, tensorMetaSize, "TensorMetadata");
  memMap_.recordRAM(arenaBufferSize_ + tensorMetaSize, nodeMetaSize,
                    "NodeMetadata");
  memMap_.recordRAM(arenaBufferSize_ + tensorMetaSize + nodeMetaSize,
                    sizeof(TfLiteContext), "TfLiteContext");

  memMap_.report();
  tflmc::UnloadCustom(custom);

  return true;
}

void tflmc::Compiler::writeSource(std::ostream &out) {
  CodeWriter wr(out, subgraph_);

  wr << R"(
#include <cassert>

#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/micro/kernels/micro_ops.h"

)";
  // declare custom registrations
  if (has_custom_ops) {
    wr << R"(namespace tflite {
namespace ops {
namespace micro {
)";
    for (size_t i = 0; i < registrations_.size(); i++) {
      if (registrations_[i].code == tflite::BuiltinOperator_CUSTOM) {
        wr << "extern TfLiteRegistration *Register_"
           << registrations_[i].custom_name << "(void);\n";
      }
    }
    wr << R"(}  // namespace micro
}  // namespace ops
}  // namespace tflite

)";
  }
  wr << R"(namespace {

constexpr int kTensorArenaSize = )"
     << arenaBufferSize_ << R"(;
uint8_t g_tensor_arena[kTensorArenaSize] __attribute__((aligned(16)));
template <int SZ, class T> struct TfArray {
  int sz; T elem[SZ];
};
enum used_operators_e {
  )";
  for (size_t i = 0; i < registrations_.size(); i++) {
    if (registrations_[i].code == tflite::BuiltinOperator_CUSTOM) {
      wr << "OP_" << registrations_[i].custom_name << ", ";
    } else {
      wr << "OP_" << tflite::EnumNameBuiltinOperator(registrations_[i].code)
         << ", ";
    }
  }
  wr << R"( OP_LAST
};
struct TensorInfo_t { // subset of TfLiteTensor used for initialization from constant memory
  TfLiteType type;
  void* data;
  TfLiteIntArray* dims;
  // TfLiteQuantizationParams params;
  // TfLiteAllocationType allocation_type;
  size_t bytes;
  const char* name;
  TfLiteQuantization quantization;
};
struct NodeInfo_t { // subset of TfLiteNode used for initialization from constant memory
  struct TfLiteIntArray* inputs;
  struct TfLiteIntArray* outputs;
  void* builtin_data;
  used_operators_e used_op_index;
};

TfLiteContext g_ctx{};
TfLiteTensor g_tensors[)"
     << tensors_.size() << R"(];
TfLiteRegistration *g_registrations[OP_LAST];
TfLiteNode g_nodes[)"
     << nodes_.size() << R"(];

)";
  for (size_t i = 0; i < tensors_.size(); i++) {
    auto &t = tensors_[i].tensor;
    if (t->allocation_type == kTfLiteMmapRo) {
      wr.writeTensor(*t, prefix_ + "tensor_data" + std::to_string(i));
    }
    wr.writeIntArray(*t->dims,
                     prefix_ + "tensor_dimension" + std::to_string(i));
    wr.writeQuantization(t->quantization,
                         prefix_ + "quant" + std::to_string(i));
  }
  for (size_t i = 0; i < nodes_.size(); i++) {
    auto &node = nodes_[i].node;
    auto &regInfo = registrations_[nodes_[i].regIndex];
    if (regInfo.code == tflite::BuiltinOperator_CUSTOM) {
      wr << "uint8_t " << prefix_ + "opdata" + std::to_string(i) << "["
         << node.custom_initial_data_size << "] = { ";
      for (uint32_t i = 0; i < node.custom_initial_data_size; ++i)
        wr << int(((uint8_t const *)node.custom_initial_data)[i]) << ", ";
      wr << " }; /* custom_initial_data */\n";
    } else {
      wr.writeBuiltin(regInfo.code, node.builtin_data,
                      prefix_ + "opdata" + std::to_string(i));
    }
    wr.writeIntArray(*node.inputs, prefix_ + "inputs" + std::to_string(i));
    wr.writeIntArray(*node.outputs, prefix_ + "outputs" + std::to_string(i));
  }
  wr << R"(const TensorInfo_t tensors[] = {
)";
  for (size_t i = 0; i < tensors_.size(); i++) {
    auto &t = tensors_[i].tensor;
    wr << "  { " << tflmc::to_string(t->type) << ", ";
    if (t->allocation_type == kTfLiteMmapRo) {
      wr << "(void*)" << prefix_ << "tensor_data" << i;
    } else {
      wr << "g_tensor_arena + "
         << ((uintptr_t)t->data.data - (uintptr_t)arena_buf_.data());
    }
    wr << ", "
       << "(TfLiteIntArray*)&" << prefix_ << "tensor_dimension" << i << ", ";
    wr << t->bytes << ", ";
    wr << "\"" << ((t->name) ? t->name : "NO-NAME") << "\", ";
    if (t->quantization.type == kTfLiteAffineQuantization) {
      wr << "{kTfLiteAffineQuantization, "
            "const_cast<void*>(static_cast<const void*>(&"
         << prefix_ << "quant" << i << "))}, ";
    } else {
      wr << "{kTfLiteNoQuantization, nullptr}, ";
    }
    wr << "},\n";
  }
  wr << "};";
  wr << R"(const NodeInfo_t nodes[] = {
)";
  for (size_t i = 0; i < nodes_.size(); i++) {
    wr << "  { (TfLiteIntArray*)&" << prefix_ << "inputs" << i << ", ";
    wr << "(TfLiteIntArray*)&" << prefix_ << "outputs" << i << ", ";
    // TODO: Is this cast safe or does the data need to be non-const?
    // CP: I think so (as it typically just carries the trained operator
    // parameters) CP: Also if it were written to, we would see a segfault
    // (write to text segment)
    if (nodes_[i].node.builtin_data || nodes_[i].node.custom_initial_data) {
      wr << "const_cast<void*>(static_cast<const void*>(&" << prefix_
         << "opdata" << i << ")), ";
    } else {
      wr << "nullptr, ";
    }
    auto regI = nodes_[i].regIndex;
    if (registrations_[i].code == tflite::BuiltinOperator_CUSTOM) {
      wr << "OP_" << registrations_[i].custom_name << ", ";
    } else {
      wr << "OP_" << tflite::EnumNameBuiltinOperator(registrations_[regI].code)
         << ", ";
    }
    wr << "},\n";
  }
  wr << "};";
  // TODO: This code assumes that persistent allocations are made from the end
  // (which is true for the current implementation)
  wr << R"(
static TfLiteStatus AllocatePersistentBuffer(struct TfLiteContext* ctx,
                                                 size_t bytes, void** ptr) {
  static uint8_t *AllocPtr = g_tensor_arena + sizeof(g_tensor_arena);

  AllocPtr -= bytes;
  *ptr = AllocPtr;
  return kTfLiteOk;
}
} // namespace

TfLiteStatus )"
     << prefix_ << R"(init() {
  g_ctx.AllocatePersistentBuffer = &AllocatePersistentBuffer;
  g_ctx.tensors = g_tensors;
)";
  wr << "  g_ctx.tensors_size = " << tensors_.size() << ";\n";
  // TODO: Do we really support variable tensors?
  // TODO: Do we encounter other than kTfLiteMmapRo and kTfLiteArenaRw, if so we
  // need to store the type separately.
  wr << "  for(size_t i = 0; i < " << tensors_.size() << R"(; ++i) {
    g_tensors[i].data.data = tensors[i].data;
    g_tensors[i].type = tensors[i].type;
    g_tensors[i].is_variable = false;
    g_tensors[i].allocation_type = (g_tensor_arena <= tensors[i].data && tensors[i].data < g_tensor_arena + kTensorArenaSize) ? kTfLiteArenaRw : kTfLiteMmapRo;
    g_tensors[i].bytes = tensors[i].bytes;
    g_tensors[i].dims = tensors[i].dims;
    g_tensors[i].quantization = tensors[i].quantization;
    if (tensors[i].quantization.type == kTfLiteAffineQuantization) {
      TfLiteAffineQuantization const* quant = ((TfLiteAffineQuantization const*)(tensors[i].quantization.params));
      g_tensors[i].params.scale = quant->scale->data[0];
      g_tensors[i].params.zero_point = quant->zero_point->data[0];
    }
  }
)";
  for (size_t i = 0; i < registrations_.size(); i++) {
    std::string opName;
    if (registrations_[i].code == tflite::BuiltinOperator_CUSTOM) {
      opName = registrations_[i].custom_name;
    } else {
      opName = tflite::EnumNameBuiltinOperator(registrations_[i].code);
    }
    wr << "  g_registrations[OP_" << opName
       << "] = tflite::ops::micro::Register_" << opName << "();\n";
  }
  wr << "\n";
  wr << "  for(size_t i = 0; i < " << nodes_.size() << R"(; ++i) {
    g_nodes[i].inputs = nodes[i].inputs;
    g_nodes[i].outputs = nodes[i].outputs;
    g_nodes[i].temporaries = nullptr;
    g_nodes[i].builtin_data = nodes[i].builtin_data;
    g_nodes[i].custom_initial_data = nullptr;
    g_nodes[i].custom_initial_data_size = 0;
    g_nodes[i].delegate = nullptr;
    if (g_registrations[nodes[i].used_op_index]->init) {
      g_nodes[i].user_data = g_registrations[nodes[i].used_op_index]->init(&g_ctx, (const char*)g_nodes[i].builtin_data, 0);
    }
  }
)";
  wr << "  for(size_t i = 0; i < " << nodes_.size() << R"(; ++i) {
    if (g_registrations[nodes[i].used_op_index]->prepare) {
      TfLiteStatus status = g_registrations[nodes[i].used_op_index]->prepare(&g_ctx, &g_nodes[i]);
      if (status != kTfLiteOk) {
        return status;
      }
    }
  }
  return kTfLiteOk;
}

static const int inTensorIndices[] = {
  )";
  for (auto inIndex : inputTensorIndices_) {
    out << inIndex << ", ";
  }
  out << R"(
};
void *)"
      << prefix_ << R"(input_ptr(int index) {
  return g_ctx.tensors[inTensorIndices[index]].data.data;
}
size_t )"
      << prefix_ << R"(input_size(int index) {
  return g_ctx.tensors[inTensorIndices[index]].bytes;
}
TfLiteTensor* )"
      << prefix_ << R"(input(int index) {
  return &g_ctx.tensors[inTensorIndices[index]];
}

static const int outTensorIndices[] = {
  )";  // TODO: perhaps use a smaller type than int?
  for (auto outIndex : outputTensorIndices_) {
    out << outIndex << ", ";
  }
  out << R"(
};
const void *)"
      << prefix_ << R"(output_ptr(int index) {
  return g_ctx.tensors[outTensorIndices[index]].data.data;
}
size_t )"
      << prefix_ << R"(output_size(int index) {
  return g_ctx.tensors[outTensorIndices[index]].bytes;
}
TfLiteTensor* )"
      << prefix_ << R"(output(int index) {
  return &g_ctx.tensors[outTensorIndices[index]];
}

TfLiteStatus )"
      << prefix_ << R"(invoke() {
  for(size_t i = 0; i < )"
      << nodes_.size() << R"(; ++i) {
    TfLiteStatus status = g_registrations[nodes[i].used_op_index]->invoke(&g_ctx, &g_nodes[i]);
    if (status != kTfLiteOk) {
      return status;
    }
  }
  return kTfLiteOk;
}
)";
}

std::string tflmc::Compiler::getTensorName(int tensorIndex) const {
  TfLiteTensor *tensor = interpreter_->tensor(tensorIndex);

  std::stringstream ss;
  ss << (tensor->allocation_type == kTfLiteMmapRo ? "ROM" : "RAM") << "Tensor_";

  auto nOps = interpreter_->operators_size();
  for (int i = 0; i < nOps; i++) {
    auto nodeAndReg = interpreter_->node_and_registration(i);
    auto node = &nodeAndReg.node;

    auto checkAndAdd = [&](const TfLiteIntArray *indices,
                           const std::string &tag) {
      if (indices) {
        for (int k = 0; k < indices->size; k++) {
          if (indices->data[k] == tensorIndex) {
            ss << "L" << i << tag;
          }
        }
      }
    };

    checkAndAdd(node->inputs, "in");
    checkAndAdd(node->outputs, "out");
    checkAndAdd(node->intermediates, "int");
    checkAndAdd(node->temporaries, "tmp");
  }

  return ss.str();
}
