#include "Compiler.h"

#include <fstream>
#include <vector>

#include "CodeWriter.h"
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
      memMap_.recordROM(romOffset, tensor->bytes,
                        "ROMTensor" + std::to_string(i));
      romOffset += tensor->bytes;
    } else {
      ptrdiff_t offset = (uint8_t *)tensor->data.data - arena_buf_.data();
      ptrdiff_t highSize = offset + tensor->bytes;
      ramTensorBufferSize = std::max(ramTensorBufferSize, highSize);
      memMap_.recordRAM(offset, tensor->bytes, "RAMTensor" + std::to_string(i));
    }
  }

  for (size_t i = 0; i < interpreter_->operators_size(); i++) {
    auto nodeAndReg = interpreter_->node_and_registration(i);
    auto node = &nodeAndReg.node;
    auto reg = nodeAndReg.registration;
    auto code = tflite::EnumValuesBuiltinOperator()[reg->builtin_code];

    printf("operation %lu: %s\n", i, tflite::EnumNamesBuiltinOperator()[code]);

    RegistrationInfo regInfo{reg, code};
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
  memMap_.recordRAM(arenaBufferSize_, tensors_.size() * sizeof(TfLiteTensor),
                    "TensorMetadata");

  memMap_.report();

  return true;
}

void tflmc::Compiler::writeSource(std::ostream &out) {
  CodeWriter wr(out, subgraph_);

  wr << R"(
#include <cassert>

#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/micro/kernels/micro_ops.h"

namespace {

constexpr int kTensorArenaSize = )"
     << arenaBufferSize_ << R"(;
uint8_t g_tensor_arena[kTensorArenaSize] __attribute__((aligned(16)));
template <int SZ, class T> struct TfArray {
  int sz; T elem[SZ];
};

TfLiteContext g_ctx{};
TfLiteTensor g_tensors[)"
     << tensors_.size() << R"(];
TfLiteRegistration *g_registrations[)"
     << registrations_.size() << R"(];
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
    wr.writeBuiltin(regInfo.code, node.builtin_data,
                    prefix_ + "opdata" + std::to_string(i));
    wr.writeIntArray(*node.inputs, prefix_ + "inputs" + std::to_string(i));
    wr.writeIntArray(*node.outputs, prefix_ + "outputs" + std::to_string(i));
  }
  // TODO: This code assumes that persistent allocations are made from the end
  // (which is true for the current implementation)
  wr << R"(
static TfLiteStatus FakeAllocatePersistentBuffer(struct TfLiteContext* ctx,
                                                 size_t bytes, void** ptr) {
  static uint8_t *fakeAllocPtr = g_tensor_arena + sizeof(g_tensor_arena);

  fakeAllocPtr -= bytes;
  *ptr = fakeAllocPtr;
  return kTfLiteOk;
}
} // namespace

void )"
     << prefix_ << R"(init() {
  g_ctx.AllocatePersistentBuffer = &FakeAllocatePersistentBuffer;
  g_ctx.tensors = g_tensors;
)";
  wr << "  g_ctx.tensors_size = " << tensors_.size() << ";\n";
  for (size_t i = 0; i < tensors_.size(); i++) {
    auto &t = tensors_[i].tensor;
    std::string tensorI = "  g_tensors[" + std::to_string(i) + "].";
    if (t->allocation_type == kTfLiteMmapRo) {
      wr << tensorI << "data.data = (void*)" << prefix_ << "tensor_data" << i
         << ";\n";
    } else {
      wr << tensorI << "data.data = g_tensor_arena + "
         << (uintptr_t)t->data.data - (uintptr_t)arena_buf_.data() << ";\n";
    }
    wr << tensorI << "type = " << tflmc::to_string(t->type) << ";\n";
    wr << tensorI << "is_variable = " << t->is_variable << ";\n";
    wr << tensorI
       << "allocation_type = " << tflmc::to_string(t->allocation_type) << ";\n";
    wr << tensorI << "bytes = " << t->bytes << ";\n";
    wr << tensorI << "dims = (TfLiteIntArray*)&" << prefix_ << "tensor_dimension"
       << i << ";\n";
    if (t->quantization.type == kTfLiteAffineQuantization) {
      wr << tensorI << "params.scale = " << t->params.scale << ";\n";
      wr << tensorI << "params.zero_point = " << t->params.zero_point << ";\n";
      // TODO: Is this cast safe or does the data need to be non-const?
      wr << tensorI
         << "quantization = {kTfLiteAffineQuantization, "
            "const_cast<void*>(static_cast<const void*>(&"
         << prefix_ << "quant" << i << "))};\n";
    }
  }
  wr << "\n";
  for (size_t i = 0; i < registrations_.size(); i++) {
    auto opName = tflite::EnumNameBuiltinOperator(registrations_[i].code);
    wr << "  g_registrations[" << i << "] = tflite::ops::micro::Register_"
       << opName << "();\n";
  }
  wr << "\n";
  for (size_t i = 0; i < nodes_.size(); i++) {
    std::string nodeI = "  g_nodes[" + std::to_string(i) + "].";
    wr << nodeI << "inputs = (TfLiteIntArray*)&" << prefix_ << "inputs" << i
       << ";\n";
    wr << nodeI << "outputs = (TfLiteIntArray*)&" << prefix_ << "outputs" << i
       << ";\n";
    wr << nodeI << "temporaries = nullptr;\n";
    // TODO: Is this cast safe or does the data need to be non-const?
    if (nodes_[i].node.builtin_data) {
      wr << nodeI
         << "builtin_data = const_cast<void*>(static_cast<const void*>(&"
         << prefix_ << "opdata" << i << "));\n";
    } else {
      wr << nodeI << "builtin_data = nullptr;\n";
    }
    wr << nodeI << "custom_initial_data = nullptr;\n";
    wr << nodeI << "custom_initial_data_size = 0;\n";
    wr << nodeI << "delegate = nullptr;\n";
  }
  wr << "\n";
  for (size_t i = 0; i < nodes_.size(); i++) {
    auto &node = nodes_[i].node;
    auto regI = nodes_[i].regIndex;
    if (registrations_[regI].reg->init) {
      std::string nodeStr = "g_nodes[" + std::to_string(i) + "]";
      std::string ptrArg = node.builtin_data
                               ? "(const char *)" + nodeStr + ".builtin_data"
                               : "nullptr";

      // Length arg should be zero according to doc.
      wr << "  " << nodeStr << ".user_data = g_registrations[" << regI
         << "]->init(&g_ctx, " << ptrArg << ", 0);\n";
    }
  }
  wr << "\n  TfLiteStatus status = kTfLiteOk;\n";
  for (size_t i = 0; i < nodes_.size(); i++) {
    auto regI = nodes_[i].regIndex;
    if (registrations_[regI].reg->prepare) {
      wr << "  status = g_registrations[" << regI
         << "]->prepare(&g_ctx, &g_nodes[" << i << "]);\n";
      wr << "  assert(status == kTfLiteOk && \"Prepare failed\");\n";
    }
  }
  wr << R"(}

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
TfLiteTensor* )" << prefix_ << R"(input(int index) {
  return &g_ctx.tensors[inTensorIndices[index]];
}

static const int outTensorIndices[] = {
  )"; // TODO: perhaps use a smaller type than int?
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
TfLiteTensor* )" << prefix_ << R"(output(int index) {
  return &g_ctx.tensors[outTensorIndices[index]];
}

void )"
      << prefix_ << R"(invoke() {
  TfLiteStatus status = kTfLiteOk;
)";
  for (size_t i = 0; i < nodes_.size(); i++) {
    wr << "  status = g_registrations[" << nodes_[i].regIndex
       << "]->invoke(&g_ctx, &g_nodes[" << i << "]);\n";
    wr << "  assert(status == kTfLiteOk && \"Invoke failed\");\n";
  }
  wr << R"(}
)";
}
