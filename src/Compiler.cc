#include "Compiler.h"

#include <fstream>
#include <memory>
#include <regex>
#include <sstream>
#include <vector>

#include "CodeWriter.h"
#include "CustomOperators.h"
#include "TypeToString.h"

#ifndef SUFFICIENT_ARENA_SIZE
#define SUFFICIENT_ARENA_SIZE (128 * 1024 * 1024)
#endif

#if TF_LITE_PACKED_QUANTIZED_DATA_VERSION
#if TF_LITE_PACKED_QUANTIZED_DATA_VERSION != 100
#error "ONLY TF_LITE_PACKED_QUANTIZED_DATA_VERSION Version 100 supported!"
#endif
#endif

static std::vector<tflmc::Allocation> g_loggedAllocations;
static int g_currentNodeIndex = -1;
static uint8_t *g_arenaPtr = nullptr;
static ptrdiff_t g_arena_size = 0;

static void *LoggingAllocatePersistentBuffer(struct TfLiteContext *ctx,
                                             size_t bytes) {
  void *ptr = tflite::GetMicroContext(ctx)->AllocatePersistentBuffer(bytes);
  assert(ptr != nullptr && "Alloc failure");
  g_loggedAllocations.push_back({-(g_arenaPtr - (uint8_t *)ptr + g_arena_size),
                                 bytes, g_currentNodeIndex});
  return ptr;
}

TfLiteStatus tflmc::AllocateTensors(
    std::unique_ptr<tflite::MicroInterpreter> &interpreter) {
  tflite::SubgraphAllocations *allocations =
      interpreter->allocator_.StartModelAllocation(interpreter->model_);

  if (allocations == nullptr) {
    TF_LITE_REPORT_ERROR(interpreter->error_reporter_,
                         "Failed starting model allocation.\n");
    interpreter->initialization_status_ = kTfLiteError;
    return kTfLiteError;
  }

  interpreter->graph_.SetSubgraphAllocations(allocations);

  TF_LITE_ENSURE_STATUS(
      interpreter->PrepareNodeAndRegistrationDataFromFlatbuffer());

  // Only allow AllocatePersistentBuffer in Init stage.
  interpreter->context_.AllocatePersistentBuffer =
      &LoggingAllocatePersistentBuffer;
  interpreter->context_.RequestScratchBufferInArena = nullptr;
  interpreter->context_.GetScratchBuffer = nullptr;
  interpreter->context_.GetExternalContext = nullptr;
  TF_LITE_ENSURE_STATUS(interpreter->graph_.InitSubgraphs());

  // Both AllocatePersistentBuffer and RequestScratchBufferInArena is
  // available in Prepare stage.
  interpreter->context_.RequestScratchBufferInArena =
      tflite::MicroContextRequestScratchBufferInArena;
  // external_context become available in Prepare stage.
  interpreter->context_.GetExternalContext =
      tflite::MicroContextGetExternalContext;

  TF_LITE_ENSURE_STATUS(interpreter->graph_.PrepareSubgraphs());

  // Prepare is done, we're ready for Invoke. Memory allocation is no longer
  // allowed. Kernels can only fetch scratch buffers via GetScratchBuffer.
  interpreter->context_.AllocatePersistentBuffer = nullptr;
  interpreter->context_.RequestScratchBufferInArena = nullptr;
  interpreter->context_.GetScratchBuffer = tflite::MicroContextGetScratchBuffer;

  TF_LITE_ENSURE_OK(
      &interpreter->context_,
      interpreter->allocator_.FinishModelAllocation(
          interpreter->model_, interpreter->graph_.GetAllocations(),
          &interpreter->scratch_buffer_handles_));

  interpreter->micro_context_.SetScratchBufferHandles(
      interpreter->scratch_buffer_handles_);

  // TODO(b/162311891): Drop these allocations when the interpreter supports
  // handling buffers from TfLiteEvalTensor.
  interpreter->input_tensors_ = reinterpret_cast<TfLiteTensor **>(
      interpreter->allocator_.AllocatePersistentBuffer(
          sizeof(TfLiteTensor *) * interpreter->inputs_size()));
  if (interpreter->input_tensors_ == nullptr) {
    TF_LITE_REPORT_ERROR(
        interpreter->error_reporter_,
        "Failed to allocate memory for context->input_tensors_, "
        "%d bytes required",
        sizeof(TfLiteTensor *) * interpreter->inputs_size());
    return kTfLiteError;
  }

  for (size_t i = 0; i < interpreter->inputs_size(); ++i) {
    interpreter->input_tensors_[i] =
        interpreter->allocator_.AllocatePersistentTfLiteTensor(
            interpreter->model_, interpreter->graph_.GetAllocations(),
            interpreter->inputs().Get(i), 0);
    if (interpreter->input_tensors_[i] == nullptr) {
      TF_LITE_REPORT_ERROR(interpreter->error_reporter_,
                           "Failed to initialize input tensor %d", i);
      return kTfLiteError;
    }
  }

  // TODO(b/162311891): Drop these allocations when the interpreter supports
  // handling buffers from TfLiteEvalTensor.
  interpreter->output_tensors_ = reinterpret_cast<TfLiteTensor **>(
      interpreter->allocator_.AllocatePersistentBuffer(
          sizeof(TfLiteTensor *) * interpreter->outputs_size()));
  if (interpreter->output_tensors_ == nullptr) {
    TF_LITE_REPORT_ERROR(
        interpreter->error_reporter_,
        "Failed to allocate memory for context->output_tensors_, "
        "%d bytes required",
        sizeof(TfLiteTensor *) * interpreter->outputs_size());
    return kTfLiteError;
  }

  for (size_t i = 0; i < interpreter->outputs_size(); ++i) {
    interpreter->output_tensors_[i] =
        interpreter->allocator_.AllocatePersistentTfLiteTensor(
            interpreter->model_, interpreter->graph_.GetAllocations(),
            interpreter->outputs().Get(i), 0);
    if (interpreter->output_tensors_[i] == nullptr) {
      TF_LITE_REPORT_ERROR(interpreter->error_reporter_,
                           "Failed to initialize output tensor %d", i);
      return kTfLiteError;
    }
  }

  TF_LITE_ENSURE_STATUS(interpreter->ResetVariableTensors());

  interpreter->tensors_allocated_ = true;
  return kTfLiteOk;
}

TfLiteTensor *tflmc::GetTensor(tflite::MicroInterpreter *interpreter, int i) {
  auto ctx = &interpreter->context_;
  return ctx->GetTensor(ctx, i);
}

bool tflmc::CompileFile(const std::string &modelFileName,
                        const std::string &outFileName,
                        const std::string &prefix) {
  // Load model flatbuffer.
  std::ifstream model_file(modelFileName, std::ios::binary | std::ios::ate);
  if (!model_file) {
    std::cerr << "Could not open " << modelFileName << " for read\n";
    return false;
  }
  auto sz = model_file.tellg();
  if (sz == std::ifstream::pos_type(-1)) {
    std::cerr << "Failed to read model file size\n";
    return false;
  }
  std::vector<char> model_data(sz);
  model_file.seekg(0, std::ios::beg);
  if (!model_file.read(model_data.data(), sz)) {
    std::cerr << "Failed to read model file\n";
    return false;
  }

  std::ofstream outFile(outFileName);
  if (!outFile) {
    std::cerr << "Failed to create output file\n";
    return false;
  }

  std::ofstream outHeaderFile(outFileName + ".h");
  if (!outHeaderFile) {
    std::cerr << "Failed to create output header file\n";
    return false;
  }

  try {
    Compiler compiler(model_data.data(), prefix);
    compiler.writeSource(outFile);
    compiler.writeHeader(outHeaderFile);
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

  g_arena_size = SUFFICIENT_ARENA_SIZE;
  std::vector<uint8_t> arena_buf(g_arena_size);
  g_arenaPtr = arena_buf_.data();

  interpreter_ = std::unique_ptr<tflite::MicroInterpreter>(
      new tflite::MicroInterpreter(model_, resolver_, arena_buf_.data(),
                                   arena_buf_.size(), &microErrReporter_));

  assert(interpreter_->graph_.NumSubgraphs() == 1);

  // Allocate memory from the tensor_arena for the model's tensors.
  TfLiteStatus allocate_status = AllocateTensors(interpreter_);
  if (allocate_status != kTfLiteOk) {
    errReporter().Report("AllocateTensors() failed");
    return false;
  }

  ptrdiff_t ramTensorBufferSize = 0;
  ptrdiff_t romOffset = 0;
  auto numTensors = tensors->size();
  if (numTensors > 0) {
    auto tensor = GetTensor(interpreter_.get(), 0);
    common_tensor_type = tensor->type;
    common_tensor_is_variable = tensor->is_variable;
  }
  for (size_t i = 0; i < numTensors; i++) {
    auto tensor = GetTensor(interpreter_.get(), i);
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
    // determine whether we need to individually set these properties for each
    // tensor
    if ((!has_quantization) &&
        tensor->quantization.type != kTfLiteNoQuantization) {
      has_quantization = true;
    }
    if ((!common_tensor_type.None) && common_tensor_type.Some != tensor->type) {
      common_tensor_type.clear();
    }
    if ((!common_tensor_is_variable.None) &&
        common_tensor_is_variable.Some != tensor->is_variable) {
      common_tensor_is_variable.clear();
    }
  }

  for (size_t k = 0; k < interpreter_->allocator_.scratch_buffer_request_count_;
       k++) {
    void *data = interpreter_->micro_context_.GetScratchBuffer(k);
    ptrdiff_t offset = (uint8_t *)data - arena_buf_.data();
    tflite::internal::ScratchBufferRequest *requests =
        interpreter_->allocator_.GetScratchBufferRequests();
    int bytes = requests[k].bytes;
    ptrdiff_t highSize = offset + bytes;
    ramTensorBufferSize = std::max(ramTensorBufferSize, highSize);
    memMap_.recordRAM(offset, bytes,
                      "Scratch_idx" + std::to_string((int)k) + "_op" +
                          std::to_string((int)requests[k].node_idx));
    scratchBufferOffsets.push_back(offset);
  }

  for (size_t i = 0; i < interpreter_->operators_size(); i++) {
    auto nodeAndReg = interpreter_->node_and_registration(i);
    auto node = &nodeAndReg.node;
    auto reg = nodeAndReg.registration;
    auto code = tflite::EnumValuesBuiltinOperator()[reg->builtin_code];

    printf("operation %lu: %s\n", i, tflite::EnumNamesBuiltinOperator()[code]);

    RegistrationInfo regInfo;
    regInfo.reg = reg;
    regInfo.code = code;
    if (code == tflite::BuiltinOperator_CUSTOM) {
      regInfo.custom_name = reg->custom_name;
      if (regInfo.custom_name == "TFLite_Detection_PostProcess") {
        has_tflite_custom_ops = true;
      }
      has_custom_ops = true;
    }
    auto itOp =
        std::find(registrations_.begin(), registrations_.end(), regInfo);
    if (itOp == registrations_.end()) {
      itOp = registrations_.insert(registrations_.end(), regInfo);
    }

    // There doesn't seem to be a way to get the node pointer, so copy it.
    nodes_.push_back(NodeInfo{*node, itOp - registrations_.begin()});
  }

  ptrdiff_t minRuntimeOffset = 0;  // These are negative so zero start is fine.
  for (const auto &alloc : g_loggedAllocations) {
    minRuntimeOffset = std::min(minRuntimeOffset, alloc.offset);
  }
  size_t totalRuntimeAllocSize = 0;
  for (const auto &alloc : g_loggedAllocations) {
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

#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/micro/kernels/conv.h"
#include "tensorflow/lite/micro/kernels/fully_connected.h"
#include "tensorflow/lite/micro/kernels/micro_ops.h"
#include "tensorflow/lite/micro/kernels/reduce.h"
#include "tensorflow/lite/micro/kernels/softmax.h"
#include "tensorflow/lite/micro/micro_context.h"

#if defined __GNUC__
#define ALIGN(X) __attribute__((aligned(X)))
#elif defined _MSC_VER
#define ALIGN(X) __declspec(align(X))
#elif defined __TASKING__
#define ALIGN(X) __align(X)
#endif

)";
  // declare custom registrations
  if (has_custom_ops) {
    wr << R"(namespace tflite {
namespace ops {
namespace micro {
)";
    for (size_t i = 0; i < registrations_.size(); i++) {
      if (registrations_[i].code == tflite::BuiltinOperator_CUSTOM &&
          registrations_[i].custom_name != "TFLite_Detection_PostProcess") {
        wr << "extern TfLiteRegistration *Register_"
           << registrations_[i].custom_name << "(void);\n";
      }
    }
    wr << R"(}  // namespace micro
}  // namespace ops
}  // namespace tflite

)";
  }
  if (has_tflite_custom_ops) {
    wr << R"(namespace tflite {
)";
    for (size_t i = 0; i < registrations_.size(); i++) {
      if (registrations_[i].code == tflite::BuiltinOperator_CUSTOM &&
          registrations_[i].custom_name == "TFLite_Detection_PostProcess") {
        wr << "extern TfLiteRegistration "
              "*Register_DETECTION_POSTPROCESS(void);\n";
      }
    }
    wr << R"(} // namespace tflite

)";
  }
  wr << R"(namespace {

constexpr int kTensorArenaSize = )"
     << arenaBufferSize_ << R"(;
uint8_t tensor_arena[kTensorArenaSize] ALIGN(16);
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
)";
  if (common_tensor_type.None) {
    wr << "  TfLiteType type;\n";
  }
  wr << R"(  void* data;
  TfLiteIntArray* dims;
  size_t bytes;
)";
  if (has_quantization) {
    wr << "  TfLiteQuantization quantization;\n";
  }
  if (common_tensor_is_variable.None) {
    wr << "  bool is_variable;\n";
  }
  wr << R"(};
struct NodeInfo_t { // subset of TfLiteNode used for initialization from constant memory
  struct TfLiteIntArray* inputs;
  struct TfLiteIntArray* outputs;
  void* builtin_data;
  used_operators_e used_op_index;
)";
  if (has_custom_ops) {
    wr << "  int custom_initial_data_size;\n";
  }
  wr << R"(};

TfLiteContext ctx{};
TfLiteTensor tflTensors[)"
     << tensors_.size() << R"(];
TfLiteEvalTensor evalTensors[)"
     << tensors_.size() << R"(];
TfLiteRegistration registrations[OP_LAST];
TfLiteNode tflNodes[)"
     << nodes_.size() << R"(];

)";
  for (size_t i = 0; i < tensors_.size(); i++) {
    auto &t = tensors_[i].tensor;
    if (t->allocation_type == kTfLiteMmapRo) {
      wr.writeTensor(*t, "tensor_data" + std::to_string(i));
    }
    wr.writeIntArray(*t->dims, "tensor_dimension" + std::to_string(i));
    wr.writeQuantization(t->quantization, "quant" + std::to_string(i));
#if TF_LITE_PACKED_QUANTIZED_DATA_VERSION
    wr.writeQuantizationDetails(t->quantization,
                                "quant_details" + std::to_string(i));
#endif
  }
  for (size_t i = 0; i < nodes_.size(); i++) {
    auto &node = nodes_[i].node;
    auto &regInfo = registrations_[nodes_[i].regIndex];
    if (regInfo.code == tflite::BuiltinOperator_CUSTOM) {
      wr << "uint8_t ALIGN(4) opdata" + std::to_string(i) << "["
         << node.custom_initial_data_size << "] = { ";
      for (int j = 0; j < node.custom_initial_data_size; ++j)
        wr << int(((uint8_t const *)node.custom_initial_data)[j]) << ", ";
      wr << " }; /* custom_initial_data */\n";
    } else {
      wr.writeBuiltin(regInfo.code, node.builtin_data,
                      "opdata" + std::to_string(i));
    }
    wr.writeIntArray(*node.inputs, "inputs" + std::to_string(i));
    wr.writeIntArray(*node.outputs, "outputs" + std::to_string(i));
  }
  wr << R"(const TensorInfo_t tensorData[] = {
)";
  for (size_t i = 0; i < tensors_.size(); i++) {
    auto &t = tensors_[i].tensor;
    wr << "  { ";
    if (common_tensor_type.None) {
      wr << tflmc::to_string(t->type) << ", ";
    }
    if (t->allocation_type == kTfLiteMmapRo) {
      wr << "(void*)tensor_data" << i;
    } else {
      wr << "tensor_arena + "
         << ((uintptr_t)t->data.data - (uintptr_t)arena_buf_.data());
    }
    wr << ", "
       << "(TfLiteIntArray*)&tensor_dimension" << i << ", ";
    wr << t->bytes << ", ";
    if (has_quantization) {
      if (t->quantization.type == kTfLiteAffineQuantization) {
        wr << "{kTfLiteAffineQuantization, "
              "const_cast<void*>(static_cast<const void*>(&quant"
           << i << ")) ";
      } else {
        wr << "{kTfLiteNoQuantization, nullptr ";
      }

#if TF_LITE_PACKED_QUANTIZED_DATA_VERSION
      if (t->quantization.details.type == kTfLiteSub8BitPackedUniformDetail) {
        wr << ", {kTfLiteSub8BitPackedUniformDetail, "
              "{&quant_details"
           << i << "}}";
      } else {
        wr << ", {kTfLiteNoDetails, {}}";
      }
#endif
      wr << "},";
    }
    if (common_tensor_is_variable.None) {
      wr << t->is_variable << ", ";
    }
    wr << "},\n";
  }
  wr << "};\n";
  wr << R"(const NodeInfo_t nodeData[] = {
)";
  for (size_t i = 0; i < nodes_.size(); i++) {
    wr << "  { (TfLiteIntArray*)&inputs" << i << ", ";
    wr << "(TfLiteIntArray*)&outputs" << i << ", ";
    // TODO: Is this cast safe or does the data need to be non-const?
    // CP: I think so (as it typically just carries the trained operator
    // parameters) CP: Also if it were written to, we would see a segfault
    // (write to text segment)
    if (nodes_[i].node.builtin_data || nodes_[i].node.custom_initial_data) {
      wr << "const_cast<void*>(static_cast<const void*>(&opdata" << i << ")), ";
    } else {
      wr << "nullptr, ";
    }
    auto regI = nodes_[i].regIndex;
    if (registrations_[regI].code == tflite::BuiltinOperator_CUSTOM) {
      wr << "OP_" << registrations_[regI].custom_name << ", "
         << nodes_[i].node.custom_initial_data_size << ", ";
    } else {
      wr << "OP_" << tflite::EnumNameBuiltinOperator(registrations_[regI].code)
         << ", ";
      if (has_custom_ops) {
        wr << "0, ";
      }
    }
    wr << "},\n";
  }
  wr << "};";
  // TODO: This code assumes that persistent allocations are made from the end
  // (which is true for the current implementation)
  wr << R"(

// Scratch buffer variables
int scratch_buffer_idx = 0;
const int scratch_buffer_offsets[)"
     << scratchBufferOffsets.size() << R"(] = { )";
  if (scratchBufferOffsets.size() > 0) {
    wr << scratchBufferOffsets[0];
    for (int i = 1; i < scratchBufferOffsets.size(); i++) {
      wr << ", " << scratchBufferOffsets[i];
    }
  }
  wr << R"( };
tflite::MicroContext mc;

// Functions to be used as function pointers for TfLiteContext and MicroContext
static void* AllocatePersistentBuffer(struct TfLiteContext* ctx,
                                                 size_t bytes) {
  static uint8_t *AllocPtr = tensor_arena + sizeof(tensor_arena);

  AllocPtr -= bytes;
  return AllocPtr;
}

static TfLiteEvalTensor *GetEvalTensor(const struct TfLiteContext *context,
                                       int tensor_idx) {
  return &evalTensors[tensor_idx];
}

static TfLiteStatus RequestScratchBufferInArena(struct TfLiteContext *context, size_t bytes,
                                       int *buffer_idx) {
  *buffer_idx = scratch_buffer_idx++;
  return kTfLiteOk;
};

static void *GetScratchBuffer(struct TfLiteContext *context,
                                       int buffer_idx) {
  return tensor_arena + scratch_buffer_offsets[buffer_idx];
}

static TfLiteTensor* AllocateTempInputTensor(const TfLiteNode* node, int index) {
      return &ctx.tensors[node->inputs->data[index]];
}

static TfLiteTensor* AllocateTempOutputTensor(const TfLiteNode* node, int index) {
      return &ctx.tensors[node->outputs->data[index]];
}

static void DeallocateTempTfLiteTensor(TfLiteTensor* tensor) {
}

static void* external_context() {
  return nullptr;
}

} // namespace

TfLiteStatus )"
     << prefix_ << R"(init() {
  // Setup microcontext functions
  mc.AllocateTempInputTensor = &AllocateTempInputTensor;
  mc.AllocateTempOutputTensor = &AllocateTempOutputTensor;
  mc.DeallocateTempTfLiteTensor = &DeallocateTempTfLiteTensor;
  mc.external_context = &external_context;

  // Setup tflitecontext functions
  ctx.AllocatePersistentBuffer = &AllocatePersistentBuffer;
  ctx.GetEvalTensor = &GetEvalTensor;
  ctx.RequestScratchBufferInArena = &RequestScratchBufferInArena;
  ctx.GetScratchBuffer = &GetScratchBuffer;

  // Set microcontext as the context ptr
  ctx.impl_ = (void*)&mc;
  ctx.tensors = tflTensors;
)";
  wr << "  ctx.tensors_size = " << tensors_.size() << ";\n";
  // TODO: Do we really support variable tensors?
  // TODO: Do we encounter other than kTfLiteMmapRo and kTfLiteArenaRw, if so we
  // need to store the type separately.
  wr << "  for(size_t i = 0; i < " << tensors_.size() << R"(; ++i) {
    tflTensors[i].data.data = tensorData[i].data;
    evalTensors[i].data.data = tensorData[i].data;
)";
  if (common_tensor_type.None) {
    wr << "    tflTensors[i].type = tensorData[i].type;\n";
    wr << "    evalTensors[i].type = tensorData[i].type;\n";
  } else {
    wr << "    tflTensors[i].type = "
       << tflmc::to_string(common_tensor_type.Some) << ";\n";
    wr << "    evalTensors[i].type = "
       << tflmc::to_string(common_tensor_type.Some) << ";\n";
  }
  if (common_tensor_is_variable.None) {
    wr << "    tflTensors[i].is_variable = tensorData[i].is_variable;\n";
  } else {
    wr << "    tflTensors[i].is_variable = "
       << std::to_string(common_tensor_is_variable.Some) << ";\n";
  }
  wr << R"(    tflTensors[i].allocation_type = (tensor_arena <= tensorData[i].data && tensorData[i].data < tensor_arena + kTensorArenaSize) ? kTfLiteArenaRw : kTfLiteMmapRo;
    tflTensors[i].bytes = tensorData[i].bytes;
    tflTensors[i].dims = tensorData[i].dims;
    evalTensors[i].dims = tensorData[i].dims;
)";
  if (has_quantization) {
    wr << R"(    tflTensors[i].quantization = tensorData[i].quantization;
    if (tflTensors[i].quantization.type == kTfLiteAffineQuantization) {
      TfLiteAffineQuantization const* quant = ((TfLiteAffineQuantization const*)(tensorData[i].quantization.params));
      tflTensors[i].params.scale = quant->scale->data[0];
      tflTensors[i].params.zero_point = quant->zero_point->data[0];
    }
)";
  } else {
    wr << "    tflTensors[i].quantization.type = kTfLiteNoQuantization;\n";
  }
  wr << R"(  }
)";
  for (size_t i = 0; i < registrations_.size(); i++) {
    std::string opName;
    if (registrations_[i].code == tflite::BuiltinOperator_CUSTOM) {
      opName = registrations_[i].custom_name;
      if (opName == "TFLite_Detection_PostProcess") {
        wr << "  registrations[OP_" << opName
           << "] = *(tflite::Register_DETECTION_POSTPROCESS());\n";
      }
    } else if ((registrations_[i].code == tflite::BuiltinOperator_ADD) ||
               (registrations_[i].code ==
                tflite::BuiltinOperator_AVERAGE_POOL_2D) ||
               (registrations_[i].code == tflite::BuiltinOperator_CONV_2D) ||
               (registrations_[i].code ==
                tflite::BuiltinOperator_DEPTHWISE_CONV_2D) ||
               (registrations_[i].code == tflite::BuiltinOperator_DEQUANTIZE) ||
               (registrations_[i].code ==
                tflite::BuiltinOperator_FULLY_CONNECTED) ||
               (registrations_[i].code == tflite::BuiltinOperator_LOGISTIC) ||
               (registrations_[i].code ==
                tflite::BuiltinOperator_MAX_POOL_2D) ||
               (registrations_[i].code == tflite::BuiltinOperator_MEAN) ||
               (registrations_[i].code == tflite::BuiltinOperator_MUL) ||
               (registrations_[i].code == tflite::BuiltinOperator_PRELU) ||
               (registrations_[i].code == tflite::BuiltinOperator_QUANTIZE) ||
               (registrations_[i].code == tflite::BuiltinOperator_RELU) ||
               (registrations_[i].code == tflite::BuiltinOperator_SHAPE) ||
               (registrations_[i].code == tflite::BuiltinOperator_SOFTMAX) ||
               (registrations_[i].code ==
                tflite::BuiltinOperator_TRANSPOSE_CONV)) {
      opName = tflite::EnumNameBuiltinOperator(registrations_[i].code);
      wr << "  registrations[OP_" << opName << "] = tflite::Register_" << opName
         << "();\n";
    } else {
      opName = tflite::EnumNameBuiltinOperator(registrations_[i].code);
      wr << "  registrations[OP_" << opName
         << "] = tflite::ops::micro::Register_" << opName << "();\n";
    }
  }
  wr << "\n";
  wr << "  for(size_t i = 0; i < " << nodes_.size() << R"(; ++i) {
    tflNodes[i].inputs = nodeData[i].inputs;
    tflNodes[i].outputs = nodeData[i].outputs;
    tflNodes[i].builtin_data = nodeData[i].builtin_data;
    tflNodes[i].custom_initial_data = nullptr;
    tflNodes[i].custom_initial_data_size = 0;
    if (registrations[nodeData[i].used_op_index].init) {
      tflNodes[i].user_data = registrations[nodeData[i].used_op_index].init(&ctx, (const char*)tflNodes[i].builtin_data, )";
  if (has_custom_ops) {
    wr << "nodeData[i].custom_initial_data_size";
  } else {
    wr << '0';
  }
  wr << R"();
    }
  }
)";
  wr << "  for(size_t i = 0; i < " << nodes_.size() << R"(; ++i) {
    if (registrations[nodeData[i].used_op_index].prepare) {
      TfLiteStatus status = registrations[nodeData[i].used_op_index].prepare(&ctx, &tflNodes[i]);
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
TfLiteTensor* )"
      << prefix_ << R"(input(int index) {
  return &ctx.tensors[inTensorIndices[index]];
}

static const int outTensorIndices[] = {
  )";  // TODO: perhaps use a smaller type than int?
  for (auto outIndex : outputTensorIndices_) {
    out << outIndex << ", ";
  }
  out << R"(
};
TfLiteTensor* )"
      << prefix_ << R"(output(int index) {
  return &ctx.tensors[outTensorIndices[index]];
}

TfLiteStatus )"
      << prefix_ << R"(invoke() {
  for(size_t i = 0; i < )"
      << nodes_.size() << R"(; ++i) {
    TfLiteStatus status = registrations[nodeData[i].used_op_index].invoke(&ctx, &tflNodes[i]);
    if (status != kTfLiteOk) {
      return status;
    }
  }
  return kTfLiteOk;
}
)";
}

void tflmc::Compiler::writeHeader(std::ostream &out) {
  tflmc::CodeWriter wr(out, subgraph_);

  std::string code = R"(
#ifndef %PREFIX%GEN_H
#define %PREFIX%GEN_H

#include "tensorflow/lite/c/common.h"

// Sets up the model with init and prepare steps.
TfLiteStatus %PREFIX%init();
// Returns the input tensor with the given index.
TfLiteTensor *%PREFIX%input(int index);
// Returns the output tensor with the given index.
TfLiteTensor *%PREFIX%output(int index);
// Runs inference for the model.
TfLiteStatus %PREFIX%invoke();

// Returns the number of input tensors.
inline size_t %PREFIX%inputs() {
  return )" + std::to_string(inputTensorIndices_.size()) +
                     R"(;
}
// Returns the number of output tensors.
inline size_t %PREFIX%outputs() {
  return )" + std::to_string(outputTensorIndices_.size()) +
                     R"(;
}

inline void *%PREFIX%input_ptr(int index) {
  return %PREFIX%input(index)->data.data;
}
inline size_t %PREFIX%input_size(int index) {
  return %PREFIX%input(index)->bytes;
}
inline int %PREFIX%input_dims_len(int index) {
  return %PREFIX%input(index)->dims->data[0];
}
inline int *%PREFIX%input_dims(int index) {
  return &%PREFIX%input(index)->dims->data[1];
}

inline void *%PREFIX%output_ptr(int index) {
  return %PREFIX%output(index)->data.data;
}
inline size_t %PREFIX%output_size(int index) {
  return %PREFIX%output(index)->bytes;
}
inline int %PREFIX%output_dims_len(int index) {
  return %PREFIX%output(index)->dims->data[0];
}
inline int *%PREFIX%output_dims(int index) {
  return &%PREFIX%output(index)->dims->data[1];
}

#endif
)";

  static std::regex rePrefix("%PREFIX%");
  code = std::regex_replace(code, rePrefix, prefix_);

  wr << code;
}

std::string tflmc::Compiler::getTensorName(int tensorIndex) const {
  auto tensor = GetTensor(interpreter_.get(), tensorIndex);

  std::stringstream ss;
  ss << (tensor->allocation_type == kTfLiteMmapRo ? "ROM" : "RAM") << "Tensor_";

  auto nOps = interpreter_->operators_size();
  for (size_t i = 0; i < nOps; i++) {
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
    //  checkAndAdd(node->intermediates, "int");
    //  checkAndAdd(node->temporaries, "tmp");
  }

  return ss.str();
}
