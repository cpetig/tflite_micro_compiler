
#include "Compiler.h"

#include <cstdio>
#include <fstream>
#include <memory>
#include <regex>
#include <vector>

#include "CodeWriter.h"
#include "CustomOperators.h"
#include "RecordAllocations.h"
#include "TypeToString.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/micro/micro_allocator.h"
#include "tensorflow/lite/micro/micro_graph.h"
#include "tensorflow/lite/micro/micro_interpreter.h"

#if TF_LITE_STATIC_KERNEL_VARIANTS_VERSION
#if TF_LITE_STATIC_KERNEL_VARIANTS_VERSION != 100
#error "ONLY TF_LITE_PACKED_QUANTIZED_DATA_VERSION Vwersion 100 supported!"
#endif
#include "tensorflow/lite/micro/kernels/static_init_support.h"
#endif

#ifndef SUFFICIENT_ARENA_SIZE
#define SUFFICIENT_ARENA_SIZE (128 * 1024 * 1024)
#endif

#ifndef SUFFICIENT_ARENA_ALIGNMENT
#define SUFFICIENT_ARENA_ALIGNMENT (16)
#endif

#if TF_LITE_PACKED_QUANTIZED_DATA_VERSION
#if !(TF_LITE_PACKED_QUANTIZED_DATA_VERSION >= 100 && \
      TF_LITE_PACKED_QUANTIZED_DATA_VERSION <= 110)
#error \
    "ONLY TF_LITE_PACKED_QUANTIZED_DATA_VERSION version [100..110] supported "
#endif
#endif

static std::vector<int> flat_namespaced_ops(
    {tflite::BuiltinOperator_CONV_2D, tflite::BuiltinOperator_DEPTHWISE_CONV_2D,
     tflite::BuiltinOperator_FULLY_CONNECTED, tflite::BuiltinOperator_QUANTIZE,
     tflite::BuiltinOperator_SHAPE, tflite::BuiltinOperator_SOFTMAX,
     tflite::BuiltinOperator_SVDF});

int tflmc::Compiler::TrackingErrorReporter::Report(const char *format,
                                                   va_list args) {
  vfprintf(stderr, format, args);
  error_reported_ = true;
  return 0;
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
    return compiler.noErrorsReported();
  } catch (const std::exception &e) {
    std::cerr << e.what() << "\n";
  } catch (...) {
    std::cerr << "Unknown exception\n";
  }

  return false;
}

tflmc::Compiler::Compiler(const void *modelData, const std::string &prefix)
    : prefix_(prefix),
      arena_(SUFFICIENT_ARENA_SIZE, SUFFICIENT_ARENA_ALIGNMENT) {
  aligned_arena_start_ = arena_.alginedBufferStart();
  arena_size_ = SUFFICIENT_ARENA_SIZE;
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
  tflmc::custom_operator_handle custom =
      tflmc::LoadCustom(static_cast<tflite::MicroOpResolver *>(&resolver_));
  
  // create allocator to be passed to the interpreter, needed to access subgraph
  // allocations.
  static tflite::MicroAllocator *allocator_ = tflite::MicroAllocator::Create(
      aligned_arena_start_, arena_size_, &errReporter());
  if (allocator_ == nullptr) {
    std::cout << "allocator failed to create\n";
  }
  // Build an interpreter to run the model with.
  
#if 0
  interpreter_ = std::unique_ptr<tflite::MicroInterpreter>(
      new tflite::MicroInterpreter(
        model_, resolver_, aligned_arena_start_, arena_size_,
        &errReporter()));
#else
  interpreter_ =
      std::unique_ptr<tflite::MicroInterpreter>(new tflite::MicroInterpreter(
          model_, resolver_, allocator_, &errReporter()));
#endif
  
#if TFLMC_USE_INTERPRETER_HOOKS
  tflmc::SetRecordAllocationhooks(interpreter_.get(), aligned_arena_start_,
                                  arena_size_);
#endif
  // Allocate memory from the tensor_arena for the model's tensors.
  TfLiteStatus allocate_status = interpreter_->AllocateTensors();
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
      ptrdiff_t offset = (uint8_t *)tensor->data.data - aligned_arena_start_;
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


  tflite::MicroGraph *graph_ = &interpreter_->graph_;
  tflite::SubgraphAllocations *subgraph_allocations = graph_->GetAllocations();
  for (size_t i = 0; i < graph_->NumSubgraphs(); i++) {
    for (size_t j = 0; j < model_->subgraphs()->Get(i)->operators()->size();
         j++) {
      TfLiteNode *node =
          &(subgraph_allocations[i].node_and_registrations[j].node);
      const TfLiteRegistration *reg =
          subgraph_allocations[i].node_and_registrations[j].registration;

      auto code = tflite::EnumValuesBuiltinOperator()[reg->builtin_code];

      std::cout << "operation " << j << ": "
                << tflite::EnumNamesBuiltinOperator()[code] << std::endl;

      RegistrationInfo regInfo;
      regInfo.reg = reg;
      regInfo.code = code;
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
      nodes_.push_back(NodeInfo{*node, itOp - registrations_.begin()});
    }
  }

  
#if TFLMC_USE_INTERPRETER_HOOKS
  tflmc::RecordScratchBufferAllocations(interpreter_.get());
#else
  tflmc::RecordAllocations(model_, SUFFICIENT_ARENA_SIZE,
                           SUFFICIENT_ARENA_ALIGNMENT);
#endif
  auto runtimeAllocations = tflmc::RecordedAllocations();
  for (const auto &alloc : runtimeAllocations) {
    switch (alloc.kind) {
      case tflmc::AllocKind::Persistent:
        memMap_.recordRAM(alloc.offset, alloc.len,
                          "PersistentBuf_" + std::to_string(alloc.nodeIndex));
        break;
      case tflmc::AllocKind::Scratch:
        memMap_.recordRAMScratchBuf(alloc.buffer_index, alloc.offset, alloc.len,
                                    "ScratchBuf_" +
                                        std::to_string(alloc.nodeIndex) + "_" +
                                        std::to_string(alloc.buffer_index));
        break;
      default:
        assert(false && "Urecognized allocation kind");
    }
  }

  // This includes:
  // - Tensors
  // - Scratch buffers
  // - Persistent buffers
  // tensor metadata is not included, since we declare them outside the arena
  memMap_.stripLargestRAMGap(SUFFICIENT_ARENA_ALIGNMENT);
  arenaBufferSize_ = memMap_.requiredBufferSize();

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

void tflmc::Compiler::writeCustomRegistrationsSource(CodeWriter &wr) {
  // declare custom registrations
  if (has_custom_ops) {
    wr << R"(namespace tflite {
namespace micro {
)";
    for (size_t i = 0; i < registrations_.size(); i++) {
      if (registrations_[i].code == tflite::BuiltinOperator_CUSTOM) {
        wr << "extern TfLiteRegistration *Register_"
           << registrations_[i].custom_name << "(void);\n";
      }
    }
    wr << R"(}  // namespace micro
}  // namespace tflite

)";
  }
}

void tflmc::Compiler::writeTypesAndWorkingArraysSource(CodeWriter &wr) {
  wr << R"(namespace {

constexpr int kTensorArenaSize = )"
     << arenaBufferSize_ << R"(;
uint8_t tensor_arena[kTensorArenaSize] ALIGN(16);

template <int SZ, class T> struct TfArray {
  int sz; T elem[SZ];
};
)";

  wr << R"(

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
)";

  wr << R"(
  
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
)";
}

void tflmc::Compiler::writeTflNodesSource(CodeWriter &wr) {
  wr << R"(
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
}

void tflmc::Compiler::writeTensorDataSource(CodeWriter &wr) {
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
         << ((uintptr_t)t->data.data - (uintptr_t)aligned_arena_start_);
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
}

void tflmc::Compiler::writeNodeDataSource(CodeWriter &wr) {
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
}

void tflmc::Compiler::writeScratchBufferOffsets(CodeWriter &wr) {
  auto &&scratchbuf_offsets = memMap_.scratchBufOffsets();
  wr << R"(
static size_t scratchbuf_offsets[] = {
  )";

  size_t offsets = 0;
  for (auto o : scratchbuf_offsets) {
    wr << std::to_string(o) << ",";
    ++offsets;
    if (offsets % 10 == 0) {
      wr << "\n";
    } else {
      wr << " ";
    }
  }

  // To suppress warnings add dummy element if no scratch bufs
  if (scratchbuf_offsets.empty()) {
    wr << "0";
  }
  wr << R"(
};  
  )";
}

void tflmc::Compiler::writeContextAllocationHandlersSource(CodeWriter &wr) {
  // TODO: This code assumes that:
  // * persistent allocations are made from the end
  // * scratch buffer indexes are allocated (couting up from 0
  // Both are true for the current (02.09.2020) implementation
  // it wuold be good to add some sanity checking for debug builds here to ease
  // maintenance in the face of upstream changes to tflite(u)
  wr << R"(
static void *AllocatePersistentBuffer(struct TfLiteContext* ignore,
                                                 size_t bytes) {
  static uint8_t *AllocPtr = tensor_arena + sizeof(tensor_arena);

  AllocPtr -= bytes;
  return AllocPtr;
}

static TfLiteEvalTensor *GetEvalTensor(const struct TfLiteContext *ignore,
                                       int tensor_idx) {
  return &evalTensors[tensor_idx];
}


static TfLiteStatus RequestScratchBufferInArena(TfLiteContext *ignored,
                                                size_t bytes_ignored,
                                                int *buffer_idx) {
  static int idx_ctr = 0;
  *buffer_idx = idx_ctr;
  ++idx_ctr;
  return kTfLiteOk;
}

static void* GetScratchBuffer(struct TfLiteContext *ignore, int buffer_idx) {
  return tensor_arena + scratchbuf_offsets[buffer_idx];
}
)";
}

void tflmc::Compiler::writeInitSource(CodeWriter &wr) {
  wr << R"(TfLiteStatus )" << prefix_ << R"(init() {
  ctx.AllocatePersistentBuffer = &AllocatePersistentBuffer;
  ctx.RequestScratchBufferInArena = RequestScratchBufferInArena;
  ctx.GetScratchBuffer = &GetScratchBuffer;
  ctx.GetEvalTensor = &GetEvalTensor;
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
    auto code = registrations_[i].code;
    if (code == tflite::BuiltinOperator_CUSTOM) {
      opName = registrations_[i].custom_name;
    } else {
      opName = tflite::EnumNameBuiltinOperator(code);
    }
    const char *op_register_fn_namspaces;
    if (std::find(flat_namespaced_ops.begin(), flat_namespaced_ops.end(),
                  code) != flat_namespaced_ops.end()) {
      op_register_fn_namspaces = "tflite::";
    } else {
      op_register_fn_namspaces = "tflite::ops::micro::";
    }
    wr << "  registrations[OP_" << opName << "] = " << op_register_fn_namspaces
       << "Register_" << opName << "();\n";
  }
  wr << "\n";
#if TF_LITE_STATIC_KERNEL_VARIANTS_VERSION
  wr << R"(
  tflite::micro::resetStaticDataCounters();
)";
#endif
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

#if TF_LITE_STATIC_KERNEL_VARIANTS_VERSION
  wr << R"(
  tflite::micro::resetStaticDataCounters();
)";
#endif
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
)";
}

void tflmc::Compiler::writeTensorAccessorsSource(CodeWriter &wr) {
  wr << R"(
  static const int inTensorIndices[] = {
    )";
  for (auto inIndex : inputTensorIndices_) {
    wr << inIndex << ", ";
  }
  wr << R"(
  };
  TfLiteTensor* )"
     << prefix_ << R"(input(int index) {
    return &ctx.tensors[inTensorIndices[index]];
  }

  static const int outTensorIndices[] = {
    )";  // TODO: perhaps use a smaller type than int?
  for (auto outIndex : outputTensorIndices_) {
    wr << outIndex << ", ";
  }
  wr << R"(
  };
  TfLiteTensor* )"
     << prefix_ << R"(output(int index) {
    return &ctx.tensors[outTensorIndices[index]];
  }
  )";
}

void tflmc::Compiler::writeInvokeSource(CodeWriter &wr) {
  wr << R"(

TfLiteStatus )"
     << prefix_ << R"(invoke() {
)";
#if TF_LITE_STATIC_KERNEL_VARIANTS_VERSION
  wr << R"(
  tflite::micro::resetStaticDataCounters();
)";
#endif
  wr << R"(
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

void tflmc::Compiler::writeSource(std::ostream &out) {
  CodeWriter wr(out, subgraph_, microErrReporter_);

  wr << R"(
#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/micro/kernels/micro_ops.h"

#if defined __GNUC__
#define ALIGN(X) __attribute__((aligned(X)))
#elif defined _MSC_VER
#define ALIGN(X) __declspec(align(X))
#elif defined __TASKING__
#define ALIGN(X) __align(X)
#endif

)";

#if TF_LITE_STATIC_KERNEL_VARIANTS_VERSION
  tflite::micro::writeStaticOpDataHeaders(out);
#endif

  writeCustomRegistrationsSource(wr);

  writeTypesAndWorkingArraysSource(wr);

  writeTflNodesSource(wr);

  writeTensorDataSource(wr);

  writeNodeDataSource(wr);

  writeScratchBufferOffsets(wr);

  writeContextAllocationHandlersSource(wr);

  // TODO:  Really need to support AllocateBufferForEval.  Should be easy - just
  // need to permit allocating a suitable "gap" in the arena or a dedicated
  // scratchpad area.

  wr << R"(
} // namespace
)";

#if TF_LITE_STATIC_KERNEL_VARIANTS_VERSION
  tflite::micro::writeCppFunctionsToInvokeRecorded(out);
  tflite::micro::writeStaticOpDataDefinitions(out);
#endif

  writeInitSource(wr);

  writeTensorAccessorsSource(wr);

  writeInvokeSource(wr);
}

void tflmc::Compiler::writeHeader(std::ostream &out) {
  tflmc::CodeWriter wr(out, subgraph_, errReporter());

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

  tflite::MicroGraph *graph_ = &interpreter_->graph_;
  tflite::SubgraphAllocations *subgraph_allocations = graph_->GetAllocations();
  for (size_t i = 0; i < graph_->NumSubgraphs(); i++) {
    for (size_t j = 0; j < model_->subgraphs()->Get(i)->operators()->size();
         j++) {
      TfLiteNode *node =
          &(subgraph_allocations[i].node_and_registrations[j].node);
      // const TfLiteRegistration *reg =
      // subgraph_allocations[i].node_and_registrations[j].registration;
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
    }
  }

  return ss.str();
}

bool tflmc::Compiler::noErrorsReported() const {
  return !microErrReporter_.getErrorReported();
}