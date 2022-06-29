
#include "Compiler.h"
#include <memory>
#include <fstream>
#include <sstream>
#include <regex>
#include <vector>
#include <cstdio>

#include "CodeWriter.h"
#include "CustomOperators.h"
#include "RecordAllocations.h"
#include "Options.h"
#include "TypeToString.h"
#include "tensorflow/lite/c/common.h"


#if TF_LITE_MICRO_RECORD_OP_USER_DATA
#include "tflite_u_preint/static_init_support.h"
#endif

#ifndef SUFFICIENT_ARENA_SIZE 
#define SUFFICIENT_ARENA_SIZE (128*1024*1024)
#endif

#ifndef SUFFICIENT_ARENA_ALIGNMENT 
#define SUFFICIENT_ARENA_ALIGNMENT (16)
#endif

const static int ILLEGAL_IF_EVER_MULTIPLE_SUBGRAPH = 0xdeadbeef;



namespace tflmc
{

  /**
   * @brief Generation of specialized TensorInfo_t POD struct 
   * 
   */
  struct GeneratedTensorInfo {

    struct Full_t{
      TfLiteType type;
      void* data;
      TfLiteIntArray* dims;
      size_t bytes;
      TfLiteQuantization quantization;
      bool is_variable;
    };

    static std::string generated(bool has_type, bool has_quantization, bool has_is_variable) {

      std::stringstream wr;

      wr << R"(
struct TensorInfo_t { // subset of TfLiteTensor used for initialization from constant memory
)";
      if (has_type) {
        wr << "  TfLiteType type;\n";
      }
      wr << R"(  void* data;
  TfLiteIntArray* dims;
  size_t bytes;
)";
      if (has_quantization) {
        wr << "  TfLiteQuantization quantization;\n";
      }
      if (has_is_variable) {
        wr << "  bool is_variable;\n";
      }
      wr << "};\n";
      return wr.str();
    }

    struct TrailingBoolField {
      bool a_bool;
    };

    static size_t size(bool has_type, bool has_quantization, bool has_is_variable) {
      auto size = sizeof(Full_t);
      if (!has_type) { size -= sizeof(TfLiteType); }
      if (!has_quantization)  { size -= sizeof(TfLiteQuantization); }
      // Dangling bool... prboably more accurate than simply sizeof(bool)
      // once alignment / packing constraints are accounted for.
      if (!has_is_variable) { size -= sizeof(TrailingBoolField); }
      return size;
    }
  };

  /**
   * @brief Generation of specialized NodeInfo_t POD struct
   * 
   */

  struct GeneratedNodeInfo {

    enum used_operators_e { DUMMY_OP_INDEX, LAST_OP };

    struct Full_t { 
      struct TfLiteIntArray* inputs;
      struct TfLiteIntArray* outputs;
      void* builtin_data;
      used_operators_e used_op_index;
      int custom_initial_data_size;
    };


    static std::string generated(bool has_custom_ops) {

      std::stringstream wr;
      wr << R"(
struct NodeInfo_t { // subset of TfLiteNode used for initialization from constant memory
  struct TfLiteIntArray* inputs;
  struct TfLiteIntArray* outputs;
  void* builtin_data;
  used_operators_e used_op_index;
  )";
      if (has_custom_ops) {
        wr << "  int custom_initial_data_size;\n";
      }
      wr << "};\n";
      return wr.str();
    }

    static size_t size(bool has_custom_ops) {
      auto size = sizeof(Full_t);
      if (!has_custom_ops) size -= sizeof(int);
      return size;
    }
  };
} // namespace tflmc

static std::vector<int> flat_namespaced_ops({
    tflite::BuiltinOperator_ADD,
    tflite::BuiltinOperator_ADD_N,
    tflite::BuiltinOperator_ASSIGN_VARIABLE,
    tflite::BuiltinOperator_AVERAGE_POOL_2D,
    tflite::BuiltinOperator_BATCH_TO_SPACE_ND,
    tflite::BuiltinOperator_CALL_ONCE,
    tflite::BuiltinOperator_CAST,
    tflite::BuiltinOperator_CONV_2D,
    tflite::BuiltinOperator_CUMSUM,
    tflite::BuiltinOperator_DEPTH_TO_SPACE,
    tflite::BuiltinOperator_DEPTHWISE_CONV_2D,
    tflite::BuiltinOperator_DIV,
    tflite::BuiltinOperator_ELU,
    tflite::BuiltinOperator_EXP,
    tflite::BuiltinOperator_EXPAND_DIMS,
    tflite::BuiltinOperator_FILL,
    tflite::BuiltinOperator_FLOOR_DIV,
    tflite::BuiltinOperator_FLOOR_MOD,
    tflite::BuiltinOperator_FULLY_CONNECTED,
    tflite::BuiltinOperator_GATHER,
    tflite::BuiltinOperator_GATHER_ND,
    tflite::BuiltinOperator_HARD_SWISH,
    tflite::BuiltinOperator_IF,
    tflite::BuiltinOperator_L2_POOL_2D,
    tflite::BuiltinOperator_LEAKY_RELU,
    tflite::BuiltinOperator_LOG_SOFTMAX,
    tflite::BuiltinOperator_LOGICAL_AND,
    tflite::BuiltinOperator_LOGICAL_OR,
    tflite::BuiltinOperator_LOGISTIC,
    tflite::BuiltinOperator_MAX_POOL_2D,
    tflite::BuiltinOperator_MIRROR_PAD,
    tflite::BuiltinOperator_MUL,
    tflite::BuiltinOperator_PRELU,
    tflite::BuiltinOperator_QUANTIZE,
    tflite::BuiltinOperator_READ_VARIABLE,
    tflite::BuiltinOperator_RELU,
    tflite::BuiltinOperator_RELU6,
    tflite::BuiltinOperator_RESIZE_BILINEAR,
    tflite::BuiltinOperator_SHAPE,
    tflite::BuiltinOperator_SLICE,
    tflite::BuiltinOperator_SOFTMAX,
    tflite::BuiltinOperator_SPACE_TO_BATCH_ND,
    tflite::BuiltinOperator_SPACE_TO_DEPTH,
    tflite::BuiltinOperator_SQUEEZE,
    tflite::BuiltinOperator_SUB,
    tflite::BuiltinOperator_SVDF,
    tflite::BuiltinOperator_TRANSPOSE,
    tflite::BuiltinOperator_TRANSPOSE_CONV,
    tflite::BuiltinOperator_VAR_HANDLE,
    tflite::BuiltinOperator_ZEROS_LIKE
  })
;


static std::vector<int> graph_dependent_ops({

    tflite::BuiltinOperator_ASSIGN_VARIABLE,
    tflite::BuiltinOperator_CALL_ONCE,
    tflite::BuiltinOperator_IF,
    tflite::BuiltinOperator_VAR_HANDLE,
    tflite::BuiltinOperator_READ_VARIABLE,
  })
;

int tflmc::Compiler::TrackingErrorReporter::Report(const char* format, va_list args) {
  vfprintf(stderr, format, args);
  error_reported_ = true;
  return 0;
}



bool tflmc::CompileFile(const std::string &modelPathName,
                        const std::string &outSrcPathName,
                        const std::string &outHdrPathName,
                        const std::string &prefix) {
  // Load model flatbuffer.
  std::ifstream model_file(modelPathName, std::ios::binary | std::ios::ate);
  if (!model_file) {
    std::cerr << "Could not open " << modelPathName << " for read\n";
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


  std::ofstream outFile(outSrcPathName);
  if (!outFile) {
    std::cerr << "Failed to create output source file: " << outSrcPathName << std::endl;;
    return false;
  }

  std::ofstream outHeaderFile(outHdrPathName);
  if (!outHeaderFile) {
    std::cerr << "Failed to create output header file: " << outHdrPathName << std::endl;
    return false;
  }

  try {
    Compiler compiler(model_data.data(), prefix);
    
    compiler.writeSource(outFile);
    compiler.writeHeader(outHeaderFile);
    compiler.reportMemUsage();
    return compiler.noErrorsReported();
  } catch (const std::exception &e) {
    std::cerr << e.what() << "\n";
  } catch (...) {
    std::cerr << "Unknown exception\n";
  }

  return false;
}

tflmc::Compiler::Compiler(const void *modelData, const std::string &prefix)
    : prefix_(prefix)
    , arena_(SUFFICIENT_ARENA_SIZE, SUFFICIENT_ARENA_ALIGNMENT) {
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

  // Build an interpreter to run the model with.

  interpreter_ = std::unique_ptr<tflite::MicroInterpreter>(
      new tflite::MicroInterpreter(
        model_, resolver_, aligned_arena_start_, arena_size_,
        &errReporter()));

  // Now know model size etc so we can initialize (tables)
  // in tensor arena memory map.
  arenaMap_.init(interpreter_->operators_size());

#if TFLMC_USE_INTERPRETER_HOOKS
  // Activate hooks to record memory alliocations to fill _arenaMaop etc.
  tflmc::SetRecordAllocationhooks( interpreter_.get(), aligned_arena_start_, arena_size_);
#endif

  // Allocate memory from the tensor_arena for the model's tensors.
  TfLiteStatus allocate_status = interpreter_->AllocateTensors();
  if (allocate_status != kTfLiteOk) {
    errReporter().Report("AllocateTensors() failed");
    return false;
  }

  ptrdiff_t ramTensorBufferSize = 0;
  auto numTensors = tensors->size();
  if (numTensors > 0) {
    auto tensor = GetTensor(interpreter_.get(), 0);
    common_tensor_type = tensor->type;
  }
  for (size_t i = 0; i < numTensors; i++) {
    auto tensor = GetTensor(interpreter_.get(), i);
    tensors_.push_back({tensor});
    if (tensor->allocation_type != kTfLiteMmapRo) {
      ptrdiff_t offset = (uint8_t *)tensor->data.data - aligned_arena_start_;
      ptrdiff_t highSize = offset + tensor->bytes;
      ramTensorBufferSize = std::max(ramTensorBufferSize, highSize);
      arenaMap_.recordPersistent(offset, tensor->bytes, getTensorName(i));
    }

    // determine whether we need to individually set these properties for each
    // tensor
    has_quantization |= ( tensor->quantization.type != kTfLiteNoQuantization);
    if ((!common_tensor_type.None) && common_tensor_type.Some != tensor->type) {
      common_tensor_type.clear();
    }
    has_is_variable |= tensor->is_variable;
  }

  int unsupported_ops = 0;
  for (size_t i = 0; i < interpreter_->operators_size(); i++) {
    auto nodeAndReg = interpreter_->node_and_registration(ILLEGAL_IF_EVER_MULTIPLE_SUBGRAPH,i);
    auto node = &nodeAndReg.node;
    auto reg = nodeAndReg.registration;
    auto code = tflite::EnumValuesBuiltinOperator()[reg->builtin_code];

    std::cout << "operation " << i 
              << ": " << tflite::EnumNamesBuiltinOperator()[code];
              
    if (std::find(graph_dependent_ops.begin(), graph_dependent_ops.end(), code) != graph_dependent_ops.end()) {
      std::cout << " - requires  operator graph access(unsupported)" << std::endl;
      ++unsupported_ops;
    } else {
       std::cout << std::endl;
    }
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

  if (unsupported_ops > 0 ) {
    errReporter().Report("Model includes %d unsupported operators", unsupported_ops);
    return false;
  }

  for (size_t i = 0; i < registrations_.size(); i++) {
    std::string opName;
    auto code = registrations_[i].code;
    if (code == tflite::BuiltinOperator_CUSTOM) {
      opName = registrations_[i].custom_name;
    } else {
      opName = tflite::EnumNameBuiltinOperator(code);
    }

  }

#if TFLMC_USE_INTERPRETER_HOOKS
   tflmc::RecordScratchBufferAllocations(interpreter_.get()); 
#else
  tflmc::RecordAllocations(model_, SUFFICIENT_ARENA_SIZE, SUFFICIENT_ARENA_ALIGNMENT);
#endif
  auto runtimeAllocations = tflmc::RecordedAllocations();

  for (const auto &alloc : runtimeAllocations) {
    switch( alloc.kind ) {
      case tflmc::AllocKind::Persistent : 
        arenaMap_.recordPersistent(alloc.offset, alloc.len,
                      "PersistentBuf_" + std::to_string(alloc.nodeIndex));
        break;
      case tflmc::AllocKind::Scratch : 
        arenaMap_.recordScratchBuf(alloc.buffer_index, alloc.offset, alloc.len, alloc.nodeIndex,
                     "ScratchBuf_" + std::to_string(alloc.nodeIndex) + "_" +  std::to_string(alloc.buffer_index));
        break;
      default:
        assert(false && "Urecognized allocation kind");
    }


  }


  // At this point memMap only records the tensor arena.
  // - Tensors
  // - Scratch buffers
  // - Persistent buffers

  // Required arena size is end of ram memory usage  after we have
  // compacted it.  Currently merely by stripping the largest
  // gap (usual the gap between head/tail of arena)
  arenaMap_.stripLargestGap(SUFFICIENT_ARENA_ALIGNMENT);
  tflmc::UnloadCustom(custom);
  return true;
}

void tflmc::Compiler::finalizeMemMap(const CodeWriter &wr)
{
  size_t tensorMetaSize = tensors_.size() * (sizeof(TfLiteTensor)+sizeof(TfLiteEvalTensor));
  uninitMemMap_.record(tensorMetaSize, "TfliteTensorTables");

  auto TensorInfo_t_size = 
    tflmc::GeneratedTensorInfo::size(!common_tensor_type.None, has_quantization, has_is_variable);
  size_t tensorInfoSize = tensors_.size() * TensorInfo_t_size;
  constMemMap_.record(tensorInfoSize, "TensorInfo");

  initMemMap_.record(sizeof(TfLiteContext), "TfLiteContext");
  constMemMap_.record(sizeof(TfLiteContext), "TfLiteContext");

  auto NodeInfo_t_size = tflmc::GeneratedNodeInfo::size(has_custom_ops);
  size_t nodeMetaSize = nodes_.size() * NodeInfo_t_size;
  constMemMap_.record(nodeMetaSize, "NodeDataTable");

  size_t registrationsSize = registrations_.size() * sizeof(TfLiteRegistration);
  initMemMap_.record(registrationsSize, "OpRegistrations");

  constMemMap_.record(wr.constDataUsage(), "TensorAndOpdata");
  initMemMap_.record(wr.initDataUsage(), "TensorAndOpdata");
  uninitMemMap_.record(wr.uninitDataUsage(), "TensorAndOpdata");

#if TF_LITE_MICRO_RECORD_OP_USER_DATA
  constMemMap_.record(tflite::micro::constDataUsage(), "OpUserData");
  initMemMap_.record(tflite::micro::initDataUsage(), "OpUserData");
  uninitMemMap_.record(tflite::micro::uninitDataUsage(), "OpUserData");
#endif

}


void tflmc::Compiler::reportMemUsage()
{
 
  size_t romUsage = constMemMap_.size() + initMemMap_.size();
  std::fstream memmap_json;
  auto options = Options::instance();
  if (!options.memmap_json.empty()) {
    	memmap_json.open(options.memmap_json, std::fstream::out);
      if (!memmap_json) {
        std::cerr << "Could not open '" << options.memmap_json << "' for writing." << std::endl;
        exit(1);
      }

     memmap_json << "{" << std::endl;
  }
  if (memmap_json.is_open()) {
         memmap_json << "\"rodata\": " << constMemMap_.size() << "," << std::endl;         
         memmap_json << "\"data\": " << initMemMap_.size() << "," << std::endl;        
         memmap_json << "\"bss\": " << uninitMemMap_.size() << "," << std::endl;
  }
  std::cout << "ROM summary: "<< romUsage << " bytes total" << std::endl;
  if (memmap_json.is_open()) {
         memmap_json << "\"rom\": " << romUsage << "," << std::endl;
  }

  size_t ramUsage = uninitMemMap_.size() + initMemMap_.size();
  if (memmap_json.is_open()) {
    memmap_json << "\"ram\": " << ramUsage << std::endl;
    memmap_json << "}" << std::endl;
    memmap_json.close();
    if (!memmap_json) {
      std::cerr << "Could not write '" << options.memmap_json << "'." << std::endl;
      exit(1);
    }
  }

  constMemMap_.report("const data (.rodata)");
  initMemMap_.report("initalized data (.data)");
  uninitMemMap_.report("uninitalized data (.bss)");
  arenaMap_.report("Tensor Arena details");
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

)";

  wr.writeTensorArena(arenaMap_.size());
  wr << R"(

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

  wr <<
    tflmc::GeneratedTensorInfo::generated(common_tensor_type.None, has_quantization, has_is_variable);

  wr <<
    tflmc::GeneratedNodeInfo::generated(has_custom_ops);

  wr << R"(

TfLiteContext ctx{};

// Tensor table with space for -1-th element used
// designate missing optional inputs/outputs.
TfLiteTensor tflTensorsWithMinus1[)"
     << tensors_.size()+1u << R"(];
     
TfLiteEvalTensor evalTensors[)"
     << tensors_.size() << R"(];

TfLiteTensor * const tflTensors = tflTensorsWithMinus1+1;

TfLiteRegistration registrations[OP_LAST];
)";

}


void tflmc::Compiler::writeTflNodesSource(CodeWriter &wr) {
  
  wr << "constexpr size_t kOpNodesCount = " << nodes_.size() <<";\n\n";
  wr << R"(
TfLiteNode tflNodes[kOpNodesCount];

)";
  for (size_t i = 0; i < tensors_.size(); i++) {
    auto &t = tensors_[i].tensor;
    if (t->allocation_type == kTfLiteMmapRo) {
      wr.writeTensor(*t, "tensor_data" + std::to_string(i));
    }
    wr.writeIntArray(*t->dims, "tensor_dimension" + std::to_string(i));
    wr.writeQuantization(t->quantization, "quant" + std::to_string(i));
  }
  for (size_t i = 0; i < nodes_.size(); i++) {
    auto &node = nodes_[i].node;
    auto &regInfo = registrations_[nodes_[i].regIndex];
    if (regInfo.code == tflite::BuiltinOperator_CUSTOM) {
      wr.writeCustom((uint8_t const *)node.custom_initial_data, i, 
                     node.custom_initial_data_size);
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
#if SUPPORT_CUSTOM_QUANT
      } else if (t->quantization.type == kTfLitePackedAffineQuantization) {
          wr << "{kTfLitePackedAffineQuantization, "
                "const_cast<void*>(static_cast<const void*>(&quant"
             << i << ")) ";
#endif  // SUPPORT_CUSTOM_QUANT
      } else {
        wr << "{kTfLiteNoQuantization, nullptr ";
      }


      wr << "},";
    }
    if (has_is_variable) {
      wr << t->is_variable << ", ";
    }
    wr << "},\n";
  }
  wr << "};\n";
}

void tflmc::Compiler::writeNodeDataSource(CodeWriter &wr) {
  wr << R"(const NodeInfo_t nodeData[kOpNodesCount] = {
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
  wr << "};\n\n";

}


void tflmc::Compiler::writeScratchBufferOffsets(CodeWriter &wr) {

  // Complication: nodes with offline pre-computed user_data (OpData)
  // won't actually call RequestScratchBufferInArena
  // so we need to compute correct  next_scratch_buffer_idx for each node
  // from calls made during pre-interpretation
  wr << R"(
  // Used by RequestScratchBufferInArena to generate buffer index
  // for each request.  Reset for each node from _init to allow
  // for nodes omitting calls as scratch buffer indexes is in pre-computed OpData
  int next_scratch_buffer_idx;
  )";

  wr.writeArray(arenaMap_.nodesScratchBufferAllocationCounts(), sizeof(uint8_t), true,
    "const uint8_t", "node_scratch_buffer_requests"
  );
  wr << "\n";
  wr.writeArray(arenaMap_.scratchBufOffsets(), sizeof(size_t),  true,
    "const size_t", "scratchbuf_offsets");
}



void tflmc::Compiler::writeContextAllocationHandlersSource(CodeWriter &wr) {

  // We assume that persistent allocations are made from the end
  // of the arena downwards.  We should really have have a CI test 
  // to verify this explicitly but it is VERY unlikely the other
  // tests will pass if tflite(u) changes this one day.
  // Obviously adding support for external memory allocation 
  // would complicate this... 

  wr << R"(
void *AllocatePersistentBuffer(struct TfLiteContext* ignore,
                                                 size_t bytes) {
  static uint8_t *AllocPtr = tensor_arena + sizeof(tensor_arena);

  AllocPtr -= bytes;
  return AllocPtr;
}

TfLiteEvalTensor *GetEvalTensor(const struct TfLiteContext *ignore,
                                       int tensor_idx) {
  return &evalTensors[tensor_idx];
}
)";

// Scratch buffers are "easy" - we simply re-use the allocations
// from our offline init/prepare phases.  This must of course
// match the target build.  Worse case same kernel library,
// target compiler settings and target compiler.
// Complication: nodes with offline pre-computed user_data (OpData)
// won't actually call RequestScratchBufferInArena
// so we record the calls each node made and corect next_scratch_buffer_idx
// from that after each prepare call.

  wr << R"(
TfLiteStatus RequestScratchBufferInArena(TfLiteContext *ignored,
                                                size_t bytes_ignored,
                                                int *buffer_idx) {
  *buffer_idx = next_scratch_buffer_idx;
  ++next_scratch_buffer_idx;
  return kTfLiteOk;
}

void* GetScratchBuffer(struct TfLiteContext *ignore, int buffer_idx) {
  return tensor_arena + scratchbuf_offsets[buffer_idx];
}
)";

}


void tflmc::Compiler::writeMicroContextSource(CodeWriter &wr) {

  wr << R"(
class )" << prefix_ << R"(PreinterpretedMicroContext : public tflite::MicroContext {
 public:
   )" << prefix_ << R"(PreinterpretedMicroContext() : 
    tflite::MicroContext(nullptr, nullptr, nullptr) {}

  // Allocate persistent buffer which has the same life time as the interpreter.
  // Returns nullptr on failure.
  // The memory is allocated from the tail.
  // This method is only available in Init or Prepare stage.
  // Virtual so that it can be faked for kernel tests.
  virtual void* AllocatePersistentBuffer(size_t bytes) {
    return ::AllocatePersistentBuffer(nullptr, bytes);
  }

  // Request a scratch buffer in the arena through static memory planning.
  // This method is only available in Prepare stage and the buffer is allocated
  // by the interpreter between Prepare and Eval stage. In Eval stage,
  // GetScratchBuffer API can be used to fetch the address.
  // Virtual so that it can be faked for kernel tests.
  virtual TfLiteStatus RequestScratchBufferInArena(size_t bytes,
                                                   int* buffer_idx) {
    return ::RequestScratchBufferInArena(nullptr, bytes, buffer_idx);
  }

  // Get the scratch buffer pointer.
  // This method is only available in Eval stage.
  // Virtual so that it can be faked for kernel tests.
  virtual void* GetScratchBuffer(int buffer_idx) {
    return ::GetScratchBuffer(nullptr, buffer_idx);
  }

  // Returns a temporary TfLiteTensor struct for a given index.
  // Virtual so that it can be faked for kernel tests.
  virtual TfLiteTensor* AllocateTempTfLiteTensor(int tensor_idx) {
    return tensor_idx >= 0 ? &tflTensors[tensor_idx] : nullptr;
  }

  // Returns a temporary TfLiteTensor struct for the specified input tensor of a
  // given mode. This is the recommended API over the deprecated
  // GetInput/GetInputSafe to get a temp input tensor. The returned tensor shall
  // be freed via calling DeallocateTempTfLiteTensor.
  virtual TfLiteTensor* AllocateTempInputTensor(const TfLiteNode* node,
                                                int index) {
    return AllocateTempTfLiteTensor(node->inputs->data[index]);
  }

  // Returns a temporary TfLiteTensor struct for the specified output tensor of
  // a given mode. This is the recommended API over the deprecated
  // GetOutput/GetOutputSafe to get a temp output tensor. The returned tensor
  // shall be freed via calling DeallocateTempTfLiteTensor.
  virtual TfLiteTensor* AllocateTempOutputTensor(const TfLiteNode* node,
                                                 int index) {
    return AllocateTempTfLiteTensor(node->outputs->data[index]);
  }

  // Deallocates a temp TfLiteTensor.
  // Virtual so that it can be faked for kernel tests.
  virtual void DeallocateTempTfLiteTensor(TfLiteTensor* tensor) {
    // No-op
  }

  // Returns a TfLiteEvalTensor struct for a given index.
  // Virtual so that it can be faked for kernel tests.
  virtual TfLiteEvalTensor* GetEvalTensor(int tensor_idx) {
    return ::GetEvalTensor(nullptr, tensor_idx);
  }


  // Does not take ownership of the pointer and the pointer must refer to valid
  // an object that outlive this class instance.
  // This can only be called once to set one external context.
  TfLiteStatus set_external_context(void* external_context_payload);

  void* external_context() { return external_context_payload_; }
protected:
  void* external_context_payload_ = nullptr;

  TF_LITE_REMOVE_VIRTUAL_DELETE
};

)";

}


void tflmc::Compiler::writeInitSource(CodeWriter &wr) {

  wr << R"(extern "C" TfLiteStatus )"
     << prefix_ << R"(init() {
  ctx.AllocatePersistentBuffer = &AllocatePersistentBuffer;
  ctx.RequestScratchBufferInArena = &RequestScratchBufferInArena;
  ctx.GetScratchBuffer = &GetScratchBuffer;
  ctx.GetEvalTensor = &GetEvalTensor;
  ctx.tensors = tflTensors;
)";
  wr << "  ctx.tensors_size = " << tensors_.size() << ";\n";

  wr << R"(
  static )" << prefix_ << R"(PreinterpretedMicroContext u_ctx;
  ctx.impl_ = static_cast<void *>(&u_ctx);
)";

  // TODO: Do we really support variable tensors?
  // TODO: Do we encounter other than kTfLiteMmapRo and kTfLiteArenaRw, if so we
  // need to store the type separately.

  wr << R"(
  TfLiteIntArray dimsEmptyTensor = {0};
  tflTensors[-1].dims = &dimsEmptyTensor;
  tflTensors[-1].data.raw = nullptr;
)";
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
  if (has_is_variable) {
    wr << "    tflTensors[i].is_variable = tensorData[i].is_variable;\n";
  } else {
    wr << "    tflTensors[i].is_variable = false;\n";
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
)";
#if SUPPORT_CUSTOM_QUANT
    wr << R"(    } else if (tflTensors[i].quantization.type == kTfLitePackedAffineQuantization) {
      TfLitePackedAffineQuantization const* quant = (TfLitePackedAffineQuantization const*)(tensorData[i].quantization.params);
      tflTensors[i].params.scale = quant->affine.scale->data[0];
      tflTensors[i].params.zero_point = quant->affine.zero_point->data[0];
)";
#endif  // SUPPORT_CUSTOM_QUANT
    wr << R"(    }
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
  }
  wr << "\n";
#if TF_LITE_MICRO_RECORD_OP_USER_DATA
  wr << R"(
#if TF_LITE_MICRO_USE_OFFLINE_OP_USER_DATA
tflite::micro::resetOfflineOpUserData( tflite::micro::)" << prefix_ << R"(model::precomputed_op_user_data);
#endif  // TF_LITE_MICRO_USE_OFFLINE_OP_USER_DATA
)";
#endif
  wr << R"(  for(size_t i = 0; i < kOpNodesCount; ++i) {
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

#if TF_LITE_MICRO_RECORD_OP_USER_DATA
  wr << R"(
#if TF_LITE_MICRO_USE_OFFLINE_OP_USER_DATA
tflite::micro::resetOfflineOpUserData( tflite::micro::)" << prefix_ << R"(model::precomputed_op_user_data);
#endif  // TF_LITE_MICRO_USE_OFFLINE_OP_USER_DATA
)";
#endif

  wr << R"(  size_t precomputed_sb_idx_ctr = 0;
  
  for(size_t i = 0; i < kOpNodesCount; ++i) {
    next_scratch_buffer_idx = precomputed_sb_idx_ctr;
    if (registrations[nodeData[i].used_op_index].prepare) {
      TfLiteStatus status = registrations[nodeData[i].used_op_index].prepare(&ctx, &tflNodes[i]);
      if (status != kTfLiteOk) {
        return status;
      }
    }
    precomputed_sb_idx_ctr += node_scratch_buffer_requests[i];
  }
  return kTfLiteOk;
}
)";

}


void tflmc::Compiler::writeTensorAccessorsSource(CodeWriter &wr) {
  wr << R"(
extern "C" TfLiteTensor* )"
        << prefix_ << R"(input(int index) {  
    static const int inTensorIndices[] = {
    )";
    for (auto inIndex : inputTensorIndices_) {
      wr << inIndex << ", ";
    }
    wr << R"(
    };
    return &ctx.tensors[inTensorIndices[index]];
  }

extern "C" TfLiteTensor* )"
        << prefix_ << R"(output(int index) {
    static const int outTensorIndices[] = {
    )";  // TODO: perhaps use a smaller type than int?
    for (auto outIndex : outputTensorIndices_) {
      wr << outIndex << ", ";
    }
    wr << R"(
    };
    return &ctx.tensors[outTensorIndices[index]];
  }
  )";

  
  std::string code = R"(

// Returns the number of input tensors.
extern "C" size_t %PREFIX%inputs() {
  return )" + std::to_string(inputTensorIndices_.size()) +
                     R"(;
}
// Returns the number of output tensors.
extern "C" size_t %PREFIX%outputs() {
  return )" + std::to_string(outputTensorIndices_.size()) +
                     R"(;
}

extern "C" void *%PREFIX%input_ptr(int index) {
  return %PREFIX%input(index)->data.data;
}
extern "C" size_t %PREFIX%input_size(int index) {
  return %PREFIX%input(index)->bytes;
}
extern "C" int %PREFIX%input_dims_len(int index) {
  return %PREFIX%input(index)->dims->size;
}
extern "C" int *%PREFIX%input_dims(int index) {
  return &%PREFIX%input(index)->dims->data[0];
}

extern "C" void *%PREFIX%output_ptr(int index) {
  return %PREFIX%output(index)->data.data;
}
extern "C" size_t %PREFIX%output_size(int index) {
  return %PREFIX%output(index)->bytes;
}
extern "C" int %PREFIX%output_dims_len(int index) {
  return %PREFIX%output(index)->dims->size;
}
extern "C" int *%PREFIX%output_dims(int index) {
  return &%PREFIX%output(index)->dims->data[0];
}

)";

  static std::regex rePrefix("%PREFIX%");
  code = std::regex_replace(code, rePrefix, prefix_);

  wr << code;

}


void tflmc::Compiler::writeInvokeSource(CodeWriter &wr) {
  wr << R"(

extern "C" TfLiteStatus )"
      << prefix_ << R"(invoke() {
)";
#if TF_LITE_MICRO_RECORD_OP_USER_DATA
  wr << R"(
#if TF_LITE_MICRO_USE_OFFLINE_OP_USER_DATA
tflite::micro::resetOfflineOpUserData( tflite::micro::)" << prefix_ << R"(model::precomputed_op_user_data);
#endif  // TF_LITE_MICRO_USE_OFFLINE_OP_USER_DATA
)";
#endif
  wr << R"(
  for(size_t i = 0; i < kOpNodesCount; ++i) {
#if LOG_OP_INPUTS
    tflite::logOpInvoke(&ctx,  &tflNodes[i]);
#endif
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
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/compatibility.h"
#include "tensorflow/lite/micro/micro_context.h"
#if LOG_OP_INPUTS
#include "tensorflow/lite/micro/micro_invoke_log.h"
#endif
)";

#if TF_LITE_MICRO_RECORD_OP_USER_DATA
  wr << R"(
#if TF_LITE_MICRO_USE_OFFLINE_OP_USER_DATA
#include "tensorflow/lite/micro/kernels/ifx_common/offline_prepare_utils.h" 
#endif  // TF_LITE_MICRO_USE_OFFLINE_OP_USER_DATA
)";
#endif

  wr << R"(

#if defined __GNUC__
#define ALIGN(X) __attribute__((aligned(X)))
#elif defined _MSC_VER
#define ALIGN(X) __declspec(align(X))
#elif defined __TASKING__
#define ALIGN(X) __align(X)
#endif

)";

#if TF_LITE_MICRO_RECORD_OP_USER_DATA
  wr << "#if TF_LITE_MICRO_USE_OFFLINE_OP_USER_DATA\n";
  tflite::micro::writeStaticOpDataHeaders(out);
  wr << "#endif  // TF_LITE_MICRO_USE_OFFLINE_OP_USER_DATA\n";
#endif

  writeCustomRegistrationsSource(wr);

  writeTypesAndWorkingArraysSource(wr);

  writeTflNodesSource(wr);

  writeTensorDataSource(wr);

  writeNodeDataSource(wr);

  writeScratchBufferOffsets(wr);

  writeContextAllocationHandlersSource(wr);

// TODO:  Really need to support AllocateBufferForEval.  Should be easy - just need to
// permit allocating a suitable "gap" in the arena or a dedicated scratchpad area.

wr << R"(
} // namespace
)";

#if TF_LITE_MICRO_RECORD_OP_USER_DATA
  wr << "#if TF_LITE_MICRO_USE_OFFLINE_OP_USER_DATA\n";
  tflite::micro::writeStaticOpDataDefinitions(prefix_, out);
  wr << "#endif  // TF_LITE_MICRO_USE_OFFLINE_OP_USER_DATA\n";
#endif

  writeMicroContextSource(wr);

  writeInitSource(wr);

  writeTensorAccessorsSource(wr);
  
  writeInvokeSource(wr);

  finalizeMemMap(wr);
}


void tflmc::Compiler::writeHeader(std::ostream &out) {
  tflmc::CodeWriter wr(out, subgraph_, errReporter());

  std::string code = R"(
#ifndef %PREFIX%GEN_H
#define %PREFIX%GEN_H

#include "tensorflow/lite/c/common.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

#define %PREFIX%MODEL_CONST_DATA_SIZE )"+ std::to_string(constMemMap_.size()) +
                R"(
#define %PREFIX%MODEL_INIT_DATA_SIZE )"+ std::to_string(initMemMap_.size()) +
                R"(
#define %PREFIX%MODEL_UNINIT_DATA_SIZE )"+ std::to_string(uninitMemMap_.size()) +
                R"(


// Sets up the model with init and prepare steps.
TfLiteStatus %PREFIX%init();
// Returns the input tensor with the given index.
TfLiteTensor *%PREFIX%input(int index);
// Returns the output tensor with the given index.
TfLiteTensor *%PREFIX%output(int index);
// Runs inference for the model.
TfLiteStatus %PREFIX%invoke();

// Returns the number of input tensors.
size_t %PREFIX%inputs();

// Returns the number of output tensors.
size_t %PREFIX%outputs();

// Return the buffer pointer of input tensor
void *%PREFIX%input_ptr(int index);

// Return the buffer size of input tensor
size_t %PREFIX%input_size(int index);

// Return the dimention size of input tensor
int %PREFIX%input_dims_len(int index);

// Return the dimention buffer pointer of input tensor
int *%PREFIX%input_dims(int index);

// Return the buffer pointer of output tensor
void *%PREFIX%output_ptr(int index);

// Return the buffer size of output tensor
size_t %PREFIX%output_size(int index);

// Return the dimention size of output tensor
int %PREFIX%output_dims_len(int index);

// Return the dimention buffer pointer of output tensor
int *%PREFIX%output_dims(int index);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

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
    auto nodeAndReg = interpreter_->node_and_registration(ILLEGAL_IF_EVER_MULTIPLE_SUBGRAPH,i);
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

bool tflmc::Compiler::noErrorsReported() const { 
  return ! microErrReporter_.getErrorReported();
}
