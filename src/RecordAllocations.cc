#include <sstream>
#include <iostream>
#include <memory>
#include <map>
#define private public
#include "tensorflow/lite/micro/micro_interpreter.h"
#undef private

#include "CustomOperators.h"
#include "RecordAllocations.h"
#include "MemMap.h"
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"

static std::vector<tflmc::Allocation> g_loggedAllocations;
static tflite::MicroAllocator *g_allocator;
static int g_currentNodeIndex = -1;
static uint8_t *g_arenaPtr = nullptr;

static ptrdiff_t g_arena_size = 0;

struct ScratchBufferInfo {
    	int node_id;
      size_t bytes;
};

static std::map<int, ScratchBufferInfo> g_logged_scratch_buffers;
static TfLiteStatus LoggingAllocatePersistentBuffer(struct TfLiteContext *ctx,
                                                    size_t bytes, void **ptr) {
  auto retVal = g_allocator->AllocatePersistentBuffer(bytes, ptr);
  assert(retVal == kTfLiteOk && "Alloc failure");
#if 0
  ptrdiff_t offset = -(g_arenaPtr - (uint8_t *)*ptr + g_arena_size);
#endif
  ptrdiff_t offset = (uint8_t *)*ptr - g_arenaPtr;
  std::cout << "PERSISTENT FOR " << g_currentNodeIndex << " offset " << offset << " for " << bytes << std::endl;

  g_loggedAllocations.push_back(
      {offset, bytes,
       g_currentNodeIndex, tflmc::AllocKind::Persistent});
  return retVal;
}
static TfLiteStatus LoggingRequestScratchBufferInArena(TfLiteContext *ctx,
                                                       size_t bytes,
                                                       int *buffer_idx) {

  auto res =  g_allocator->RequestScratchBufferInArena(g_currentNodeIndex, bytes,
                                                       buffer_idx);
  if (res == kTfLiteOk) {
    std::cout << "SCRATCH FOR " << g_currentNodeIndex << " " << bytes << std::endl;
    g_logged_scratch_buffers[*buffer_idx] = {g_currentNodeIndex, bytes};
  }
  return res;                                         
}


  // HACK: here in essence, create a duplicate interpreter here and re-execute
  // Fragmnents of MicroInterpreter::AllocateTensors() with instrumented context
  // API calls.  Painful to maintain - surely simpler to maintain a patch
  // that adds hooks to MicroInterpreter to gather data from an actual MicroInterpreter::AllocateTensors() 
  // call

std::vector<tflmc::Allocation> tflmc::RecordAllocations(
    const tflite::Model *model, 
     size_t arena_size,  size_t arena_alignment) {

  tflmc::SufficientArena arena(arena_size, arena_alignment);
  g_arenaPtr = arena.alginedBufferStart();
  g_arena_size = arena_size;

  tflite::MicroErrorReporter error_reporter;

  // Resolver must be passed in  as otherwise pointers to its internal table
  // in the arena will be invalidated....

  
  tflite::AllOpsResolver resolver;
  tflmc::custom_operator_handle custom = tflmc::LoadCustom(&resolver);
  tflite::MicroInterpreter interpreter(model, resolver, g_arenaPtr,
                                       g_arena_size, &error_reporter);

  auto ctx = &interpreter.context_;
  auto allocator = &interpreter.allocator_;

  tflite::NodeAndRegistration *nodeAndRegs;
  TfLiteEvalTensor *eval_tensors=nullptr;
  allocator->StartModelAllocation(model, resolver, &nodeAndRegs, &eval_tensors);

  g_allocator = allocator;
  ctx->AllocatePersistentBuffer = &LoggingAllocatePersistentBuffer;
  ctx->RequestScratchBufferInArena = nullptr;
  ctx->GetScratchBuffer = nullptr;


  auto subgraph = model->subgraphs()->Get(0);
  for (size_t i = 0; i < subgraph->operators()->size(); i++) {
    auto node = &nodeAndRegs[i].node;
    auto reg = nodeAndRegs[i].registration;
    if (reg->init) {
      g_currentNodeIndex = i;
      node->user_data = reg->init(ctx, (const char *)node->builtin_data, 0);
    }
  }


  ctx->RequestScratchBufferInArena = &LoggingRequestScratchBufferInArena;

  for (size_t i = 0; i < subgraph->operators()->size(); i++) {
    auto node = &nodeAndRegs[i].node;
    auto reg = nodeAndRegs[i].registration;
    if (reg->prepare) {
      g_currentNodeIndex = i;
      reg->prepare(ctx, node);
    }
    allocator->ResetTempAllocations();
  }
  
  allocator->FinishModelAllocation(model, eval_tensors);

  tflmc::UnloadCustom(custom);
  for( auto &sb_i : g_logged_scratch_buffers )
  {
      auto sb_idx = sb_i.first;
      void *sb_start = allocator->GetScratchBuffer( sb_idx );
      assert(sb_start != nullptr && "Unknown Scratch Buffer");
  #if 0
      ptrdiff_t offset = -(g_arenaPtr - (uint8_t *)sb_start + g_arena_size);
  #endif
      ptrdiff_t offset = (uint8_t *)sb_start - g_arenaPtr;
      std::cout << "SCRATCH FOR " << sb_i.second.node_id << " offset " << offset << " for " << sb_i.second.bytes << std::endl;

      g_loggedAllocations.push_back(
      {offset, sb_i.second.bytes,
       sb_i.second.node_id, tflmc::AllocKind::Scratch});
  }
  
  return g_loggedAllocations;
}

TfLiteEvalTensor *tflmc::GetEvalTensor(tflite::MicroInterpreter *interpreter, int i) {
  auto ctx = &interpreter->context_;
  return ctx->GetEvalTensor(ctx, i);
}

TfLiteTensor *tflmc::GetTensor(tflite::MicroInterpreter *interpreter, int i) {
  auto ctx = &interpreter->context_;
  return ctx->GetTensor(ctx, i);
}
