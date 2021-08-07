#include <iostream>
#include <sstream>
#include <memory>
#include <map>

#if !TFLMC_USE_INTERPRETER_HOOKS
#define private public
#endif
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_allocator.h"
#include "tensorflow/lite/micro/micro_graph.h"

#if !TFLMC_USE_INTERPRETER_HOOKS
#undef private
#endif

#include "CustomOperators.h"
#include "RecordAllocations.h"
#include "MemMap.h"
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"

static std::vector<tflmc::Allocation> g_loggedAllocations;
static int g_currentNodeIndex = -1;
static uint8_t *g_arenaPtr = nullptr;

static ptrdiff_t g_arena_size = 0;


struct ScratchBufferInfo {
    	int node_id;
      size_t bytes;
};

static std::map<int, ScratchBufferInfo> g_logged_scratch_buffers;



#if TFLMC_USE_INTERPRETER_HOOKS


static  tflite::MicroInterpreter::TfLiteContextHooks *g_tflm_hooks;


static void *LoggingAllocatePersistentBuffer(struct TfLiteContext *ctx, size_t bytes) {
  auto ptr =  g_tflm_hooks->AllocatePersistentBuffer(ctx, bytes);
  assert(ptr != nullptr && "Alloc failure");
  ptrdiff_t offset = (uint8_t *)ptr - g_arenaPtr;

  g_loggedAllocations.push_back(
      {offset, bytes,
       g_currentNodeIndex, tflmc::AllocKind::Persistent, 0});
  return ptr;
}


static TfLiteStatus LoggingRequestScratchBufferInArena(TfLiteContext *ctx,
                                                       size_t bytes,
                                                       int *buffer_idx) {

  auto res = g_tflm_hooks->RequestScratchBufferInArena(ctx, bytes,  buffer_idx);
  if (res == kTfLiteOk) {
    g_logged_scratch_buffers[*buffer_idx] = {g_currentNodeIndex, bytes};
  }
  return res;                                         
}



static void* LoggingGetScratchBuffer(struct TfLiteContext* ctx, int buffer_idx) {
  return g_tflm_hooks->GetScratchBuffer (ctx, buffer_idx);
}

static void LoggingSetNodeIndex(const struct TfLiteContext* context,
                                int idx) {
  g_currentNodeIndex = idx;
  return g_tflm_hooks->SetNodeIndex(context, idx);
}

static  tflite::MicroInterpreter::TfLiteContextHooks g_recording_hooks =
{
  LoggingAllocatePersistentBuffer,
  LoggingRequestScratchBufferInArena,
  LoggingGetScratchBuffer,
  LoggingSetNodeIndex
  
};

void tflmc::SetRecordAllocationhooks( tflite::MicroInterpreter *interpreter, 
                              uint8_t *arena_start,
                              size_t arena_size) {
  g_tflm_hooks = interpreter->getHooks();
  g_arenaPtr = arena_start;
  g_arena_size = arena_size;
  interpreter->setHooks(&g_recording_hooks);
}

void  tflmc::RecordScratchBufferAllocations(tflite::MicroInterpreter *interpreter)
{
  auto ctx = interpreter->getTFLContext();
  for( auto &sb_i : g_logged_scratch_buffers )
  {
      auto sb_idx = sb_i.first;
      void *sb_start = g_tflm_hooks->GetScratchBuffer(ctx, sb_idx );
      assert(sb_start != nullptr && "Unknown Scratch Buffer");
      ptrdiff_t offset = (uint8_t *)sb_start - g_arenaPtr;
      g_loggedAllocations.push_back(
        {offset, sb_i.second.bytes,
         sb_i.second.node_id, tflmc::AllocKind::Scratch, sb_i.first});
  }
}


TfLiteEvalTensor *tflmc::GetEvalTensor(tflite::MicroInterpreter *interpreter, int i) {
  auto ctx = interpreter->getTFLContext();
  return ctx->GetEvalTensor(ctx, i);
}

TfLiteTensor *tflmc::GetTensor(tflite::MicroInterpreter *interpreter, int i) {
  auto ctx = interpreter->getTFLContext();
  return ctx->GetTensor(ctx, i);
}

#else

static tflite::MicroAllocator *g_allocator;
static void *LoggingAllocatePersistentBuffer(struct TfLiteContext *ctx,
                                                    size_t bytes) {
  auto ptr = g_allocator->AllocatePersistentBuffer(bytes);
  assert(ptr != nullptr && "Alloc failure");
  ptrdiff_t offset = (uint8_t *)ptr - g_arenaPtr;

  g_loggedAllocations.push_back(
      {offset, bytes,
       g_currentNodeIndex, tflmc::AllocKind::Persistent, -1});
  return ptr;
}

static TfLiteStatus LoggingRequestScratchBufferInArena(TfLiteContext *ctx,
                                                       size_t bytes,
                                                       int *buffer_idx) {

  auto res =  g_allocator->RequestScratchBufferInArena(g_currentNodeIndex, bytes,
                                                       buffer_idx);
  if (res == kTfLiteOk) {
    g_logged_scratch_buffers[*buffer_idx] = {g_currentNodeIndex, bytes};
  }
  return res;                                         
}


  // HACK: here in essence, we create a duplicate interpreter here and re-execute
  // Fragmnents of MicroInterpreter::AllocateTensors() with instrumented context
  // API calls.  

void tflmc::RecordAllocations(
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
  
  tflite::SubgraphAllocations *subgraph_allocations = allocator->StartModelAllocation(model);
  std::cout<<__FILE__<<" "<<__LINE__<<" "<<__PRETTY_FUNCTION__<<"\n";  
  
    
  g_allocator = allocator;
  ctx->AllocatePersistentBuffer = &LoggingAllocatePersistentBuffer;
  ctx->RequestScratchBufferInArena = nullptr;
  auto ctx_GetScratchBuffer =  ctx->GetScratchBuffer;
  ctx->GetScratchBuffer = nullptr;


  interpreter.graph_.SetSubgraphAllocations(subgraph_allocations);
  interpreter.PrepareNodeAndRegistrationDataFromFlatbuffer();
  tflite::MicroGraph graph_ = interpreter.graph_;

  for (size_t i = 0; i< graph_.NumSubgraphs();i++)  {
    for(size_t j = 0 ; j< model->subgraphs()->Get(i)->operators()->size();j++){
      TfLiteNode *node = &(subgraph_allocations[i].node_and_registrations[j].node);
      const TfLiteRegistration *reg = subgraph_allocations[i].node_and_registrations[j].registration;
      if(reg == nullptr) std::cout<<"reg is null\n";
      auto code = tflite::EnumValuesBuiltinOperator()[reg->builtin_code];

    std::cout << "operation " << j 
              << ": " << tflite::EnumNamesBuiltinOperator()[code]
              << std::endl;

      if (reg->init) {
      g_currentNodeIndex = i;
      node->user_data = reg->init(ctx, (const char *)node->builtin_data, 0);
      }
    }
  }

  ctx->RequestScratchBufferInArena = &LoggingRequestScratchBufferInArena;

for (size_t i = 0; i< graph_.NumSubgraphs();i++)  {
    for(size_t j = 0 ; j< model->subgraphs()->Get(i)->operators()->size();j++){
      TfLiteNode *node = &(subgraph_allocations[i].node_and_registrations[j].node);
      const TfLiteRegistration *reg = subgraph_allocations[i].node_and_registrations[j].registration;
      if (reg->prepare) {
        g_currentNodeIndex = j;
        reg->prepare(ctx, node);
      }
      allocator->FinishPrepareNodeAllocations(/*node_id=*/j);

    }
  }

#endif

  tflmc::UnloadCustom(custom);
  for( auto &sb_i : g_logged_scratch_buffers )
  {
      auto sb_idx = sb_i.first;
      void *sb_start = ctx_GetScratchBuffer( ctx, sb_idx );
      assert(sb_start != nullptr && "Unknown Scratch Buffer");
      ptrdiff_t offset = (uint8_t *)sb_start - g_arenaPtr;
      g_loggedAllocations.push_back(
        {offset, sb_i.second.bytes,
         sb_i.second.node_id, tflmc::AllocKind::Scratch, -1});
  }

}


TfLiteEvalTensor *tflmc::GetEvalTensor(tflite::MicroInterpreter *interpreter, int i) {
  auto ctx = &interpreter->context_;
  return ctx->GetEvalTensor(ctx, i);
}

TfLiteTensor *tflmc::GetTensor(tflite::MicroInterpreter *interpreter, int i) {
  auto ctx = &interpreter->context_;
  return ctx->GetTensor(ctx, i);
}

#endif

const std::vector<tflmc::Allocation> &tflmc::RecordedAllocations() { 
  return g_loggedAllocations; 
}
