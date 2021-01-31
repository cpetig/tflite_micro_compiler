#include <sstream>
#define private public
#include "tensorflow/lite/micro/micro_interpreter.h"
#undef private

#include "CustomOperators.h"
#include "RecordAllocations.h"
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"

static std::vector<tflmc::Allocation> g_loggedAllocations;
static tflite::MicroAllocator *g_allocator;
static int g_currentNodeIndex = -1;
static uint8_t *g_arenaPtr = nullptr;

static ptrdiff_t g_arena_size = 0;

static void* LoggingAllocatePersistentBuffer(struct TfLiteContext *ctx,
                                                    size_t bytes) {
  void* ptr = g_allocator->AllocatePersistentBuffer(bytes);
  assert(ptr!=nullptr && "Alloc failure");
  g_loggedAllocations.push_back(
      {-(g_arenaPtr - (uint8_t *)ptr + g_arena_size), bytes,
       g_currentNodeIndex});
  return ptr;
}
static TfLiteStatus LoggingRequestScratchBufferInArena(TfLiteContext *ctx,
                                                       size_t bytes,
                                                       int *buffer_idx) {
  assert(false && "Not handling scratch buffers currently");
  return g_allocator->RequestScratchBufferInArena(bytes,
                                                  buffer_idx);
}

std::vector<tflmc::Allocation> tflmc::RecordAllocations(
    const tflite::Model *model, ptrdiff_t arena_size) {
  g_arena_size = arena_size;
  std::vector<uint8_t> arena_buf(g_arena_size);
  g_arenaPtr = arena_buf.data();

  tflite::MicroErrorReporter error_reporter;
  tflite::AllOpsResolver resolver;
  tflmc::custom_operator_handle custom = tflmc::LoadCustom(&resolver);
  tflite::MicroInterpreter interpreter(model, resolver, arena_buf.data(),
                                       g_arena_size, &error_reporter);

  auto ctx = &interpreter.context_;
  auto allocator = &interpreter.allocator_;

  tflite::NodeAndRegistration *nodeAndRegs;
  TfLiteEvalTensor *eval_tensors=nullptr;
  tflite::ScratchBufferHandle* scratchhandle=nullptr;

  allocator->StartModelAllocation(model, resolver, &nodeAndRegs, &eval_tensors);
  allocator->FinishModelAllocation(model, eval_tensors, &scratchhandle);

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
  }
  tflmc::UnloadCustom(custom);
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
