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

static TfLiteStatus LoggingAllocatePersistentBuffer(struct TfLiteContext *ctx,
                                                    size_t bytes, void **ptr) {
  auto retVal = g_allocator->AllocatePersistentBuffer(bytes, ptr);
  assert(retVal == kTfLiteOk && "Alloc failure");
  g_loggedAllocations.push_back(
      {-(g_arenaPtr - (uint8_t *)*ptr + SUFFICIENT_ARENA_SIZE), bytes,
       g_currentNodeIndex});
  return retVal;
}
static TfLiteStatus LoggingRequestScratchBufferInArena(TfLiteContext *ctx,
                                                       size_t bytes,
                                                       int *buffer_idx) {
  assert(false && "Not handling scratch buffers currently");
  return g_allocator->RequestScratchBufferInArena(g_currentNodeIndex, bytes,
                                                  buffer_idx);
}

std::vector<tflmc::Allocation> tflmc::RecordAllocations(
    const tflite::Model *model) {
  std::vector<uint8_t> arena_buf(SUFFICIENT_ARENA_SIZE);
  g_arenaPtr = arena_buf.data();

  tflite::MicroErrorReporter error_reporter;
  tflite::AllOpsResolver resolver;
  tflmc::custom_operator_handle custom = tflmc::LoadCustom(&resolver);
  tflite::MicroInterpreter interpreter(model, resolver, arena_buf.data(),
                                       SUFFICIENT_ARENA_SIZE, &error_reporter);

  auto ctx = &interpreter.context_;
  auto allocator = &interpreter.allocator_;

  tflite::NodeAndRegistration *nodeAndRegs;
  allocator->StartModelAllocation(model, ctx, resolver, &nodeAndRegs);
  allocator->FinishModelAllocation(model, ctx);

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
