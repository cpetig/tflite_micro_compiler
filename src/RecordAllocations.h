#ifndef TFLMCOMPILER_RECORDALLOCATIONS_H
#define TFLMCOMPILER_RECORDALLOCATIONS_H

#include "tensorflow/lite/schema/schema_generated.h"
#include <cinttypes>

namespace tflmc {

enum AllocKind : int {
  Persistent,
  Scratch
};

struct Allocation {
  ptrdiff_t offset;
  size_t len;
  int nodeIndex;
  AllocKind kind;
};


  // We can try to use a stock kernel but
  // this requires us to access private data and re-execute
  // Fragmnents of MicroInterpreter::AllocateTensors() with instrumented context
  // API calls.  Painful to maintain and prone subtle Bugs.  Simpler to maintain a patch
  // that adds hooks to MicroInterpreter to gather data from an 
  // actual MicroInterpreter::AllocateTensors() by intercepting the TfliteContext vectors
  // which are a reasonably stable API.

#if TFLMC_USE_INTERPRETER_HOOKS

void SetRecordAllocationhooks(tflite::MicroInterpreter *interpreter,  
                              uint8_t *arena_start,
                              size_t arena_size);

void RecordScratchBufferAllocations(tflite::MicroInterpreter *interpreter);

#else
void RecordAllocations(
  const tflite::Model *model,  size_t arena_size, size_t arena_alignment);
#endif

const std::vector<Allocation> &RecordedAllocations();

TfLiteEvalTensor *GetEvalTensor(tflite::MicroInterpreter *interpreter, int i);
TfLiteTensor *GetTensor(tflite::MicroInterpreter *interpreter, int i);

}  // namespace tflmc

#endif
