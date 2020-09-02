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


std::vector<Allocation> RecordAllocations(
  const tflite::Model *model,  size_t arena_size, size_t arena_alignment);


TfLiteEvalTensor *GetEvalTensor(tflite::MicroInterpreter *interpreter, int i);
TfLiteTensor *GetTensor(tflite::MicroInterpreter *interpreter, int i);

}  // namespace tflmc

#endif
