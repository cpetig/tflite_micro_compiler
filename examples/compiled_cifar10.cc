// This file is generated. Do not edit.
// Generated on: 27.07.2020 22:27:22

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

namespace {

constexpr int kTensorArenaSize = 1536;
uint8_t tensor_arena[kTensorArenaSize] ALIGN(16);
template <int SZ, class T> struct TfArray {
  int sz; T elem[SZ];
};
enum used_operators_e {
  OP_CONV_2D, OP_MAX_POOL_2D, OP_RESHAPE, OP_FULLY_CONNECTED,  OP_LAST
};
struct TensorInfo_t { // subset of TfLiteTensor used for initialization from constant memory
  TfLiteType type;
  void* data;
  TfLiteIntArray* dims;
  size_t bytes;
  bool is_variable;
};
struct NodeInfo_t { // subset of TfLiteNode used for initialization from constant memory
  struct TfLiteIntArray* inputs;
  struct TfLiteIntArray* outputs;
  void* builtin_data;
  used_operators_e used_op_index;
};

TfLiteContext ctx{};
TfLiteTensor tflTensors[0];
TfLiteRegistration registrations[OP_LAST];
TfLiteNode tflNodes[8];

const TfLiteConvParams opdata0 = { kTfLitePaddingValid, 1,1, kTfLiteActRelu, 1,1 };
const TfArray<3, int> inputs0 = { 3, { 0,9,1 } };
const TfArray<1, int> outputs0 = { 1, { 12 } };
const TfLitePoolParams opdata1 = { kTfLitePaddingValid, 2,2, 2,2, kTfLiteActNone, { { 0,0, 0, 0 } } };
const TfArray<1, int> inputs1 = { 1, { 12 } };
const TfArray<1, int> outputs1 = { 1, { 13 } };
const TfLiteConvParams opdata2 = { kTfLitePaddingValid, 1,1, kTfLiteActRelu, 1,1 };
const TfArray<3, int> inputs2 = { 3, { 13,10,2 } };
const TfArray<1, int> outputs2 = { 1, { 14 } };
const TfLitePoolParams opdata3 = { kTfLitePaddingValid, 2,2, 2,2, kTfLiteActNone, { { 0,0, 0, 0 } } };
const TfArray<1, int> inputs3 = { 1, { 14 } };
const TfArray<1, int> outputs3 = { 1, { 15 } };
const TfLiteConvParams opdata4 = { kTfLitePaddingValid, 1,1, kTfLiteActRelu, 1,1 };
const TfArray<3, int> inputs4 = { 3, { 15,11,3 } };
const TfArray<1, int> outputs4 = { 1, { 16 } };
const TfLiteReshapeParams opdata5 = { { 0, 0, 0, 0, 0, 0, 0, 0, }, 0 };
const TfArray<2, int> inputs5 = { 2, { 16,6 } };
const TfArray<1, int> outputs5 = { 1, { 17 } };
const TfLiteFullyConnectedParams opdata6 = { kTfLiteActRelu, kTfLiteFullyConnectedWeightsFormatDefault, false, false };
const TfArray<3, int> inputs6 = { 3, { 17,7,4 } };
const TfArray<1, int> outputs6 = { 1, { 18 } };
const TfLiteFullyConnectedParams opdata7 = { kTfLiteActNone, kTfLiteFullyConnectedWeightsFormatDefault, false, false };
const TfArray<3, int> inputs7 = { 3, { 18,8,5 } };
const TfArray<1, int> outputs7 = { 1, { 19 } };
const TensorInfo_t tensorData[] = {
};const NodeInfo_t nodeData[] = {
  { (TfLiteIntArray*)&inputs0, (TfLiteIntArray*)&outputs0, const_cast<void*>(static_cast<const void*>(&opdata0)), OP_CONV_2D, },
  { (TfLiteIntArray*)&inputs1, (TfLiteIntArray*)&outputs1, const_cast<void*>(static_cast<const void*>(&opdata1)), OP_MAX_POOL_2D, },
  { (TfLiteIntArray*)&inputs2, (TfLiteIntArray*)&outputs2, const_cast<void*>(static_cast<const void*>(&opdata2)), OP_CONV_2D, },
  { (TfLiteIntArray*)&inputs3, (TfLiteIntArray*)&outputs3, const_cast<void*>(static_cast<const void*>(&opdata3)), OP_MAX_POOL_2D, },
  { (TfLiteIntArray*)&inputs4, (TfLiteIntArray*)&outputs4, const_cast<void*>(static_cast<const void*>(&opdata4)), OP_CONV_2D, },
  { (TfLiteIntArray*)&inputs5, (TfLiteIntArray*)&outputs5, const_cast<void*>(static_cast<const void*>(&opdata5)), OP_RESHAPE, },
  { (TfLiteIntArray*)&inputs6, (TfLiteIntArray*)&outputs6, const_cast<void*>(static_cast<const void*>(&opdata6)), OP_FULLY_CONNECTED, },
  { (TfLiteIntArray*)&inputs7, (TfLiteIntArray*)&outputs7, const_cast<void*>(static_cast<const void*>(&opdata7)), OP_FULLY_CONNECTED, },
};
static void* AllocatePersistentBuffer(struct TfLiteContext* ctx,
                                                 size_t bytes) {
  static uint8_t *AllocPtr = tensor_arena + sizeof(tensor_arena);

  AllocPtr -= bytes;
  return AllocPtr;
}
} // namespace

TfLiteStatus cifar_init() {
  ctx.AllocatePersistentBuffer = &AllocatePersistentBuffer;
  ctx.tensors = tflTensors;
  ctx.tensors_size = 0;
  for(size_t i = 0; i < 0; ++i) {
    tflTensors[i].data.data = tensorData[i].data;
    tflTensors[i].type = tensorData[i].type;
    tflTensors[i].is_variable = tensorData[i].is_variable;
    tflTensors[i].allocation_type = (tensor_arena <= tensorData[i].data && tensorData[i].data < tensor_arena + kTensorArenaSize) ? kTfLiteArenaRw : kTfLiteMmapRo;
    tflTensors[i].bytes = tensorData[i].bytes;
    tflTensors[i].dims = tensorData[i].dims;
    tflTensors[i].quantization.type = kTfLiteNoQuantization;
  }
  registrations[OP_CONV_2D] = tflite::ops::micro::Register_CONV_2D();
  registrations[OP_MAX_POOL_2D] = tflite::ops::micro::Register_MAX_POOL_2D();
  registrations[OP_RESHAPE] = tflite::ops::micro::Register_RESHAPE();
  registrations[OP_FULLY_CONNECTED] = tflite::ops::micro::Register_FULLY_CONNECTED();

  for(size_t i = 0; i < 8; ++i) {
    tflNodes[i].inputs = nodeData[i].inputs;
    tflNodes[i].outputs = nodeData[i].outputs;
    tflNodes[i].builtin_data = nodeData[i].builtin_data;
    tflNodes[i].custom_initial_data = nullptr;
    tflNodes[i].custom_initial_data_size = 0;
    if (registrations[nodeData[i].used_op_index].init) {
      tflNodes[i].user_data = registrations[nodeData[i].used_op_index].init(&ctx, (const char*)tflNodes[i].builtin_data, 0);
    }
  }
  for(size_t i = 0; i < 8; ++i) {
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
  0, 
};
TfLiteTensor* cifar_input(int index) {
  return &ctx.tensors[inTensorIndices[index]];
}

static const int outTensorIndices[] = {
  19, 
};
TfLiteTensor* cifar_output(int index) {
  return &ctx.tensors[outTensorIndices[index]];
}

TfLiteStatus cifar_invoke() {
  for(size_t i = 0; i < 8; ++i) {
    TfLiteStatus status = registrations[nodeData[i].used_op_index].invoke(&ctx, &tflNodes[i]);
    if (status != kTfLiteOk) {
      return status;
    }
  }
  return kTfLiteOk;
}
