// This file is generated. Do not edit.
// Generated on: 06.08.2020 14:58:04

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

constexpr int kTensorArenaSize = 31424;
uint8_t tensor_arena[kTensorArenaSize] ALIGN(16);
template <int SZ, class T> struct TfArray {
  int sz; T elem[SZ];
};
enum used_operators_e {
  OP_CONV_2D, OP_DEPTHWISE_CONV_2D, OP_AVERAGE_POOL_2D, OP_RESHAPE, OP_SOFTMAX,  OP_LAST
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
TfLiteNode tflNodes[31];

const TfLiteConvParams opdata0 = { kTfLitePaddingSame, 2,2, kTfLiteActNone, 1,1 };
const TfArray<3, int> inputs0 = { 3, { 88,8,6 } };
const TfArray<1, int> outputs0 = { 1, { 7 } };
const TfLiteDepthwiseConvParams opdata1 = { kTfLitePaddingSame, 1,1, 1, kTfLiteActNone, 1,1 };
const TfArray<3, int> inputs1 = { 3, { 7,35,34 } };
const TfArray<1, int> outputs1 = { 1, { 33 } };
const TfLiteConvParams opdata2 = { kTfLitePaddingSame, 1,1, kTfLiteActNone, 1,1 };
const TfArray<3, int> inputs2 = { 3, { 33,38,36 } };
const TfArray<1, int> outputs2 = { 1, { 37 } };
const TfLiteDepthwiseConvParams opdata3 = { kTfLitePaddingSame, 2,2, 1, kTfLiteActNone, 1,1 };
const TfArray<3, int> inputs3 = { 3, { 37,41,40 } };
const TfArray<1, int> outputs3 = { 1, { 39 } };
const TfLiteConvParams opdata4 = { kTfLitePaddingSame, 1,1, kTfLiteActNone, 1,1 };
const TfArray<3, int> inputs4 = { 3, { 39,44,42 } };
const TfArray<1, int> outputs4 = { 1, { 43 } };
const TfLiteDepthwiseConvParams opdata5 = { kTfLitePaddingSame, 1,1, 1, kTfLiteActNone, 1,1 };
const TfArray<3, int> inputs5 = { 3, { 43,47,46 } };
const TfArray<1, int> outputs5 = { 1, { 45 } };
const TfLiteConvParams opdata6 = { kTfLitePaddingSame, 1,1, kTfLiteActNone, 1,1 };
const TfArray<3, int> inputs6 = { 3, { 45,50,48 } };
const TfArray<1, int> outputs6 = { 1, { 49 } };
const TfLiteDepthwiseConvParams opdata7 = { kTfLitePaddingSame, 2,2, 1, kTfLiteActNone, 1,1 };
const TfArray<3, int> inputs7 = { 3, { 49,53,52 } };
const TfArray<1, int> outputs7 = { 1, { 51 } };
const TfLiteConvParams opdata8 = { kTfLitePaddingSame, 1,1, kTfLiteActNone, 1,1 };
const TfArray<3, int> inputs8 = { 3, { 51,56,54 } };
const TfArray<1, int> outputs8 = { 1, { 55 } };
const TfLiteDepthwiseConvParams opdata9 = { kTfLitePaddingSame, 1,1, 1, kTfLiteActNone, 1,1 };
const TfArray<3, int> inputs9 = { 3, { 55,59,58 } };
const TfArray<1, int> outputs9 = { 1, { 57 } };
const TfLiteConvParams opdata10 = { kTfLitePaddingSame, 1,1, kTfLiteActNone, 1,1 };
const TfArray<3, int> inputs10 = { 3, { 57,62,60 } };
const TfArray<1, int> outputs10 = { 1, { 61 } };
const TfLiteDepthwiseConvParams opdata11 = { kTfLitePaddingSame, 2,2, 1, kTfLiteActNone, 1,1 };
const TfArray<3, int> inputs11 = { 3, { 61,65,64 } };
const TfArray<1, int> outputs11 = { 1, { 63 } };
const TfLiteConvParams opdata12 = { kTfLitePaddingSame, 1,1, kTfLiteActNone, 1,1 };
const TfArray<3, int> inputs12 = { 3, { 63,68,66 } };
const TfArray<1, int> outputs12 = { 1, { 67 } };
const TfLiteDepthwiseConvParams opdata13 = { kTfLitePaddingSame, 1,1, 1, kTfLiteActNone, 1,1 };
const TfArray<3, int> inputs13 = { 3, { 67,71,70 } };
const TfArray<1, int> outputs13 = { 1, { 69 } };
const TfLiteConvParams opdata14 = { kTfLitePaddingSame, 1,1, kTfLiteActNone, 1,1 };
const TfArray<3, int> inputs14 = { 3, { 69,74,72 } };
const TfArray<1, int> outputs14 = { 1, { 73 } };
const TfLiteDepthwiseConvParams opdata15 = { kTfLitePaddingSame, 1,1, 1, kTfLiteActNone, 1,1 };
const TfArray<3, int> inputs15 = { 3, { 73,77,76 } };
const TfArray<1, int> outputs15 = { 1, { 75 } };
const TfLiteConvParams opdata16 = { kTfLitePaddingSame, 1,1, kTfLiteActNone, 1,1 };
const TfArray<3, int> inputs16 = { 3, { 75,80,78 } };
const TfArray<1, int> outputs16 = { 1, { 79 } };
const TfLiteDepthwiseConvParams opdata17 = { kTfLitePaddingSame, 1,1, 1, kTfLiteActNone, 1,1 };
const TfArray<3, int> inputs17 = { 3, { 79,83,82 } };
const TfArray<1, int> outputs17 = { 1, { 81 } };
const TfLiteConvParams opdata18 = { kTfLitePaddingSame, 1,1, kTfLiteActNone, 1,1 };
const TfArray<3, int> inputs18 = { 3, { 81,86,84 } };
const TfArray<1, int> outputs18 = { 1, { 85 } };
const TfLiteDepthwiseConvParams opdata19 = { kTfLitePaddingSame, 1,1, 1, kTfLiteActNone, 1,1 };
const TfArray<3, int> inputs19 = { 3, { 85,11,10 } };
const TfArray<1, int> outputs19 = { 1, { 9 } };
const TfLiteConvParams opdata20 = { kTfLitePaddingSame, 1,1, kTfLiteActNone, 1,1 };
const TfArray<3, int> inputs20 = { 3, { 9,14,12 } };
const TfArray<1, int> outputs20 = { 1, { 13 } };
const TfLiteDepthwiseConvParams opdata21 = { kTfLitePaddingSame, 1,1, 1, kTfLiteActNone, 1,1 };
const TfArray<3, int> inputs21 = { 3, { 13,17,16 } };
const TfArray<1, int> outputs21 = { 1, { 15 } };
const TfLiteConvParams opdata22 = { kTfLitePaddingSame, 1,1, kTfLiteActNone, 1,1 };
const TfArray<3, int> inputs22 = { 3, { 15,20,18 } };
const TfArray<1, int> outputs22 = { 1, { 19 } };
const TfLiteDepthwiseConvParams opdata23 = { kTfLitePaddingSame, 2,2, 1, kTfLiteActNone, 1,1 };
const TfArray<3, int> inputs23 = { 3, { 19,23,22 } };
const TfArray<1, int> outputs23 = { 1, { 21 } };
const TfLiteConvParams opdata24 = { kTfLitePaddingSame, 1,1, kTfLiteActNone, 1,1 };
const TfArray<3, int> inputs24 = { 3, { 21,26,24 } };
const TfArray<1, int> outputs24 = { 1, { 25 } };
const TfLiteDepthwiseConvParams opdata25 = { kTfLitePaddingSame, 1,1, 1, kTfLiteActNone, 1,1 };
const TfArray<3, int> inputs25 = { 3, { 25,29,28 } };
const TfArray<1, int> outputs25 = { 1, { 27 } };
const TfLiteConvParams opdata26 = { kTfLitePaddingSame, 1,1, kTfLiteActNone, 1,1 };
const TfArray<3, int> inputs26 = { 3, { 27,32,30 } };
const TfArray<1, int> outputs26 = { 1, { 31 } };
const TfLitePoolParams opdata27 = { kTfLitePaddingValid, 2,2, 5,5, kTfLiteActNone, { { 0,0, 0, 0 } } };
const TfArray<1, int> inputs27 = { 1, { 31 } };
const TfArray<1, int> outputs27 = { 1, { 0 } };
const TfLiteConvParams opdata28 = { kTfLitePaddingSame, 1,1, kTfLiteActNone, 1,1 };
const TfArray<3, int> inputs28 = { 3, { 0,3,2 } };
const TfArray<1, int> outputs28 = { 1, { 1 } };
const TfLiteReshapeParams opdata29 = { { 1, 1001, 0, 0, 0, 0, 0, 0, }, 2 };
const TfArray<2, int> inputs29 = { 2, { 1,5 } };
const TfArray<1, int> outputs29 = { 1, { 4 } };
const TfLiteSoftmaxParams opdata30 = { 1 };
const TfArray<1, int> inputs30 = { 1, { 4 } };
const TfArray<1, int> outputs30 = { 1, { 87 } };
const TensorInfo_t tensorData[] = {
};const NodeInfo_t nodeData[] = {
  { (TfLiteIntArray*)&inputs0, (TfLiteIntArray*)&outputs0, const_cast<void*>(static_cast<const void*>(&opdata0)), OP_CONV_2D, },
  { (TfLiteIntArray*)&inputs1, (TfLiteIntArray*)&outputs1, const_cast<void*>(static_cast<const void*>(&opdata1)), OP_DEPTHWISE_CONV_2D, },
  { (TfLiteIntArray*)&inputs2, (TfLiteIntArray*)&outputs2, const_cast<void*>(static_cast<const void*>(&opdata2)), OP_CONV_2D, },
  { (TfLiteIntArray*)&inputs3, (TfLiteIntArray*)&outputs3, const_cast<void*>(static_cast<const void*>(&opdata3)), OP_DEPTHWISE_CONV_2D, },
  { (TfLiteIntArray*)&inputs4, (TfLiteIntArray*)&outputs4, const_cast<void*>(static_cast<const void*>(&opdata4)), OP_CONV_2D, },
  { (TfLiteIntArray*)&inputs5, (TfLiteIntArray*)&outputs5, const_cast<void*>(static_cast<const void*>(&opdata5)), OP_DEPTHWISE_CONV_2D, },
  { (TfLiteIntArray*)&inputs6, (TfLiteIntArray*)&outputs6, const_cast<void*>(static_cast<const void*>(&opdata6)), OP_CONV_2D, },
  { (TfLiteIntArray*)&inputs7, (TfLiteIntArray*)&outputs7, const_cast<void*>(static_cast<const void*>(&opdata7)), OP_DEPTHWISE_CONV_2D, },
  { (TfLiteIntArray*)&inputs8, (TfLiteIntArray*)&outputs8, const_cast<void*>(static_cast<const void*>(&opdata8)), OP_CONV_2D, },
  { (TfLiteIntArray*)&inputs9, (TfLiteIntArray*)&outputs9, const_cast<void*>(static_cast<const void*>(&opdata9)), OP_DEPTHWISE_CONV_2D, },
  { (TfLiteIntArray*)&inputs10, (TfLiteIntArray*)&outputs10, const_cast<void*>(static_cast<const void*>(&opdata10)), OP_CONV_2D, },
  { (TfLiteIntArray*)&inputs11, (TfLiteIntArray*)&outputs11, const_cast<void*>(static_cast<const void*>(&opdata11)), OP_DEPTHWISE_CONV_2D, },
  { (TfLiteIntArray*)&inputs12, (TfLiteIntArray*)&outputs12, const_cast<void*>(static_cast<const void*>(&opdata12)), OP_CONV_2D, },
  { (TfLiteIntArray*)&inputs13, (TfLiteIntArray*)&outputs13, const_cast<void*>(static_cast<const void*>(&opdata13)), OP_DEPTHWISE_CONV_2D, },
  { (TfLiteIntArray*)&inputs14, (TfLiteIntArray*)&outputs14, const_cast<void*>(static_cast<const void*>(&opdata14)), OP_CONV_2D, },
  { (TfLiteIntArray*)&inputs15, (TfLiteIntArray*)&outputs15, const_cast<void*>(static_cast<const void*>(&opdata15)), OP_DEPTHWISE_CONV_2D, },
  { (TfLiteIntArray*)&inputs16, (TfLiteIntArray*)&outputs16, const_cast<void*>(static_cast<const void*>(&opdata16)), OP_CONV_2D, },
  { (TfLiteIntArray*)&inputs17, (TfLiteIntArray*)&outputs17, const_cast<void*>(static_cast<const void*>(&opdata17)), OP_DEPTHWISE_CONV_2D, },
  { (TfLiteIntArray*)&inputs18, (TfLiteIntArray*)&outputs18, const_cast<void*>(static_cast<const void*>(&opdata18)), OP_CONV_2D, },
  { (TfLiteIntArray*)&inputs19, (TfLiteIntArray*)&outputs19, const_cast<void*>(static_cast<const void*>(&opdata19)), OP_DEPTHWISE_CONV_2D, },
  { (TfLiteIntArray*)&inputs20, (TfLiteIntArray*)&outputs20, const_cast<void*>(static_cast<const void*>(&opdata20)), OP_CONV_2D, },
  { (TfLiteIntArray*)&inputs21, (TfLiteIntArray*)&outputs21, const_cast<void*>(static_cast<const void*>(&opdata21)), OP_DEPTHWISE_CONV_2D, },
  { (TfLiteIntArray*)&inputs22, (TfLiteIntArray*)&outputs22, const_cast<void*>(static_cast<const void*>(&opdata22)), OP_CONV_2D, },
  { (TfLiteIntArray*)&inputs23, (TfLiteIntArray*)&outputs23, const_cast<void*>(static_cast<const void*>(&opdata23)), OP_DEPTHWISE_CONV_2D, },
  { (TfLiteIntArray*)&inputs24, (TfLiteIntArray*)&outputs24, const_cast<void*>(static_cast<const void*>(&opdata24)), OP_CONV_2D, },
  { (TfLiteIntArray*)&inputs25, (TfLiteIntArray*)&outputs25, const_cast<void*>(static_cast<const void*>(&opdata25)), OP_DEPTHWISE_CONV_2D, },
  { (TfLiteIntArray*)&inputs26, (TfLiteIntArray*)&outputs26, const_cast<void*>(static_cast<const void*>(&opdata26)), OP_CONV_2D, },
  { (TfLiteIntArray*)&inputs27, (TfLiteIntArray*)&outputs27, const_cast<void*>(static_cast<const void*>(&opdata27)), OP_AVERAGE_POOL_2D, },
  { (TfLiteIntArray*)&inputs28, (TfLiteIntArray*)&outputs28, const_cast<void*>(static_cast<const void*>(&opdata28)), OP_CONV_2D, },
  { (TfLiteIntArray*)&inputs29, (TfLiteIntArray*)&outputs29, const_cast<void*>(static_cast<const void*>(&opdata29)), OP_RESHAPE, },
  { (TfLiteIntArray*)&inputs30, (TfLiteIntArray*)&outputs30, const_cast<void*>(static_cast<const void*>(&opdata30)), OP_SOFTMAX, },
};
static TfLiteStatus AllocatePersistentBuffer(struct TfLiteContext* ignored,
                                                 size_t bytes, void **ptr) {
  static uint8_t *AllocPtr = tensor_arena + sizeof(tensor_arena);

  AllocPtr -= bytes;
  *ptr =  AllocPtr;
  return kTfLiteOk;
}
} // namespace

TfLiteStatus mobilenet_init() {
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
  registrations[OP_DEPTHWISE_CONV_2D] = tflite::ops::micro::Register_DEPTHWISE_CONV_2D();
  registrations[OP_AVERAGE_POOL_2D] = tflite::ops::micro::Register_AVERAGE_POOL_2D();
  registrations[OP_RESHAPE] = tflite::ops::micro::Register_RESHAPE();
  registrations[OP_SOFTMAX] = tflite::ops::micro::Register_SOFTMAX();

  for(size_t i = 0; i < 31; ++i) {
    tflNodes[i].inputs = nodeData[i].inputs;
    tflNodes[i].outputs = nodeData[i].outputs;
    tflNodes[i].builtin_data = nodeData[i].builtin_data;
    tflNodes[i].custom_initial_data = nullptr;
    tflNodes[i].custom_initial_data_size = 0;
    if (registrations[nodeData[i].used_op_index].init) {
      tflNodes[i].user_data = registrations[nodeData[i].used_op_index].init(&ctx, (const char*)tflNodes[i].builtin_data, 0);
    }
  }
  for(size_t i = 0; i < 31; ++i) {
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
  88, 
};
TfLiteTensor* mobilenet_input(int index) {
  return &ctx.tensors[inTensorIndices[index]];
}

static const int outTensorIndices[] = {
  87, 
};
TfLiteTensor* mobilenet_output(int index) {
  return &ctx.tensors[outTensorIndices[index]];
}

TfLiteStatus mobilenet_invoke() {
  for(size_t i = 0; i < 31; ++i) {
    TfLiteStatus status = registrations[nodeData[i].used_op_index].invoke(&ctx, &tflNodes[i]);
    if (status != kTfLiteOk) {
      return status;
    }
  }
  return kTfLiteOk;
}
