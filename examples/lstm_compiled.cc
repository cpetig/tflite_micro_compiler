// This file is generated. Do not edit.
// Generated on: 12.08.2020 18:54:29

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

constexpr int kTensorArenaSize = 984;
uint8_t tensor_arena[kTensorArenaSize] ALIGN(16);
template <int SZ, class T> struct TfArray {
  int sz; T elem[SZ];
};
enum used_operators_e {
  OP_FULLY_CONNECTED, OP_UNPACK, OP_ADD, OP_SPLIT, OP_MUL, OP_MINIMUM, OP_MAXIMUM,  OP_LAST
};
struct TensorInfo_t { // subset of TfLiteTensor used for initialization from constant memory
  TfLiteType type;
  void* data;
  TfLiteIntArray* dims;
  size_t bytes;
};
struct NodeInfo_t { // subset of TfLiteNode used for initialization from constant memory
  struct TfLiteIntArray* inputs;
  struct TfLiteIntArray* outputs;
  void* builtin_data;
  used_operators_e used_op_index;
};

TfLiteContext ctx{};
TfLiteTensor tflTensors[48];
TfLiteEvalTensor evalTensors[48];
TfLiteRegistration registrations[OP_LAST];
TfLiteNode tflNodes[23];

const TfArray<3, int> tensor_dimension0 = { 3, { 1,1,1 } };
const TfArray<2, int> tensor_dimension1 = { 2, { 1,1 } };
const ALIGN(4) float tensor_data2[1] = { -0.10025534778833389, };
const TfArray<1, int> tensor_dimension2 = { 1, { 1 } };
const ALIGN(8) float tensor_data3[1*6] = { 
  -0.67682290077209473, 0.54124373197555542, 1.032249927520752, -0.26129794120788574, -0.64947265386581421, 0.76698356866836548, 
};
const TfArray<2, int> tensor_dimension3 = { 2, { 1,6 } };
const TfArray<2, int> tensor_dimension4 = { 2, { 1,6 } };
const TfArray<2, int> tensor_dimension5 = { 2, { 1,6 } };
const TfArray<2, int> tensor_dimension6 = { 2, { 1,6 } };
const TfArray<2, int> tensor_dimension7 = { 2, { 1,24 } };
const float tensor_data8[1] = { 0.20000000298023224 };
const int tensor_dimension8 = 0; /* empty TfLiteIntArray */
const float tensor_data9[1] = { 0.5 };
const int tensor_dimension9 = 0; /* empty TfLiteIntArray */
const float tensor_data10[1] = { 0.20000000298023224 };
const int tensor_dimension10 = 0; /* empty TfLiteIntArray */
const float tensor_data11[1] = { 0.5 };
const int tensor_dimension11 = 0; /* empty TfLiteIntArray */
const float tensor_data12[1] = { 0.20000000298023224 };
const int tensor_dimension12 = 0; /* empty TfLiteIntArray */
const float tensor_data13[1] = { 0.5 };
const int tensor_dimension13 = 0; /* empty TfLiteIntArray */
const TfArray<2, int> tensor_dimension14 = { 2, { 1,24 } };
const TfArray<2, int> tensor_dimension15 = { 2, { 1,24 } };
const ALIGN(8) float tensor_data16[24] = { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, };
const TfArray<1, int> tensor_dimension16 = { 1, { 24 } };
const TfArray<2, int> tensor_dimension17 = { 2, { 1,6 } };
const TfArray<2, int> tensor_dimension18 = { 2, { 1,6 } };
const TfArray<2, int> tensor_dimension19 = { 2, { 1,6 } };
const TfArray<2, int> tensor_dimension20 = { 2, { 1,24 } };
const TfArray<2, int> tensor_dimension21 = { 2, { 1,6 } };
const ALIGN(8) float tensor_data22[24] = { 0.81149637699127197, 0.68789982795715332, 0.50034081935882568, -0.33782359957695007, 0.46354654431343079, -0.10486250370740891, 1, 1, 1, 1, 1, 1, 0.43911349773406982, 0.28772261738777161, 0.31830844283103943, -0.062725074589252472, -0.36620119214057922, -0.23563581705093384, 0.81197762489318848, 0.7107347846031189, 0.50316768884658813, -0.33033576607704163, 0.47111541032791138, -0.096191838383674622, };
const TfArray<1, int> tensor_dimension22 = { 1, { 24 } };
const TfArray<2, int> tensor_dimension23 = { 2, { 1,6 } };
const TfArray<2, int> tensor_dimension24 = { 2, { 1,6 } };
const float tensor_data25[1] = { 1 };
const int tensor_dimension25 = 0; /* empty TfLiteIntArray */
const float tensor_data26[1] = { 0 };
const int tensor_dimension26 = 0; /* empty TfLiteIntArray */
const TfArray<2, int> tensor_dimension27 = { 2, { 1,6 } };
const TfArray<2, int> tensor_dimension28 = { 2, { 1,6 } };
const float tensor_data29[1] = { 1 };
const int tensor_dimension29 = 0; /* empty TfLiteIntArray */
const float tensor_data30[1] = { 0 };
const int tensor_dimension30 = 0; /* empty TfLiteIntArray */
const TfArray<2, int> tensor_dimension31 = { 2, { 1,6 } };
const TfArray<2, int> tensor_dimension32 = { 2, { 1,6 } };
const float tensor_data33[1] = { 1 };
const int tensor_dimension33 = 0; /* empty TfLiteIntArray */
const float tensor_data34[1] = { 0 };
const int tensor_dimension34 = 0; /* empty TfLiteIntArray */
const ALIGN(8) float tensor_data35[24*1] = { 
  -0.78892803192138672, 
  -0.066408462822437286, 
  0.35161417722702026, 
  0.82873892784118652, 
  0.80997800827026367, 
  -0.64194154739379883, 
  0.48032155632972717, 
  0.26661530137062073, 
  -0.12920093536376953, 
  -0.046929717063903809, 
  -0.45629832148551941, 
  -0.15663215517997742, 
  -0.85566461086273193, 
  0.7620043158531189, 
  0.75519061088562012, 
  0.14026883244514465, 
  -0.34299778938293457, 
  0.1274896115064621, 
  -0.77789837121963501, 
  0.52178329229354858, 
  0.52061921358108521, 
  0.43844866752624512, 
  0.42333868145942688, 
  -0.5325203537940979, 
};
const TfArray<2, int> tensor_dimension35 = { 2, { 24,1 } };
const TfArray<2, int> tensor_dimension36 = { 2, { 1,6 } };
const TfArray<2, int> tensor_dimension37 = { 2, { 1,6 } };
const TfArray<2, int> tensor_dimension38 = { 2, { 1,6 } };
const ALIGN(8) float tensor_data39[24*6] = { 
  -0.022579669952392578, 0.043456412851810455, 0.30239257216453552, -0.20567117631435394, -0.074365966022014618, -0.092949599027633667, 
  0.00019303386216051877, -0.0035804288927465677, 0.17888210713863373, 0.13906814157962799, -0.13194261491298676, -0.29369610548019409, 
  -0.08762822300195694, 0.0013085408136248589, -0.40365827083587646, -0.022184055298566818, 0.014901737682521343, -0.21936503052711487, 
  0.16428762674331665, 0.072828143835067749, 0.2114323228597641, -0.25273096561431885, -0.21754516661167145, 0.097414277493953705, 
  0.38464519381523132, 0.18493148684501648, 0.20381182432174683, -0.14378896355628967, -0.20844249427318573, 0.18730002641677856, 
  0.14259083569049835, -0.047009404748678207, 0.14904215931892395, -0.060763522982597351, -0.16218960285186768, 0.13128620386123657, 
  -0.1546204537153244, -0.059371381998062134, 0.19501221179962158, -0.10580887645483017, -0.23217323422431946, -0.22936071455478668, 
  -0.0021029131021350622, 0.064688146114349365, 0.03878474235534668, 0.22142297029495239, -0.16029603779315948, -0.3753601610660553, 
  -0.32715627551078796, 0.13923661410808563, -0.070977889001369476, -0.032542452216148376, 0.014878313988447189, 0.40545779466629028, 
  -0.060356121510267258, 0.25173348188400269, 0.11893317848443985, 0.1443747878074646, 0.0079025859013199806, 0.01033000648021698, 
  0.3641963005065918, -0.50888949632644653, -0.039992153644561768, -0.092163838446140289, -0.061423711478710175, 0.14505985379219055, 
  -0.15009048581123352, 0.086118385195732117, 0.33083158731460571, -0.42140620946884155, 0.48624292016029358, 0.039929948747158051, 
  0.090693637728691101, -0.01283695176243782, 0.17024977505207062, -0.22920811176300049, 0.15299870073795319, -0.31904631853103638, 
  0.12285808473825455, 0.065751530230045319, 0.091771945357322693, 0.11162638664245605, -0.073564663529396057, -0.065775707364082336, 
  0.33217611908912659, 0.41962692141532898, -0.10504135489463806, -0.07918558269739151, 0.023805687204003334, -0.062702901661396027, 
  -0.055794805288314819, 0.1239912211894989, 0.1370149701833725, 0.46396678686141968, -0.064550302922725677, -0.02640213631093502, 
  -0.073347911238670349, -0.059045586735010147, -0.11495655030012131, 0.040152005851268768, -0.12168806046247482, 0.095367684960365295, 
  0.074778921902179718, 0.056080017238855362, 0.13821674883365631, -0.1471501886844635, -0.23102031648159027, -0.20438291132450104, 
  -0.14502057433128357, 0.1057831272482872, -0.14780488610267639, -0.16759186983108521, 0.24482570588588715, -0.38713780045509338, 
  -0.37223842740058899, 0.11535821110010147, -0.16456326842308044, -0.30136668682098389, -0.45508956909179688, 0.17768050730228424, 
  -0.15122713148593903, -0.48300850391387939, -0.018385179340839386, -0.16502243280410767, -0.25257453322410583, -0.223208948969841, 
  0.18494662642478943, 0.31131488084793091, -0.44953763484954834, -0.28992098569869995, -0.20313996076583862, -0.14405348896980286, 
  -0.21683275699615479, -0.066648311913013458, 0.00025516003370285034, -0.18808770179748535, 0.07135278731584549, 0.017622401937842369, 
  0.30879741907119751, -0.19441895186901093, -0.29071500897407532, -0.077798321843147278, 0.24645853042602539, 0.024298662319779396, 
};
const TfArray<2, int> tensor_dimension39 = { 2, { 24,6 } };
const TfArray<2, int> tensor_dimension40 = { 2, { 1,6 } };
const int32_t tensor_data41[1] = { 1 };
const int tensor_dimension41 = 0; /* empty TfLiteIntArray */
const TfArray<2, int> tensor_dimension42 = { 2, { 1,6 } };
const TfArray<2, int> tensor_dimension43 = { 2, { 1,6 } };
const TfArray<2, int> tensor_dimension44 = { 2, { 1,6 } };
const TfArray<2, int> tensor_dimension45 = { 2, { 1,1 } };
const TfArray<2, int> tensor_dimension46 = { 2, { 1,6 } };
const TfArray<2, int> tensor_dimension47 = { 2, { 1,6 } };
const TfLiteFullyConnectedParams opdata0 = { kTfLiteActNone, kTfLiteFullyConnectedWeightsFormatDefault, false, false };
const TfArray<3, int> inputs0 = { 3, { 47,39,16 } };
const TfArray<1, int> outputs0 = { 1, { 15 } };
const ALIGN(1) uint8_t opdata1[0] = {  }; /* op type 88=UNPACK */
const TfArray<1, int> inputs1 = { 1, { 0 } };
const TfArray<1, int> outputs1 = { 1, { 45 } };
const TfLiteFullyConnectedParams opdata2 = { kTfLiteActNone, kTfLiteFullyConnectedWeightsFormatDefault, false, false };
const TfArray<3, int> inputs2 = { 3, { 45,35,16 } };
const TfArray<1, int> outputs2 = { 1, { 14 } };
const TfLiteAddParams opdata3 = { kTfLiteActNone };
const TfArray<2, int> inputs3 = { 2, { 14,15 } };
const TfArray<1, int> outputs3 = { 1, { 20 } };
const TfLiteAddParams opdata4 = { kTfLiteActNone };
const TfArray<2, int> inputs4 = { 2, { 20,22 } };
const TfArray<1, int> outputs4 = { 1, { 7 } };
const ALIGN(1) uint8_t opdata5[0] = {  }; /* op type 49=SPLIT */
const TfArray<2, int> inputs5 = { 2, { 41,7 } };
const TfArray<4, int> outputs5 = { 4, { 40,42,43,44 } };
const TfLiteMulParams opdata6 = { kTfLiteActNone };
const TfArray<2, int> inputs6 = { 2, { 40,8 } };
const TfArray<1, int> outputs6 = { 1, { 17 } };
const TfLiteMulParams opdata7 = { kTfLiteActNone };
const TfArray<2, int> inputs7 = { 2, { 42,10 } };
const TfArray<1, int> outputs7 = { 1, { 18 } };
const TfLiteMulParams opdata8 = { kTfLiteActNone };
const TfArray<2, int> inputs8 = { 2, { 44,12 } };
const TfArray<1, int> outputs8 = { 1, { 19 } };
const TfLiteAddParams opdata9 = { kTfLiteActNone };
const TfArray<2, int> inputs9 = { 2, { 17,9 } };
const TfArray<1, int> outputs9 = { 1, { 4 } };
const TfLiteAddParams opdata10 = { kTfLiteActNone };
const TfArray<2, int> inputs10 = { 2, { 18,11 } };
const TfArray<1, int> outputs10 = { 1, { 5 } };
const TfLiteAddParams opdata11 = { kTfLiteActNone };
const TfArray<2, int> inputs11 = { 2, { 19,13 } };
const TfArray<1, int> outputs11 = { 1, { 6 } };
const TfArray<2, int> inputs12 = { 2, { 4,25 } };
const TfArray<1, int> outputs12 = { 1, { 24 } };
const TfArray<2, int> inputs13 = { 2, { 5,29 } };
const TfArray<1, int> outputs13 = { 1, { 28 } };
const TfArray<2, int> inputs14 = { 2, { 6,33 } };
const TfArray<1, int> outputs14 = { 1, { 32 } };
const TfArray<2, int> inputs15 = { 2, { 24,26 } };
const TfArray<1, int> outputs15 = { 1, { 23 } };
const TfArray<2, int> inputs16 = { 2, { 28,30 } };
const TfArray<1, int> outputs16 = { 1, { 27 } };
const TfArray<2, int> inputs17 = { 2, { 32,34 } };
const TfArray<1, int> outputs17 = { 1, { 31 } };
const TfLiteMulParams opdata18 = { kTfLiteActNone };
const TfArray<2, int> inputs18 = { 2, { 23,43 } };
const TfArray<1, int> outputs18 = { 1, { 37 } };
const TfLiteMulParams opdata19 = { kTfLiteActNone };
const TfArray<2, int> inputs19 = { 2, { 27,46 } };
const TfArray<1, int> outputs19 = { 1, { 36 } };
const TfLiteAddParams opdata20 = { kTfLiteActNone };
const TfArray<2, int> inputs20 = { 2, { 36,37 } };
const TfArray<1, int> outputs20 = { 1, { 21 } };
const TfLiteMulParams opdata21 = { kTfLiteActNone };
const TfArray<2, int> inputs21 = { 2, { 31,21 } };
const TfArray<1, int> outputs21 = { 1, { 38 } };
const TfLiteFullyConnectedParams opdata22 = { kTfLiteActNone, kTfLiteFullyConnectedWeightsFormatDefault, false, false };
const TfArray<3, int> inputs22 = { 3, { 38,3,2 } };
const TfArray<1, int> outputs22 = { 1, { 1 } };
const TensorInfo_t tensorData[] = {
  { kTfLiteFloat32, tensor_arena + 32, (TfLiteIntArray*)&tensor_dimension0, 4, },
  { kTfLiteFloat32, tensor_arena + 32, (TfLiteIntArray*)&tensor_dimension1, 4, },
  { kTfLiteFloat32, (void*)tensor_data2, (TfLiteIntArray*)&tensor_dimension2, 4, },
  { kTfLiteFloat32, (void*)tensor_data3, (TfLiteIntArray*)&tensor_dimension3, 24, },
  { kTfLiteFloat32, tensor_arena + 160, (TfLiteIntArray*)&tensor_dimension4, 24, },
  { kTfLiteFloat32, tensor_arena + 0, (TfLiteIntArray*)&tensor_dimension5, 24, },
  { kTfLiteFloat32, tensor_arena + 128, (TfLiteIntArray*)&tensor_dimension6, 24, },
  { kTfLiteFloat32, tensor_arena + 96, (TfLiteIntArray*)&tensor_dimension7, 96, },
  { kTfLiteFloat32, (void*)tensor_data8, (TfLiteIntArray*)&tensor_dimension8, 4, },
  { kTfLiteFloat32, (void*)tensor_data9, (TfLiteIntArray*)&tensor_dimension9, 4, },
  { kTfLiteFloat32, (void*)tensor_data10, (TfLiteIntArray*)&tensor_dimension10, 4, },
  { kTfLiteFloat32, (void*)tensor_data11, (TfLiteIntArray*)&tensor_dimension11, 4, },
  { kTfLiteFloat32, (void*)tensor_data12, (TfLiteIntArray*)&tensor_dimension12, 4, },
  { kTfLiteFloat32, (void*)tensor_data13, (TfLiteIntArray*)&tensor_dimension13, 4, },
  { kTfLiteFloat32, tensor_arena + 192, (TfLiteIntArray*)&tensor_dimension14, 96, },
  { kTfLiteFloat32, tensor_arena + 96, (TfLiteIntArray*)&tensor_dimension15, 96, },
  { kTfLiteFloat32, (void*)tensor_data16, (TfLiteIntArray*)&tensor_dimension16, 96, },
  { kTfLiteFloat32, tensor_arena + 128, (TfLiteIntArray*)&tensor_dimension17, 24, },
  { kTfLiteFloat32, tensor_arena + 96, (TfLiteIntArray*)&tensor_dimension18, 24, },
  { kTfLiteFloat32, tensor_arena + 64, (TfLiteIntArray*)&tensor_dimension19, 24, },
  { kTfLiteFloat32, tensor_arena + 0, (TfLiteIntArray*)&tensor_dimension20, 96, },
  { kTfLiteFloat32, tensor_arena + 96, (TfLiteIntArray*)&tensor_dimension21, 24, },
  { kTfLiteFloat32, (void*)tensor_data22, (TfLiteIntArray*)&tensor_dimension22, 96, },
  { kTfLiteFloat32, tensor_arena + 128, (TfLiteIntArray*)&tensor_dimension23, 24, },
  { kTfLiteFloat32, tensor_arena + 96, (TfLiteIntArray*)&tensor_dimension24, 24, },
  { kTfLiteFloat32, (void*)tensor_data25, (TfLiteIntArray*)&tensor_dimension25, 4, },
  { kTfLiteFloat32, (void*)tensor_data26, (TfLiteIntArray*)&tensor_dimension26, 4, },
  { kTfLiteFloat32, tensor_arena + 96, (TfLiteIntArray*)&tensor_dimension27, 24, },
  { kTfLiteFloat32, tensor_arena + 64, (TfLiteIntArray*)&tensor_dimension28, 24, },
  { kTfLiteFloat32, (void*)tensor_data29, (TfLiteIntArray*)&tensor_dimension29, 4, },
  { kTfLiteFloat32, (void*)tensor_data30, (TfLiteIntArray*)&tensor_dimension30, 4, },
  { kTfLiteFloat32, tensor_arena + 64, (TfLiteIntArray*)&tensor_dimension31, 24, },
  { kTfLiteFloat32, tensor_arena + 0, (TfLiteIntArray*)&tensor_dimension32, 24, },
  { kTfLiteFloat32, (void*)tensor_data33, (TfLiteIntArray*)&tensor_dimension33, 4, },
  { kTfLiteFloat32, (void*)tensor_data34, (TfLiteIntArray*)&tensor_dimension34, 4, },
  { kTfLiteFloat32, (void*)tensor_data35, (TfLiteIntArray*)&tensor_dimension35, 96, },
  { kTfLiteFloat32, tensor_arena + 32, (TfLiteIntArray*)&tensor_dimension36, 24, },
  { kTfLiteFloat32, tensor_arena + 0, (TfLiteIntArray*)&tensor_dimension37, 24, },
  { kTfLiteFloat32, tensor_arena + 0, (TfLiteIntArray*)&tensor_dimension38, 24, },
  { kTfLiteFloat32, (void*)tensor_data39, (TfLiteIntArray*)&tensor_dimension39, 576, },
  { kTfLiteFloat32, tensor_arena + 192, (TfLiteIntArray*)&tensor_dimension40, 24, },
  { kTfLiteInt32, (void*)tensor_data41, (TfLiteIntArray*)&tensor_dimension41, 4, },
  { kTfLiteFloat32, tensor_arena + 64, (TfLiteIntArray*)&tensor_dimension42, 24, },
  { kTfLiteFloat32, tensor_arena + 32, (TfLiteIntArray*)&tensor_dimension43, 24, },
  { kTfLiteFloat32, tensor_arena + 0, (TfLiteIntArray*)&tensor_dimension44, 24, },
  { kTfLiteFloat32, tensor_arena + 0, (TfLiteIntArray*)&tensor_dimension45, 4, },
  { kTfLiteFloat32, tensor_arena + 288, (TfLiteIntArray*)&tensor_dimension46, 24, },
  { kTfLiteFloat32, tensor_arena + 0, (TfLiteIntArray*)&tensor_dimension47, 24, },
};const NodeInfo_t nodeData[] = {
  { (TfLiteIntArray*)&inputs0, (TfLiteIntArray*)&outputs0, const_cast<void*>(static_cast<const void*>(&opdata0)), OP_FULLY_CONNECTED, },
  { (TfLiteIntArray*)&inputs1, (TfLiteIntArray*)&outputs1, const_cast<void*>(static_cast<const void*>(&opdata1)), OP_UNPACK, },
  { (TfLiteIntArray*)&inputs2, (TfLiteIntArray*)&outputs2, const_cast<void*>(static_cast<const void*>(&opdata2)), OP_FULLY_CONNECTED, },
  { (TfLiteIntArray*)&inputs3, (TfLiteIntArray*)&outputs3, const_cast<void*>(static_cast<const void*>(&opdata3)), OP_ADD, },
  { (TfLiteIntArray*)&inputs4, (TfLiteIntArray*)&outputs4, const_cast<void*>(static_cast<const void*>(&opdata4)), OP_ADD, },
  { (TfLiteIntArray*)&inputs5, (TfLiteIntArray*)&outputs5, const_cast<void*>(static_cast<const void*>(&opdata5)), OP_SPLIT, },
  { (TfLiteIntArray*)&inputs6, (TfLiteIntArray*)&outputs6, const_cast<void*>(static_cast<const void*>(&opdata6)), OP_MUL, },
  { (TfLiteIntArray*)&inputs7, (TfLiteIntArray*)&outputs7, const_cast<void*>(static_cast<const void*>(&opdata7)), OP_MUL, },
  { (TfLiteIntArray*)&inputs8, (TfLiteIntArray*)&outputs8, const_cast<void*>(static_cast<const void*>(&opdata8)), OP_MUL, },
  { (TfLiteIntArray*)&inputs9, (TfLiteIntArray*)&outputs9, const_cast<void*>(static_cast<const void*>(&opdata9)), OP_ADD, },
  { (TfLiteIntArray*)&inputs10, (TfLiteIntArray*)&outputs10, const_cast<void*>(static_cast<const void*>(&opdata10)), OP_ADD, },
  { (TfLiteIntArray*)&inputs11, (TfLiteIntArray*)&outputs11, const_cast<void*>(static_cast<const void*>(&opdata11)), OP_ADD, },
  { (TfLiteIntArray*)&inputs12, (TfLiteIntArray*)&outputs12, nullptr, OP_MINIMUM, },
  { (TfLiteIntArray*)&inputs13, (TfLiteIntArray*)&outputs13, nullptr, OP_MINIMUM, },
  { (TfLiteIntArray*)&inputs14, (TfLiteIntArray*)&outputs14, nullptr, OP_MINIMUM, },
  { (TfLiteIntArray*)&inputs15, (TfLiteIntArray*)&outputs15, nullptr, OP_MAXIMUM, },
  { (TfLiteIntArray*)&inputs16, (TfLiteIntArray*)&outputs16, nullptr, OP_MAXIMUM, },
  { (TfLiteIntArray*)&inputs17, (TfLiteIntArray*)&outputs17, nullptr, OP_MAXIMUM, },
  { (TfLiteIntArray*)&inputs18, (TfLiteIntArray*)&outputs18, const_cast<void*>(static_cast<const void*>(&opdata18)), OP_MUL, },
  { (TfLiteIntArray*)&inputs19, (TfLiteIntArray*)&outputs19, const_cast<void*>(static_cast<const void*>(&opdata19)), OP_MUL, },
  { (TfLiteIntArray*)&inputs20, (TfLiteIntArray*)&outputs20, const_cast<void*>(static_cast<const void*>(&opdata20)), OP_ADD, },
  { (TfLiteIntArray*)&inputs21, (TfLiteIntArray*)&outputs21, const_cast<void*>(static_cast<const void*>(&opdata21)), OP_MUL, },
  { (TfLiteIntArray*)&inputs22, (TfLiteIntArray*)&outputs22, const_cast<void*>(static_cast<const void*>(&opdata22)), OP_FULLY_CONNECTED, },
};
static void* AllocatePersistentBuffer(struct TfLiteContext* ctx,
                                                 size_t bytes) {
  static uint8_t *AllocPtr = tensor_arena + sizeof(tensor_arena);

  AllocPtr -= bytes;
  return AllocPtr;
}

static TfLiteEvalTensor *GetEvalTensor(const struct TfLiteContext *context,
                                       int tensor_idx) {
  return &evalTensors[tensor_idx];
}
} // namespace

TfLiteStatus lstm_init() {
  ctx.AllocatePersistentBuffer = &AllocatePersistentBuffer;
  ctx.GetEvalTensor = &GetEvalTensor;
  ctx.tensors = tflTensors;
  ctx.tensors_size = 48;
  for(size_t i = 0; i < 48; ++i) {
    tflTensors[i].data.data = tensorData[i].data;
    evalTensors[i].data.data = tensorData[i].data;
    tflTensors[i].type = tensorData[i].type;
    evalTensors[i].type = tensorData[i].type;
    tflTensors[i].is_variable = 0;
    tflTensors[i].allocation_type = (tensor_arena <= tensorData[i].data && tensorData[i].data < tensor_arena + kTensorArenaSize) ? kTfLiteArenaRw : kTfLiteMmapRo;
    tflTensors[i].bytes = tensorData[i].bytes;
    tflTensors[i].dims = tensorData[i].dims;
    evalTensors[i].dims = tensorData[i].dims;
    tflTensors[i].quantization.type = kTfLiteNoQuantization;
  }
  registrations[OP_FULLY_CONNECTED] = tflite::ops::micro::Register_FULLY_CONNECTED();
  registrations[OP_UNPACK] = tflite::ops::micro::Register_UNPACK();
  registrations[OP_ADD] = tflite::ops::micro::Register_ADD();
  registrations[OP_SPLIT] = tflite::ops::micro::Register_SPLIT();
  registrations[OP_MUL] = tflite::ops::micro::Register_MUL();
  registrations[OP_MINIMUM] = tflite::ops::micro::Register_MINIMUM();
  registrations[OP_MAXIMUM] = tflite::ops::micro::Register_MAXIMUM();

  for(size_t i = 0; i < 23; ++i) {
    tflNodes[i].inputs = nodeData[i].inputs;
    tflNodes[i].outputs = nodeData[i].outputs;
    tflNodes[i].builtin_data = nodeData[i].builtin_data;
    tflNodes[i].custom_initial_data = nullptr;
    tflNodes[i].custom_initial_data_size = 0;
    if (registrations[nodeData[i].used_op_index].init) {
      tflNodes[i].user_data = registrations[nodeData[i].used_op_index].init(&ctx, (const char*)tflNodes[i].builtin_data, 0);
    }
  }
  for(size_t i = 0; i < 23; ++i) {
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
  0, 47, 46, 
};
TfLiteTensor* lstm_input(int index) {
  return &ctx.tensors[inTensorIndices[index]];
}

static const int outTensorIndices[] = {
  1, 38, 21, 
};
TfLiteTensor* lstm_output(int index) {
  return &ctx.tensors[outTensorIndices[index]];
}

TfLiteStatus lstm_invoke() {
  for(size_t i = 0; i < 23; ++i) {
    TfLiteStatus status = registrations[nodeData[i].used_op_index].invoke(&ctx, &tflNodes[i]);
    if (status != kTfLiteOk) {
      return status;
    }
  }
  return kTfLiteOk;
}
