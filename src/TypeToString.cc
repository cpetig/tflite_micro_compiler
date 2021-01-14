#include "TypeToString.h"

#include <sstream>
#include <stdexcept>

#define NAME(X) \
  case X:       \
    return #X

std::string tflmc::to_string(TfLiteType t) {
  switch (t) {
    NAME(kTfLiteFloat32);
    NAME(kTfLiteInt32);
    NAME(kTfLiteUInt8);
    NAME(kTfLiteInt64);
    NAME(kTfLiteString);
    NAME(kTfLiteBool);
    NAME(kTfLiteInt16);
    NAME(kTfLiteComplex64);
    NAME(kTfLiteInt8);
    NAME(kTfLiteFloat16);
    NAME(kTfLiteFloat64);
    default:
      throw std::runtime_error(
          "Missing case in TfLiteType to string conversion");
  }
}

std::string tflmc::c_type(TfLiteType t) {
  switch (t) {
    case kTfLiteFloat32:
      return "float";
    case kTfLiteInt32:
      return "int32_t";
    case kTfLiteUInt8:
      return "uint8_t";
    case kTfLiteInt64:
      return "int64_t";
    // case kTfLiteString: return "float";
    // case kTfLiteBool: return "float";
    case kTfLiteInt16:
      return "int16_t";
    // case kTfLiteComplex64: return "float";
    case kTfLiteInt8:
      return "int8_t";
    // case kTfLiteFloat16: return "float";
    case kTfLiteFloat64:
      return "double";
    default:
      throw std::runtime_error(
          "Missing case in TfLiteType to C type conversion");
  }
}

std::string tflmc::to_string(TfLiteAllocationType t) {
  switch (t) {
    NAME(kTfLiteMmapRo);
    NAME(kTfLiteArenaRw);
    default:
      throw std::runtime_error(
          "Missing case in TfLiteAllocationType to string "
          "conversion");
  }
}

std::string tflmc::to_string(TfLiteFusedActivation t) {
  switch (t) {
    NAME(kTfLiteActNone);
    NAME(kTfLiteActRelu);
    NAME(kTfLiteActReluN1To1);
    NAME(kTfLiteActRelu6);
    NAME(kTfLiteActTanh);
    NAME(kTfLiteActSignBit);
    NAME(kTfLiteActSigmoid);
    default:
      throw std::runtime_error(
          "Missing case in TfLiteFusedActivation to string conversion");
  }
}

std::string tflmc::to_string(TfLiteFullyConnectedWeightsFormat t) {
  switch (t) {
    NAME(kTfLiteFullyConnectedWeightsFormatDefault);
    NAME(kTfLiteFullyConnectedWeightsFormatShuffled4x16Int8);
    default:
      throw std::runtime_error(
          "Missing case in TfLiteFullyConnectedWeightsFormat to string "
          "conversion");
  }
}

std::string tflmc::to_string(TfLitePadding t) {
  switch (t) {
    NAME(kTfLitePaddingUnknown);
    NAME(kTfLitePaddingSame);
    NAME(kTfLitePaddingValid);
    default:
      throw std::runtime_error(
          "Missing case in TfLitePadding to string conversion");
  }
}

std::string tflmc::to_string(TfLitePaddingValues const& v) {
  std::stringstream out;
  out << "{ " << v.width << "," << v.height << ", " << v.width_offset << ", "
      << v.height_offset << " }";
  return out.str();
}
