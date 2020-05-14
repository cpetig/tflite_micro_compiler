
#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/micro/kernels/all_ops_resolver.h"

namespace tflite { namespace ops { namespace micro {
    namespace reduce_max { extern TfLiteStatus Eval(TfLiteContext*, TfLiteNode*) { return kTfLiteOk; } }
    TfLiteRegistration *Register_REDUCE_MAX(void) {
        static TfLiteRegistration res = {
            nullptr, nullptr, nullptr, reduce_max::Eval,
        };
        return &res;
    }
    namespace exp { extern TfLiteStatus Eval(TfLiteContext*, TfLiteNode*) { return kTfLiteOk; } }
    TfLiteRegistration *Register_EXP(void) {
        static TfLiteRegistration res = {
            nullptr, nullptr, nullptr, exp::Eval,
        };
        return &res;
    }
    namespace sum { extern TfLiteStatus Eval(TfLiteContext*, TfLiteNode*) { return kTfLiteOk; } }
    TfLiteRegistration *Register_SUM(void) {
        static TfLiteRegistration res = {
            nullptr, nullptr, nullptr, sum::Eval,
        };
        return &res;
    }
    namespace div { extern TfLiteStatus Eval(TfLiteContext*, TfLiteNode*) { return kTfLiteOk; } }
    TfLiteRegistration *Register_DIV(void) {
        static TfLiteRegistration res = {
            nullptr, nullptr, nullptr, div::Eval,
        };
        return &res;
    }
    namespace squeeze { extern TfLiteStatus Eval(TfLiteContext*, TfLiteNode*) { return kTfLiteOk; } }
    TfLiteRegistration *Register_SQUEEZE(void) {
        static TfLiteRegistration res = {
            nullptr, nullptr, nullptr, squeeze::Eval,
        };
        return &res;
    }
}}}

void register_addons(tflite::ops::micro::AllOpsResolver *res) {
    res->AddBuiltin(tflite::BuiltinOperator_REDUCE_MAX, tflite::ops::micro::Register_REDUCE_MAX());
    res->AddBuiltin(tflite::BuiltinOperator_EXP, tflite::ops::micro::Register_EXP());
    res->AddBuiltin(tflite::BuiltinOperator_SUM, tflite::ops::micro::Register_SUM());
    res->AddBuiltin(tflite::BuiltinOperator_DIV, tflite::ops::micro::Register_DIV());
    res->AddBuiltin(tflite::BuiltinOperator_SQUEEZE, tflite::ops::micro::Register_SQUEEZE());
}
