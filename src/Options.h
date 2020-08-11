#ifndef TFLMCOMPILER_OPTIONS_H
#define TFLMCOMPILER_OPTIONS_H

#include <string>

namespace tflmc {
class Options
{
private:
    Options() {}
public:
    bool verbose = false;
    std::string  memmap_json;

    static Options &instance() {
        static Options options;
        return options;
    }
};

}

#endif // TFLMCOMPILER_OPTIONS_H
