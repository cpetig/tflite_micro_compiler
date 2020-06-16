#include "CodeWriter.h"
#include "Compiler.h"

int main(int argc, char *argv[]) {
  if (argc < 3 || argc > 4) {
    printf(
        "Usage: %s modelFile.tflite outFile.cpp [NamingPrefix = \"model_\"]\n",
        argv[0]);
    return 1;
  }

  std::string prefix = "model_";
  if (argc == 4) {
    prefix = argv[3];
  }

  if (!tflmc::CompileFile(argv[1], argv[2], prefix)) {
    return 1;
  }

  return 0;
}
