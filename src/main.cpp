#include "CodeWriter.h"
#include "Compiler.h"

int main(int argc, char *argv[]) {
  if (argc != 3) {
    printf("Usage: %s modelFile.tflite outFile.cpp\n", argv[0]);
    return 1;
  }

  if (!tflmc::CompileFile(argv[1], argv[2])) {
    return 1;
  }

  return 0;
}
