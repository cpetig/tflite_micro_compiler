#include <fstream>
#include <iomanip>
#include <iostream>
#include <vector>

#include "out.cpp.h"

int main(int argc, char *argv[]) {
  if (argc != 2) {
    std::cerr << "Usage: " << argv[0] << " inDataFile\n";
    return 1;
  }

  if (model_inputs() != 1 || model_outputs() != 1) {
    std::cerr << "Mismatch for number of inputs/outputs\n";
    return 1;
  }

  std::ifstream inFile(argv[1], std::ios::binary);

  model_init();

  std::vector<char> inData(model_input_size(0));
  if (!inFile.read((char *)model_input_ptr(0), model_input_size(0))) {
    std::cerr << "Failed to read input file\n";
    return 1;
  }

  model_invoke();
  for (size_t i = 0; i < model_output_size(0); i++) {
    std::cout << "\\x" << std::setw(2) << std::setfill('0') << std::hex
              << (int)((unsigned char *)model_output_ptr(0))[i];
  }
  std::cout << std::endl;

  return 0;
}
