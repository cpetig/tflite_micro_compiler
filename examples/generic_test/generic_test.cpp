#include <fstream>
#include <iomanip>
#include <iostream>
#include <vector>

extern void model_init();
extern void *model_input_ptr();
extern const void *model_output_ptr();
extern size_t model_output_size();
extern void model_invoke();

int main(int argc, char *argv[]) {
  if (argc != 2) {
    std::cerr << "Usage: " << argv[0] << " inDataFile\n";
    return 1;
  }
  std::ifstream inFile(argv[1], std::ios::binary | std::ios::ate);
  auto sz = inFile.tellg();
  inFile.seekg(0, std::ios::beg);

  model_init();

  std::vector<char> inData(sz);
  if (!inFile.read((char *)model_input_ptr(), sz)) {
    std::cerr << "Failed to read input file\n";
    return 1;
  }

  model_invoke();
  for (size_t i = 0; i < model_output_size(); i++) {
    std::cout << "\\x" << std::setw(2) << std::setfill('0') << std::hex
              << (int)((unsigned char *)model_output_ptr())[i];
  }
  std::cout << std::endl;

  return 0;
}
