# tflite_micro_compiler
generate tflite micro code which bypasses the interpreter (directly calls into kernels)

Basically this code uses a fully set up tflite micro instance to dump the internal allocations and
function calls assigned to the model, then dumps the tensor and node settings into a compileable file.

Limitations:
- no quantization yet
- very limited set of supported operators (due to low number of entries in code generator), typically two to fifteen lines are needed to support a new operator 

Building the code:
- check out tensorflow master next to this project (in ../tensorflow)
- start with building the library as described in https://www.tensorflow.org/lite/microcontrollers/library:
    - cd ../tensorflow
    - make -f tensorflow/lite/micro/tools/make/Makefile generate_projects
    - make -f tensorflow/lite/micro/tools/make/Makefile hello_world_bin
- now run make in this project and regerenerate the code by running
    - ./mobilnet >compiled_mobilnet.cpp
    - ./hello_world >compiled_hello.cpp
