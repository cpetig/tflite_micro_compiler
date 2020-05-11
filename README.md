# tflite_micro_compiler
generate tflite micro code which bypasses the interpreter (directly calls into kernels)

Basically this code uses a fully set up tflite micro instance to dump the internal allocations and
function calls assigned to the model, then dumps the tensor and node settings into a compileable 
file, eliminating the need for running the interpreter at each program start and resolving the correct
kernel at run time.

Limitations:
- limited set of supported operators (due to low number of entries in code generator), typically two to fifteen lines are needed to support a new operator 

Building the code:
- check out tensorflow master next to this project (in ../tensorflow)
- start with building the tflite micro library as described in https://www.tensorflow.org/lite/microcontrollers/library:
    - cd ../tensorflow
    - make -f tensorflow/lite/micro/tools/make/Makefile hello_world_bin
    [optionally add BUILD_TARGET=debug]
- now run  make  in this project

USAGE:
- the !NEW! compiler is invoked as:
    - ./compiler input.tflite arena_size prefix >output.cpp
        e.g. ./compiler hello_world.tflite 3000 hello >hello_compiled.cpp
- there is also a test and example:
    - ./hello_world : execute using standard tflite micro interpreter
    - ./hello_world_compiled : execute using the dumped and compiled in data structure

- regerenerate the compileable code by running
    - make regenerate
