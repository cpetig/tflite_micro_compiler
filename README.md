# tflite_micro_compiler

Generate tflite micro code which bypasses the interpreter (directly calls into kernels)

Basically this code uses a fully set up tflite micro instance to dump the internal allocations and
function calls assigned to the model, then dumps the tensor and node settings into a compileable 
file, eliminating the need for running the interpreter at each program start and for resolving the correct
kernel at run time.

Building the code:

- check out tensorflow master next to this project (in ../tensorflow)
- start with building the tflite micro library as described in https://www.tensorflow.org/lite/microcontrollers/library:

  - cd ../tensorflow

  - make -f tensorflow/lite/micro/tools/make/Makefile hello_world_bin
    [optionally add BUILD_TYPE=debug]

- now run  make  in this project to get the compiler

USAGE:

- the compiler is invoked as:

  - ./compiler input.tflite arena_size prefix >output.cpp

    e.g. ./compiler hello_world.tflite 3000 hello_ >hello_compiled.cpp

    "arena_size" is the size of the temporary tensor buffer (activation storage during execution)
    please note that this arena is also used for storing precalculated vales in the prepare stage
    simply re-run init if you disturb the values inside

- for a quick view into the generated code see https://github.com/cpetig/tflite_micro_compiler/blob/master/examples/compiled_hello.cc

- The example directory contains a collection of traditional tflite micro and compiled versions:

  - hello_world: Standard tflite micro example
  - cifar10: Computer vision CNN example

Customization:

You can decide whether to compile in the original tflite file or dump tensors and metadata into the compiled code:
Simply change EMBED_TENSORS in compiler_config.h

This is motivated by avoiding large binary blobs in source code (qualification) but introduces a slight rounding error. It also saves some additional bytes.

Limitations:

- no support for big endian machines, yet
- currently the compiler seems to crash but produces correct code (under investigation)
