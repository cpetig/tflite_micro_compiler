# tflite_micro_compiler

Generate tflite micro code which bypasses the interpreter (directly calls into kernels)

Basically this code uses a fully set up tflite micro instance to dump the internal allocations and
function calls assigned to the model, then dumps the tensor and node settings into a compilable
file, eliminating the need for running the interpreter at each program start and for resolving the correct
kernel at run time.

An in depth explanation of the motivation and benefits is included in the matching [RFC](https://docs.google.com/document/d/1wDqC50sjCaWyQxsSn_Y-XAGh8-ozIgm2HDzX_b9DIyo/edit?usp=sharing).

# Building

## CMake

Below the two methods of incorporating the TensorFlow sources into your build are
explained.

The basic flow of building with CMake is

``` bash
mkdir build
cd build
cmake [options] ..
make
```

### Examples 
The examples cmake [here](examples/CMakeLists.txt) is by default not included due to issues with TensorFlow source code compatibility when using specific code versions.
To enable building the examples pass `-DTF_EXAMPLES=ON` to CMake.

## Automatic TensorFlow Source Fetching

To pull the TensorFlow sources using CMake with the variable `GET_TF_SRC`
set to `ON`. 

e.g.

``` bash
cmake -DGET_TF_SRC=ON ..
```

This will retrieve the TensorFlow master branch's code. 
It should also be noted that `GET_TF_SRC` is prioritized over `TF_DIR` (see below).
If you want to specify a TensorFlow tag to checkout then this can be passed to
CMake using the option `TF_TAG`. 

e.g.

``` bash
cmake -DGET_TF_SRC=ON TF_TAG=v2.2.0 ..
```

Similarly a Git commit hash can be provided using `TF_COMMIT`. Note that
`TF_TAG` takes precedence if both are provided.

e.g.

```bash
cmake -DGET_TF_SRC=ON TF_COMMIT=0fecf6f89fd7bacc1ec4213b946a254e885b82ac ..
```

To checkout a different TensorFlow code base without clearing the CMake cache
the argument `TF_RECACHE` should be set, this will force the TensorFlow
source to be checked-out again.

e.g.

```bash
cmake -DGET_TF_SRC=ON -DTF_RECACHE=ON TF_COMMIT=0fecf6f89fd7bacc1ec4213b946a254e885b82ac ..
```

## Providing TensorFlow Source Manually

By default CMake looks for the TensorFlow source in the directory `../tensorflow`.
If you want to specify you TensorFlow source directory this can be done by
providing the argument `TF_DIR`. 

e.g.

``` bash
cmake -DTF_DIR=../my_tensorflow ..
```

## Additional Targets

### format

To invoke `clang-format` CMake provides the `format` target.

e.g.

```bash 
cmake ..
make format
```

## Make

- check out tensorflow master next to this project (in ../tensorflow)
- start with building the tflite micro library as described in https://www.tensorflow.org/lite/microcontrollers/library:

  - `cd ../tensorflow`

  - `make -f tensorflow/lite/micro/tools/make/Makefile hello_world_bin`
    [optionally add BUILD_TYPE=debug]

- now run  make  in this project to get the compiler

# Usage

- the compiler is invoked as `./compiler input.tflite output.cpp [prefix]`

    e.g.

    ``` bash 
    ./compiler hello_world.tflite hello_compiled.cpp hello_
    ```

- for a quick view into the generated code see [`compiled_hello_world.cc`](https://github.com/cpetig/tflite_micro_compiler/blob/master/examples/compiled_hello_world.cc)

  You can compare calling into interpreter and compiled code between [`hello_world.cc`](https://github.com/cpetig/tflite_micro_compiler/blob/master/examples/hello_world.cc)
  and [`hello_world2.cc`](https://github.com/cpetig/tflite_micro_compiler/blob/master/examples/hello_world2.cc)

- The example directory contains a collection of traditional tflite micro and compiled versions:

  - hello_world: Standard tflite micro example
  - cifar10: Computer vision CNN example

# Limitations

- no support for big endian machines, yet
