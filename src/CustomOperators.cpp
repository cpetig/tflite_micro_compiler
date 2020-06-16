/* Copyright 2020 Christof Petig. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "CustomOperators.h"

#include <unistd.h>

#include <iostream>

// dynamic loading for custom operators
#ifndef _WIN32
#include <dlfcn.h>

tflmc::custom_operator_handle tflmc::LoadCustom(
    tflite::AllOpsResolver *resolver) {
  const char *filename = "./libtflite_micro_custom.so";
  void *custom_lib = dlopen(filename, RTLD_NOW);
  if (custom_lib) {
    TfLiteStatus (*reg_fun)(tflite::AllOpsResolver * res);
    // see "man dlopen" for an explanation of this nasty construct
    *(void **)(&reg_fun) = dlsym(custom_lib, "register_custom");
    char *error = dlerror();
    if (error) {
      std::cerr << filename << ": " << error << "\n";
    } else if (reg_fun) {
      (*reg_fun)(resolver);
    }
  } else if (!access(filename, 0)) {  // only output error if the plugin exists
    char *error = dlerror();
    if (error) {
      std::cerr << filename << ": " << error << "\n";
    }
  }
  return custom_lib;
}

void tflmc::UnloadCustom(tflmc::custom_operator_handle custom_lib) {
  if (custom_lib) {
    dlclose(custom_lib);
  }
}

#else
// anyone interested in implementing this for Windows (LoadLibrary+GetProcAddr)
tflmc::custom_operator_handle tflmc::LoadCustom(
    tflite::AllOpsResolver *resolver) {
  return nullptr;
}

void tflmc::UnloadCustom(tflmc::custom_operator_handle) {}
#endif
