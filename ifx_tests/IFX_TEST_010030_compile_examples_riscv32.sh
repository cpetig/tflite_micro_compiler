#!/bin/bash
set -e

cd `dirname "$BASH_SOURCE"`
make -C .. -f IFX_Makefile mrproper
make -C .. -f IFX_Makefile -j 4 TARGET=ifx_riscv TARGET_ARCH=hosted run_examples




