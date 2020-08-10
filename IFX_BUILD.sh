#!/bin/bash

make -f IFX_Makefile TARGET_ARCH=x86_64 clean
make -f IFX_Makefile -j 4 TARGET_ARCH=x86_64 install
