#!/bin/bash

make -f IFX_Makefile TARGET_ARCH=i386 clean
make -f IFX_Makefile clean
make -f IFX_Makefile -j 4 install

