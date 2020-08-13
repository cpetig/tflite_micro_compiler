#!/bin/bash
set -e

cd `dirname "$BASH_SOURCE"`
make -C .. -f IFX_Makefile clean 
make -C .. -f IFX_Makefile TARGET_ARCH=i386 clean




