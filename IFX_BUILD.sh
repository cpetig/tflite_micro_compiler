#!/bin/bash

make -f IFX_Makefile mrproper
make -f IFX_Makefile -j 6  install

