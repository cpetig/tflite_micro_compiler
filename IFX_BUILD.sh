#!/bin/bash
set -e
source ../SETTINGS_AND_VERSIONS.sh

TFLITE_MICRO_ROOT=${TOOLSPREFIX}/tflite_u-${TFLITE_MICRO_VERSION}

while [[ "$1" != "" && "$1" != "--" ]]
do
    case "$1" in
    "--help"|"-?") 
	echo " [--force-install] " 1>&2
        exit 1
        ;;

    "--force-install")
        FORCE_INSTALL=1
        ;;
    "*")
        break
        ;;
    esac
    shift
done

make -f IFX_Makefile mrproper
make -f IFX_Makefile -j 6  build_compiler
make -f IFX_Makefile  install
