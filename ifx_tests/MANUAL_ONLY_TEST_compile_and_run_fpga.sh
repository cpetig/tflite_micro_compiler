#!/bin/bash
set -e
trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM EXIT

# keep track of the last executed command
trap 'last_command=$current_command; current_command=$BASH_COMMAND' DEBUG
# echo an error message before exiting

function exit_code_handling {
	if [[ $1 -eq 0 ]]
	then
	  echo "***** SUCCESS *****"
	else
	  echo "An error occured! GDB threw exit code $1 ."
	  trap - SIGTERM && kill -- -$$
	fi
}

scriptPath="$( cd "$( dirname "{BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
openocdPath="/c/openOCD/bin/openocd"
openocdConfig="${scriptPath}/minimumOCDS.cfg"
gdbScript="${scriptPath}/gdb_startup_commands.gdb"

gdbPath="/c/Inicio/tools/64/rvtc-ilp32-rival-1.0.2.4/bin/riscv32-unknown-elf-gdb"
executableRoot="${scriptPath}/../gen/"
executableFile="/bin/hello_world_compiled"


echo "----------------------------------------------------------------------------"
echo "Testing examples on FPGA. Make sure the J-Link dongle with the FPGA is plugged in and the paths to GDB and openocd are set correctly."
echo " - openOCD Path: ${openocdPath} "
echo " - gdb     Path: ${gdbPath} " 
echo "----------------------------------------------------------------------------"


cd `dirname "$BASH_SOURCE"`
make -C .. -f IFX_Makefile mrproper
${openocdPath} -f ${openocdConfig} &  # Start openOCD in background
make -C .. -f IFX_Makefile examples_fpga # Compile at the same time
executableFolder="portable_optimized_recorded_model_ifx_riscv32_fpga_rv32imc_debug"
trap 'EXIT_CODE=$? && echo "\"${last_command}\" command exited with exit code $EXIT_CODE ." && exit_code_handling $EXIT_CODE ' EXIT
${gdbPath} --command=${gdbScript} ${executableRoot}${executableFolder}${executableFile}
echo " *** SUCCESS *** Exit code $? "
trap "trap - SIGTERM && kill -- -$$" EXIT

make -C .. -f IFX_Makefile examples_fpga_mac_emustream
executableFolder="portable_optimized_recorded_model_ifx_emu_rival_ifx_riscv32_fpga_rv32imc_debug"
trap 'EXIT_CODE=$? && echo "\"${last_command}\" command exited with exit code $EXIT_CODE ." && exit_code_handling $EXIT_CODE ' EXIT
${gdbPath} --command=${gdbScript} ${executableRoot}${executableFolder}${executableFile}
echo " *** SUCCESS *** Exit code $? "
trap "trap - SIGTERM && kill -- -$$" EXIT

make -C .. -f IFX_Makefile examples_fpga_mac_nostream
executableFolder="portable_optimized_recorded_model_ifx_rival_non_streaming_ifx_riscv32_fpga_rv32imc_debug"
trap 'EXIT_CODE=$? && echo "\"${last_command}\" command exited with exit code $EXIT_CODE ." && exit_code_handling $EXIT_CODE ' EXIT
${gdbPath} --command=${gdbScript} ${executableRoot}${executableFolder}${executableFile}
echo " *** SUCCESS *** Exit code $? "
trap "trap - SIGTERM && kill -- -$$" EXIT

make -C .. -f IFX_Makefile examples_fpga_mac_stream
executableFolder="portable_optimized_recorded_model_ifx_strm_rival_ifx_riscv32_fpga_rv32imc_debug"
trap 'EXIT_CODE=$? && echo "\"${last_command}\" command exited with exit code $EXIT_CODE ." && exit_code_handling $EXIT_CODE ' EXIT
${gdbPath} --command=${gdbScript} ${executableRoot}${executableFolder}${executableFile}
echo " *** SUCCESS *** Exit code $? "
trap "trap - SIGTERM && kill -- -$$ && exit 0" EXIT SIGINT SIGTERM

# If openOCD can't be started (No J-Link Device found), it might be running in the background.
# Check that using 'ps -eaf | grep openocd'
# Then kill the process using 'kill process_id'


