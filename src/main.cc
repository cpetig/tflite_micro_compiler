#include "CodeWriter.h"
#include "Compiler.h"
#include "Options.h"

#ifndef LOG_ARGC_ARGV
#define LOG_ARGC_ARGV 0
#endif  // LOG_ARGC_ARGV


/** The "real" main - physical main has workarounds for various
 * semi-hosting environments
 */

int hosted_main(int argc, char *argv[]) {

  tflmc::Options &options = tflmc::Options::instance();
  int cur_arg = 1;
  bool usage_error = false;
  while (cur_arg < argc && !usage_error) {

    const std::string verbose_flag("--verbose");
    const std::string memory_flag("--mem_summary");
    if (verbose_flag == argv[cur_arg]) {
      options.verbose = true;
    } else if (memory_flag == argv[cur_arg]) {
      if (cur_arg+1 < argc) {
        options.memmap_json = argv[cur_arg+1];
        ++cur_arg;
      } else {
        usage_error = true;
      }
    } else if (argv[cur_arg][0] == '-') {
      // No other "flag"
      usage_error = true;
    } else {
      break;
    }
    ++cur_arg;
  }
  if (cur_arg+1 >= argc || cur_arg+4 < argc) {
    printf(
        "Usage: %s [--verbose] [--mem_summary filename.json] modelFile.tflite outputSrcFile outputHdrFile [NamingPrefix (default: \"model_\")]\n",
        argv[0]);
    return 1;
  }

  std::string prefix = "model_";
  if (cur_arg+3 < argc) {
    prefix = argv[cur_arg+3];
  }

  if (!tflmc::CompileFile(argv[cur_arg], argv[cur_arg+1], argv[cur_arg+2], prefix)) {
    return 1;
  }

  return 0;
}

#ifdef __ARMCOMPILER_VERSION
extern "C" int arm_sh_parse_cmdline( char ***p_argv);
extern "C" void arm_sh_exit(int code);
#endif

int main(int argc, char *argv[]) {
  
#ifdef __ARMCOMPILER_VERSION
  // ARMClang runtime library has a very low (undocumented) maximum command line length
  // for its internal argv parsing - exceeding it silently results in empty argv/argc.
  argc = arm_sh_parse_cmdline(&argv);
#endif
#if LOG_ARGC_ARGV
  printf( "ARGC=%d ", argc);
  for(int i=0; i < argc; ++i) {
    printf(",%s", argv[i]);
  }
  printf("\n");
#endif
  int status = hosted_main(argc,argv);
#ifdef __ARMCOMPILER_VERSION
  if (status) {
    // ARMClang runtime library ignores exit status it always exits
    // angel_SWIreason_ReportException with ADP_Stopped_ApplicationExit
    arm_sh_exit(status);
  }
#endif
  return status;
}
