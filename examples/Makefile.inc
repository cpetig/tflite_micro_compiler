
MOBILENET_SRCS := \
  examples/mobilenet.cc  examples/mobilenet_v1_0_25_160_quantized.c  examples/gnu.c

MOBILENET_COMPILED_SRCS := \
  examples/mobilenet2.cc  examples/compiled_mobilenet.cc examples/gnu.c

MOBILENET_HDRS := 
MOBILENET_COMPILED_HDRS := 

HELLO_WORLD_SRC := \
  examples/hello_world.cc examples/hello_world_model.cc
HELLO_WORLD_HDRS :=

HELLO_WORLD_COMPILED_SRC := \
  examples/hello_world2.cc examples/compiled_hello_world.cc
HELLO_WORLD_COMPILED_HDRS :=


CIFAR10_SRC := \
  examples/cifar10.cc examples/cifar10_model.cc examples/truck.c
CIFAR10_HDRS :=


CIFAR10_COMPILED_SRC := \
  examples/cifar10_compiled.cc examples/cifar10_run_comp.cc examples/truck.c
CIFAR10_COMPILED_HDRS :=

$(info Adding mobilenet mobilenet_compiled )

# Builds a standalone binary.
$(eval $(call microlite_test,mobilenet,\
  $(MOBILENET_SRCS),$(MOBILENET_HDRS)))

$(eval $(call microlite_test,mobilenet_compiled,\
  $(MOBILENET_COMPILED_SRCS),$(MOBILENET_COMPILED_HDRS)))

$(info Adding cifar10 cifar10_compiled )
$(eval $(call microlite_test,cifar10,\
  $(CIFAR10_SRC),$(CIFAR10_HDRS)))

$(eval $(call microlite_test,cifar10_compiled,\
  $(CIFAR10_SRC),$(CIFAR10_HDRS)))

$(info Adding hello_world  )
$(eval $(call microlite_test,hello_world,\
  $(HELLO_WORLD_SRC),$(HELLO_WORLD_HDRS)))

$(eval $(call microlite_test,hello_world_compiled,\
  $(HELLO_WORLD_COMPILED_SRC),$(HELLO_WORLD_COMPILED_HDRS)))