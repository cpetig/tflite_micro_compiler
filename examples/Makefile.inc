COMPILED_FILES_DIR =  examples/$(COMPILED_FILES_SELECTION)
INCLUDES += -I $(COMPILED_FILES_DIR)

MOBILENET_SRCS := \
  examples/mobilenet.cc  examples/mobilenet_model.c  examples/gnu.c

MOBILENET_COMPILED_SRCS := \
  examples/mobilenet2.cc   $(COMPILED_FILES_DIR)/compiled_mobilenet.cc examples/gnu.c

MOBILENET_HDRS := 
MOBILENET_COMPILED_HDRS := 

HELLO_WORLD_SRC := \
  examples/hello_world.cc examples/hello_world_model.cc
HELLO_WORLD_HDRS :=

HELLO_WORLD_COMPILED_SRC := \
  examples/hello_world2.cc $(COMPILED_FILES_DIR)/compiled_hello_world.cc
HELLO_WORLD_COMPILED_HDRS :=

HELLO_WORLD_5_SRC := \
  examples/hello_world_5.cc examples/hello_world_5_model.cc
HELLO_WORLD_5_HDRS :=

HELLO_WORLD_5_COMPILED_SRC := \
  examples/hello_world2_5.cc  $(COMPILED_FILES_DIR)/compiled_hello_world_5.cc
HELLO_WORLD_5_COMPILED_HDRS :=

HELLO_WORLD_SPARSE_SRC := \
  examples/hello_world_sparse.cc examples/hello_world_pruned_8.cc
HELLO_WORLD_SPARSE_HDRS :=

HELLO_WORLD_SPARSE_COMPILED_SRC := \
  examples/hello_world2_sparse.cc  $(COMPILED_FILES_DIR)/compiled_hello_world_pruned_8.cc
HELLO_WORLD_SPARSE_COMPILED_HDRS :=

CIFAR10_SRC := \
  examples/cifar10.cc examples/cifar10_model.cc examples/truck.c
CIFAR10_HDRS :=


CIFAR10_COMPILED_SRC := \
  $(COMPILED_FILES_DIR)/compiled_cifar10.cc  examples/cifar10_run_comp.cc examples/truck.c
CIFAR10_COMPILED_HDRS :=


# Builds a standalone binary.
$(eval $(call microlite_test,mobilenet,\
  $(MOBILENET_SRCS),$(MOBILENET_HDRS)))

$(eval $(call microlite_test,mobilenet_compiled,\
  $(MOBILENET_COMPILED_SRCS),$(MOBILENET_COMPILED_HDRS)))

$(eval $(call microlite_test,cifar10,\
  $(CIFAR10_SRC),$(CIFAR10_HDRS)))

$(eval $(call microlite_test,cifar10_compiled,\
  $(CIFAR10_COMPILED_SRC),$(CIFAR10_COMPILED_HDRS)))

$(eval $(call microlite_test,hello_world,\
  $(HELLO_WORLD_SRC),$(HELLO_WORLD_HDRS)))

$(eval $(call microlite_test,hello_world_compiled,\
  $(HELLO_WORLD_COMPILED_SRC),$(HELLO_WORLD_COMPILED_HDRS)))

$(eval $(call microlite_test,hello_world_5,\
  $(HELLO_WORLD_5_SRC),$(HELLO_WORLD_5_HDRS)))

$(eval $(call microlite_test,hello_world_5_compiled,\
  $(HELLO_WORLD_5_COMPILED_SRC),$(HELLO_WORLD_5_COMPILED_HDRS)))


$(eval $(call microlite_test,hello_world_pruned_8,\
  $(HELLO_WORLD_SPARSE_SRC),$(HELLO_WORLD_SPARSE_HDRS)))

$(eval $(call microlite_test,hello_world_pruned_8_compiled,\
  $(HELLO_WORLD_SPARSE_COMPILED_SRC),$(HELLO_WORLD_SPARSE)))