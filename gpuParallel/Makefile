################################################################################
#
# Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
#
# NOTICE TO USER:
#
# This source code is subject to NVIDIA ownership rights under U.S. and
# international Copyright laws.
#
# NVIDIA MAKES NO REPRESENTATION ABOUT THE SUITABILITY OF THIS SOURCE
# CODE FOR ANY PURPOSE.  IT IS PROVIDED "AS IS" WITHOUT EXPRESS OR
# IMPLIED WARRANTY OF ANY KIND.  NVIDIA DISCLAIMS ALL WARRANTIES WITH
# REGARD TO THIS SOURCE CODE, INCLUDING ALL IMPLIED WARRANTIES OF
# MERCHANTABILITY, NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
# IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY SPECIAL, INDIRECT, INCIDENTAL,
# OR CONSEQUENTIAL DAMAGES, OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS
# OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE
# OR OTHER TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE
# OR PERFORMANCE OF THIS SOURCE CODE.
#
# U.S. Government End Users.  This source code is a "commercial item" as
# that term is defined at 48 C.F.R. 2.101 (OCT 1995), consisting  of
# "commercial computer software" and "commercial computer software
# documentation" as such terms are used in 48 C.F.R. 12.212 (SEPT 1995)
# and is provided to the U.S. Government only as a commercial end item.
# Consistent with 48 C.F.R.12.212 and 48 C.F.R. 227.7202-1 through
# 227.7202-4 (JUNE 1995), all U.S. Government End Users acquire the
# source code with only those rights set forth herein.
#
################################################################################
#
# Makefile project only supported on Mac OS X and Linux Platforms)
#
################################################################################

# Location of the CUDA Toolkit
CUDA_PATH ?= "/usr/local/cuda-10.1"

##############################
# start deprecated interface #
##############################
ifeq ($(x86_64),1)
    $(info WARNING - x86_64 variable has been deprecated)
    $(info WARNING - please use TARGET_ARCH=x86_64 instead)
    TARGET_ARCH ?= x86_64
endif
ifeq ($(ARMv7),1)
    $(info WARNING - ARMv7 variable has been deprecated)
    $(info WARNING - please use TARGET_ARCH=armv7l instead)
    TARGET_ARCH ?= armv7l
endif
ifeq ($(aarch64),1)
    $(info WARNING - aarch64 variable has been deprecated)
    $(info WARNING - please use TARGET_ARCH=aarch64 instead)
    TARGET_ARCH ?= aarch64
endif
ifeq ($(ppc64le),1)
    $(info WARNING - ppc64le variable has been deprecated)
    $(info WARNING - please use TARGET_ARCH=ppc64le instead)
    TARGET_ARCH ?= ppc64le
endif
ifneq ($(GCC),)
    $(info WARNING - GCC variable has been deprecated)
    $(info WARNING - please use HOST_COMPILER=$(GCC) instead)
    HOST_COMPILER ?= $(GCC)
endif
ifneq ($(abi),)
    $(error ERROR - abi variable has been removed)
endif
############################
# end deprecated interface #
############################

# architecture
HOST_ARCH   := $(shell uname -m)
TARGET_ARCH ?= $(HOST_ARCH)
ifneq (,$(filter $(TARGET_ARCH),x86_64 aarch64 ppc64le armv7l))
    ifneq ($(TARGET_ARCH),$(HOST_ARCH))
        ifneq (,$(filter $(TARGET_ARCH),x86_64 aarch64 ppc64le))
            TARGET_SIZE := 64
        else ifneq (,$(filter $(TARGET_ARCH),armv7l))
            TARGET_SIZE := 32
        endif
    else
        TARGET_SIZE := $(shell getconf LONG_BIT)
    endif
else
    $(error ERROR - unsupported value $(TARGET_ARCH) for TARGET_ARCH!)
endif
ifneq ($(TARGET_ARCH),$(HOST_ARCH))
    ifeq (,$(filter $(HOST_ARCH)-$(TARGET_ARCH),aarch64-armv7l x86_64-armv7l x86_64-aarch64 x86_64-ppc64le))
        $(error ERROR - cross compiling from $(HOST_ARCH) to $(TARGET_ARCH) is not supported!)
    endif
endif

# When on native aarch64 system with userspace of 32-bit, change TARGET_ARCH to armv7l
ifeq ($(HOST_ARCH)-$(TARGET_ARCH)-$(TARGET_SIZE),aarch64-aarch64-32)
    TARGET_ARCH = armv7l
endif

# operating system
HOST_OS   := $(shell uname -s 2>/dev/null | tr "[:upper:]" "[:lower:]")
TARGET_OS ?= $(HOST_OS)
ifeq (,$(filter $(TARGET_OS),linux darwin qnx android))
    $(error ERROR - unsupported value $(TARGET_OS) for TARGET_OS!)
endif

# host compiler
ifeq ($(TARGET_OS),darwin)
    ifeq ($(shell expr `xcodebuild -version | grep -i xcode | awk '{print $$2}' | cut -d'.' -f1` \>= 5),1)
        HOST_COMPILER ?= clang++
    endif
else ifneq ($(TARGET_ARCH),$(HOST_ARCH))
    ifeq ($(HOST_ARCH)-$(TARGET_ARCH),x86_64-armv7l)
        ifeq ($(TARGET_OS),linux)
            HOST_COMPILER ?= arm-linux-gnueabihf-g++
        else ifeq ($(TARGET_OS),qnx)
            ifeq ($(QNX_HOST),)
                $(error ERROR - QNX_HOST must be passed to the QNX host toolchain)
            endif
            ifeq ($(QNX_TARGET),)
                $(error ERROR - QNX_TARGET must be passed to the QNX target toolchain)
            endif
            export QNX_HOST
            export QNX_TARGET
            HOST_COMPILER ?= $(QNX_HOST)/usr/bin/arm-unknown-nto-qnx6.6.0eabi-g++
        else ifeq ($(TARGET_OS),android)
            HOST_COMPILER ?= arm-linux-androideabi-g++
        endif
    else ifeq ($(TARGET_ARCH),aarch64)
        ifeq ($(TARGET_OS), linux)
            HOST_COMPILER ?= aarch64-linux-gnu-g++
        else ifeq ($(TARGET_OS),qnx)
            ifeq ($(QNX_HOST),)
                $(error ERROR - QNX_HOST must be passed to the QNX host toolchain)
            endif
            ifeq ($(QNX_TARGET),)
                $(error ERROR - QNX_TARGET must be passed to the QNX target toolchain)
            endif
            export QNX_HOST
            export QNX_TARGET
            HOST_COMPILER ?= $(QNX_HOST)/usr/bin/aarch64-unknown-nto-qnx7.0.0-g++
        else ifeq ($(TARGET_OS), android)
            HOST_COMPILER ?= aarch64-linux-android-g++
        endif
    else ifeq ($(TARGET_ARCH),ppc64le)
        HOST_COMPILER ?= powerpc64le-linux-gnu-g++
    endif
endif
HOST_COMPILER ?= g++
NVCC          := $(CUDA_PATH)/bin/nvcc -ccbin $(HOST_COMPILER)

# internal flags
NVCCFLAGS   := -m${TARGET_SIZE}
CCFLAGS     := -std=c++11
LDFLAGS     :=

# build flags
ifeq ($(TARGET_OS),darwin)
    LDFLAGS += -rpath $(CUDA_PATH)/lib
    CCFLAGS += -arch $(HOST_ARCH)
else ifeq ($(HOST_ARCH)-$(TARGET_ARCH)-$(TARGET_OS),x86_64-armv7l-linux)
    LDFLAGS += --dynamic-linker=/lib/ld-linux-armhf.so.3
    CCFLAGS += -mfloat-abi=hard
else ifeq ($(TARGET_OS),android)
    LDFLAGS += -pie
    CCFLAGS += -fpie -fpic -fexceptions
endif

ifneq ($(TARGET_ARCH),$(HOST_ARCH))
    ifeq ($(TARGET_ARCH)-$(TARGET_OS),armv7l-linux)
        ifneq ($(TARGET_FS),)
            GCCVERSIONLTEQ46 := $(shell expr `$(HOST_COMPILER) -dumpversion` \<= 4.6)
            ifeq ($(GCCVERSIONLTEQ46),1)
                CCFLAGS += --sysroot=$(TARGET_FS)
            endif
            LDFLAGS += --sysroot=$(TARGET_FS)
            LDFLAGS += -rpath-link=$(TARGET_FS)/lib
            LDFLAGS += -rpath-link=$(TARGET_FS)/usr/lib
            LDFLAGS += -rpath-link=$(TARGET_FS)/usr/lib/arm-linux-gnueabihf
        endif
    endif
endif

ifeq ($(TARGET_OS),qnx)
    CCFLAGS += -DWIN_INTERFACE_CUSTOM
    LDFLAGS += -lsocket
endif

# Install directory of different arch
CUDA_INSTALL_TARGET_DIR :=
ifeq ($(TARGET_ARCH)-$(TARGET_OS),armv7l-linux)
    CUDA_INSTALL_TARGET_DIR = targets/armv7-linux-gnueabihf/
else ifeq ($(TARGET_ARCH)-$(TARGET_OS),aarch64-linux)
    CUDA_INSTALL_TARGET_DIR = targets/aarch64-linux/
else ifeq ($(TARGET_ARCH)-$(TARGET_OS),armv7l-android)
    CUDA_INSTALL_TARGET_DIR = targets/armv7-linux-androideabi/
else ifeq ($(TARGET_ARCH)-$(TARGET_OS),aarch64-android)
    CUDA_INSTALL_TARGET_DIR = targets/aarch64-linux-androideabi/
else ifeq ($(TARGET_ARCH)-$(TARGET_OS),armv7l-qnx)
    CUDA_INSTALL_TARGET_DIR = targets/ARMv7-linux-QNX/
else ifeq ($(TARGET_ARCH)-$(TARGET_OS),aarch64-qnx)
    CUDA_INSTALL_TARGET_DIR = targets/aarch64-qnx/
else ifeq ($(TARGET_ARCH),ppc64le)
    CUDA_INSTALL_TARGET_DIR = targets/ppc64le-linux/
endif

# Debug build flags
ifeq ($(dbg),1)
      NVCCFLAGS += -g -G
      BUILD_TYPE := debug
else
      BUILD_TYPE := release
endif

ALL_CCFLAGS :=
ALL_CCFLAGS += -std=c++11
ALL_CCFLAGS += -w
ALL_CCFLAGS += $(NVCCFLAGS)
ALL_CCFLAGS += $(EXTRA_NVCCFLAGS)
ALL_CCFLAGS += $(addprefix -Xcompiler ,$(CCFLAGS))
ALL_CCFLAGS += $(addprefix -Xcompiler ,$(EXTRA_CCFLAGS))
ALL_CCFLAGS += -Xcompiler -fopenmp

SAMPLE_ENABLED := 1

ALL_LDFLAGS :=
ALL_LDFLAGS += $(ALL_CCFLAGS)
ALL_LDFLAGS += $(addprefix -Xlinker ,$(LDFLAGS))
ALL_LDFLAGS += $(addprefix -Xlinker ,$(EXTRA_LDFLAGS))

# Common includes and paths for CUDA
INCLUDES  :=
INCLUDES  := -I ./tfheSource/customSource/newInclude2/
INCLUDES  += -I /usr/local/include/
INCLUDES  += -I ./cuda_common/inc/
INCLUDES  += -I ./
LIBRARIES :=
#LIBRARIES += -lgomp
LIBRARIES += -L/usr/local/cuda/lib64
LIBRARIES += -L/usr/local/cuda-10.1/lib64
LIBRARIES += -lfftw3
LIBRARIES += -lcufft
LIBRARIES += -lcudart


################################################################################

# Gencode arguments
SMS ?= 30 35 37 50 52 60 70

ifeq ($(SMS),)
$(info >>> WARNING - no SM architectures have been specified - waiving sample <<<)
SAMPLE_ENABLED := 0
endif

ifeq ($(GENCODE_FLAGS),)
# Generate SASS code for each SM architecture listed in $(SMS)
$(foreach sm,$(SMS),$(eval GENCODE_FLAGS += -gencode arch=compute_$(sm),code=sm_$(sm)))

# Generate PTX code from the highest SM architecture in $(SMS) to guarantee forward-compatibility
HIGHEST_SM := $(lastword $(sort $(SMS)))
ifneq ($(HIGHEST_SM),)
GENCODE_FLAGS += -gencode arch=compute_$(HIGHEST_SM),code=compute_$(HIGHEST_SM)
endif
endif

ifeq ($(SAMPLE_ENABLED),0)
EXEC ?= @echo "[@]"
endif

################################################################################

# Target rules
#all: build

#build: matrixMul

#check.deps:
#ifeq ($(SAMPLE_ENABLED),0)
#	@echo "Sample will be waived due to the above missing dependencies"
#else
#	@echo "Sample is ready - all dependencies have been met"
#endif

#matrixMul.o: main.cpp
#	$(EXEC) $(NVCC) $(INCLUDES) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -o $@ -c $<

#matrixMul: matrixMul.o
#	$(EXEC) $(NVCC) $(ALL_LDFLAGS) $(GENCODE_FLAGS) -o $@ $+ $(LIBRARIES)
#	$(EXEC) mkdir -p bin/$(TARGET_ARCH)/$(TARGET_OS)/$(BUILD_TYPE)
#	$(EXEC) cp $@ bin/$(TARGET_ARCH)/$(TARGET_OS)/$(BUILD_TYPE)

#run: build
#	$(EXEC) ./matrixMul

#clean:
#	rm -f matrixMul matrixMul.o
#	rm -rf bin/$(TARGET_ARCH)/$(TARGET_OS)/$(BUILD_TYPE)/matrixMul


#clobber: clean
#print:
###########create variables
# MAIN_HEADERS :=
# CLOUD_HEADERS :=
# MAIN_HEADERS += main.cu Cipher.cu
# CLOUD_HEADERS += cloud.cu Cipher.cu
# MAIN_HEADERS += $(wildcard ./tfheSource/customSource/newInclude2/*.cu)
# CLOUD_HEADERS += $(wildcard ./tfheSource/customSource/newInclude2/*.cu)
# OBJECTS := $(wildcard *.o)

# all: cloud main

# #build: cloud

# cloud: $(CLOUD_HEADERS)
# 	$(EXEC) $(NVCC) $(INCLUDES) $(ALL_LDFLAGS) $(GENCODE_FLAGS) $(LIBRARIES) -o $@ $+

# main: $(MAIN_HEADERS)
# 	$(EXEC) $(NVCC) $(INCLUDES) $(ALL_LDFLAGS) $(GENCODE_FLAGS) $(LIBRARIES) -o $@ $+

# clean:
# 	rm -f cloud
# 	rm -f main
SRC_FILES :=

SRC_FILES += autogenerated.cu
SRC_FILES += boot-gates.cu
SRC_FILES += tgsw-functions.cu tgsw-functions.cu
SRC_FILES += cudaFFTTest.h cudaFFTTest.cu
SRC_FILES += fft_processor_fftw.cu
SRC_FILES += lagrangehalfc_impl.cu
SRC_FILES += lwe-bootstrapping-functions.cu
SRC_FILES += lwe-bootstrapping-functions-fft.cu
SRC_FILES += lwebootstrappingkey.cu
SRC_FILES += lwe.cu
SRC_FILES += lwe-functions.cu
SRC_FILES += lwekey.cu
SRC_FILES += lwekeyswitch.cu
SRC_FILES += lwe-keyswitch-functions.cu
SRC_FILES += lweparams.cu
SRC_FILES += lwesamples.cu
SRC_FILES += multiplication.cu
SRC_FILES += numeric-functions.cu
SRC_FILES += tfhe_garbage_collector.cu
SRC_FILES += tfhe_gate_bootstrapping.cu
SRC_FILES += tfhe_gate_bootstrapping_structures.cu
SRC_FILES += tfhe_generic_streams.cu
SRC_FILES += tfhe_io.cu
SRC_FILES += tgsw.cu
SRC_FILES += polynomials.cu
SRC_FILES += tgsw-fft-operations.cu
SRC_FILES += tlwe.cu
SRC_FILES += tlwe-fft-operations.cu
SRC_FILES += tlwe-functions.cu
SRC_FILES += toruspolynomial-functions.cu
SRC_FILES += Cipher.cu
#SRC_FILES += matrixUtility.h matrixUtility.cu

CLOUD_SRC_FILES := $(SRC_FILES)
CLOUD_SRC_FILES += cloud.cu

MAIN_SRC_FILES := $(SRC_FILES)
MAIN_SRC_FILES += main.cu

OBJ_FILES :=
OBJ_FILES += cudaFFTTest.o
OBJ_FILES += autogenerated.o
OBJ_FILES += tgsw-functions.o
OBJ_FILES += boot-gates.o
OBJ_FILES += fft_processor_fftw.o
OBJ_FILES += lagrangehalfc_impl.o
OBJ_FILES += lwe-bootstrapping-functions.o
OBJ_FILES += lwe-bootstrapping-functions-fft.o
OBJ_FILES += lwebootstrappingkey.o
OBJ_FILES += lwe.o
OBJ_FILES += lwe-functions.o
OBJ_FILES += lwekey.o
OBJ_FILES += lwekeyswitch.o
OBJ_FILES += lwe-keyswitch-functions.o
OBJ_FILES += lweparams.o
OBJ_FILES += lwesamples.o
OBJ_FILES += multiplication.o
OBJ_FILES += numeric-functions.o
OBJ_FILES += tfhe_garbage_collector.o
OBJ_FILES += tfhe_gate_bootstrapping.o
OBJ_FILES += tfhe_gate_bootstrapping_structures.o
OBJ_FILES += tfhe_generic_streams.o
OBJ_FILES += tfhe_io.o
OBJ_FILES += tgsw.o
OBJ_FILES += polynomials.o
OBJ_FILES += tgsw-fft-operations.o
OBJ_FILES += tlwe.o
OBJ_FILES += tlwe-fft-operations.o
OBJ_FILES += tlwe-functions.o
OBJ_FILES += toruspolynomial-functions.o
OBJ_FILES += Cipher.o
OBJ_FILES += matrixUtility.o


CLOUD_OBJ_FILES := $(OBJ_FILES)
CLOUD_OBJ_FILES += cloud.o

MAIN_OBJ_FILES:= $(OBJ_FILES)
MAIN_OBJ_FILES += main.o



# $DEPENDS  = -MT $@  -MP -MF $(subst .o,.d,$@)

OBJDIR := ./object

# cloud: $(OBJ_FILES)
	# $(EXEC) $(NVCC) $(INCLUDES) $(ALL_LDFLAGS) $(GENCODE_FLAGS) $(LIBRARIES) -o $@ $+

# $(OBJ_DIR)/%.o: $(SRC_FILES)
# 	$(EXEC) $(NVCC) $(INCLUDES) $(ALL_LDFLAGS) $(GENCODE_FLAGS) $(LIBRARIES) -c $@ $<
all: cloud main

cloud: $(CLOUD_OBJ_FILES)
	$(EXEC) $(NVCC) $(INCLUDES) $(ALL_LDFLAGS) $(GENCODE_FLAGS) $(LIBRARIES) -o $@ $+
    # g++ -o engine main.o correlation. matcher.o ...

%.o: %.cu
	$(EXEC) $(NVCC) $(INCLUDES) $(ALL_LDFLAGS) $(GENCODE_FLAGS) $(LIBRARIES) -c  $@ $+

main: $(MAIN_OBJ_FILES)
	rm -f input_old.txt input_new.txt output_old.txt output_new.txt new_ifft_input.txt new_ifft_output.txt old_ifft_input.txt old_ifft_output.txt newComplex.txt oldComplex.txt newAllSample.txt oldAllSample.txt
	$(EXEC) $(NVCC) $(INCLUDES) $(ALL_LDFLAGS) $(GENCODE_FLAGS) $(LIBRARIES) -o $@ $+

# %.o: %.cu
	# $(EXEC) $(NVCC) $(INCLUDES) $(ALL_LDFLAGS) $(GENCODE_FLAGS) $(LIBRARIES) -c  $@ $+
# ./tfheSource/customSource/newInclude2/%.o: ./tfheSource/customSource/newInclude2/%.cu
	# $(EXEC) $(NVCC) $(INCLUDES) $(ALL_LDFLAGS) $(GENCODE_FLAGS) $(LIBRARIES) -c $@ $<


# %.o: ./tfheSource/customSource/newInclude2/%.cu %.cu
	# $(EXEC) $(NVCC) $(INCLUDES) $(ALL_LDFLAGS) $(GENCODE_FLAGS) $(LIBRARIES) -c $@ $<

clean: 
	rm -f *.o
	rm -f cloud
	rm -f main




