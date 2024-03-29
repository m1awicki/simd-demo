#Copyright (c) 2013-2022, Bartosz Taudul <wolf@nereid.pl>
#All rights reserved.

ARCH := $(shell uname -m)

CFLAGS := -O3 -s
DEFINES := -DNDEBUG

ifeq ($(ARCH),x86_64)
CFLAGS += -march=native
endif
ifeq ($(ARCH),aarch64)
CFLAGS += -mcpu=native
endif

include build.mk
