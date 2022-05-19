#Copyright (c) 2013-2022, Bartosz Taudul <wolf@nereid.pl>
#All rights reserved.

ARCH := $(shell uname -m)

CFLAGS := -g3 -Wall
DEFINES := -DDEBUG

ifeq ($(ARCH),x86_64)
CFLAGS += -msse4.1
endif

include build.mk
