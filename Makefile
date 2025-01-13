# Basic Makefile for building the Transformer CUDA project

CUDA_PATH ?= /usr/local/cuda-12.4
CUDNN_PATH ?= /usr/local/cuda-12.4
CC := $(CUDA_PATH)/bin/nvcc
TARGET := transformer

SRCDIR := src
INCDIR := include
BINDIR := bin

# Include all source files, including those in subdirectories
SOURCES := $(shell find $(SRCDIR) -name '*.cpp' -o -name '*.cu')
OBJECTS := $(SOURCES:.cpp=.o)
OBJECTS := $(OBJECTS:.cu=.o)

CFLAGS := -I$(INCDIR) -I$(CUDNN_PATH)/include -I$(CUDA_PATH)/include -I/usr/include -I/usr/local/include
LDFLAGS := -L$(CUDNN_PATH)/lib64 -lcudnn -lcublas -lcurand

debug: CFLAGS += -DDEBUG
debug: all

all: $(BINDIR)/$(TARGET)

$(BINDIR)/$(TARGET): $(OBJECTS)
	mkdir -p $(BINDIR)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

%.o: %.cpp
	$(CC) $(CFLAGS) -c -o $@ $<

%.o: %.cu
	$(CC) $(CFLAGS) -c -o $@ $<

clean:
	rm -rf $(BINDIR) $(OBJECTS)

run: $(BINDIR)/$(TARGET)
	./$(BINDIR)/$(TARGET)