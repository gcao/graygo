CXX = g++
CXXFLAGS = -O3 -std=c++17 -fPIC -march=native
PYTHON ?= python3
PYBIND = $(shell $(PYTHON) -m pybind11 --includes)
SUFFIX = $(shell $(PYTHON) -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX') or '')")
UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Darwin)
  LDFLAGS = -undefined dynamic_lookup
endif

# libtorch from pip-installed torch
TORCH_DIR = $(shell $(PYTHON) -c "import torch; import os; print(os.path.dirname(torch.__file__))")
TORCH_INCLUDES = -I$(TORCH_DIR)/include -I$(TORCH_DIR)/include/torch/csrc/api/include
TORCH_LIBS = -L$(TORCH_DIR)/lib -ltorch -ltorch_cpu -lc10 -ltorch_cuda -Wl,-rpath,$(TORCH_DIR)/lib

all: graygo_engine$(SUFFIX) mcts_engine_callback$(SUFFIX)

cpp-selfplay: graygo_engine$(SUFFIX) mcts_engine$(SUFFIX)

graygo_engine$(SUFFIX): engine.cpp
	$(CXX) $(CXXFLAGS) -shared $(PYBIND) $(LDFLAGS) $< -o $@

mcts_engine_callback$(SUFFIX): mcts_engine_callback.cpp
	$(CXX) $(CXXFLAGS) -shared $(PYBIND) $(LDFLAGS) $< -o $@

mcts_engine$(SUFFIX): mcts_engine.cpp
	$(CXX) $(CXXFLAGS) -shared $(PYBIND) $(TORCH_INCLUDES) $(LDFLAGS) $< $(TORCH_LIBS) -o $@

clean:
	rm -f graygo_engine$(SUFFIX) mcts_engine_callback$(SUFFIX) mcts_engine$(SUFFIX)

.PHONY: all cpp-selfplay clean
