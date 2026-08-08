# ============================================================
# Chris Torch - AdaptiveCpp Backend
# ============================================================

CXX := /etc/acpp/bin/acpp

CXXFLAGS := -O3 \
            -std=c++17 \
            -fPIC \
            -Inative

# Directories
NATIVE_DIR := native
BENCH_DIR  := benchmarks
BUILD_DIR  := build
BENCH_BUILD_DIR := $(BUILD_DIR)/benchmarks

# Shared library used by Python
TARGET := $(BUILD_DIR)/libchristorch_acpp.so

# ------------------------------------------------------------
# Native sources
# ------------------------------------------------------------

# Automatically find all native C++ sources.
#
# Example:
# native/matmul.cpp
# native/relu.cpp
# native/softmax.cpp

SOURCES := $(shell find $(NATIVE_DIR) -name '*.cpp')

# ------------------------------------------------------------
# Benchmarks
# ------------------------------------------------------------

BENCH_SOURCES := $(shell find $(BENCH_DIR) -name '*.cpp' 2>/dev/null)

BENCH_TARGETS := $(patsubst \
	$(BENCH_DIR)/%.cpp,\
	$(BENCH_BUILD_DIR)/%,\
	$(BENCH_SOURCES))

# ------------------------------------------------------------
# Targets
# ------------------------------------------------------------

.PHONY: all clean rebuild benchmark

all: $(TARGET)

# ------------------------------------------------------------
# Shared library
# ------------------------------------------------------------

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(TARGET): $(SOURCES) | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) \
		-shared \
		$(SOURCES) \
		-o $(TARGET)

# ------------------------------------------------------------
# Benchmarks
# ------------------------------------------------------------

$(BENCH_BUILD_DIR)/%: $(BENCH_DIR)/%.cpp $(SOURCES)
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) \
		$< \
		$(SOURCES) \
		-o $@

benchmark: $(BENCH_TARGETS)
	@set -e; \
	for bench in $(BENCH_TARGETS); do \
		echo ""; \
		echo "Running $$bench"; \
		./$$bench; \
	done

# ------------------------------------------------------------
# Clean
# ------------------------------------------------------------

clean:
	rm -rf $(BUILD_DIR)

# ------------------------------------------------------------
# Rebuild
# ------------------------------------------------------------

rebuild: clean all