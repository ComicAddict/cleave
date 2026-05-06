# ─────────────────────────────────────────────────────────────────────────────
# Cleave — Makefile
# ─────────────────────────────────────────────────────────────────────────────

# Installation prefix — override with: make install PREFIX=/usr/local
PREFIX     ?= /usr/local
BINDIR     := $(PREFIX)/bin

TARGET     := cleave
SRC        := divisive_cluster.cpp

CXX        ?= g++
CXXFLAGS   := -O3 -march=native -mavx2 -mfma -std=c++17 -fopenmp
LDFLAGS    := -lpthread

# ── Eigen include path ────────────────────────────────────────────────────────
# Try pkg-config first, fall back to common system paths.
EIGEN_CFLAGS := $(shell pkg-config --cflags eigen3 2>/dev/null)
ifeq ($(EIGEN_CFLAGS),)
    # pkg-config not available — try common locations
    ifneq ($(wildcard /usr/include/eigen3),)
        EIGEN_CFLAGS := -I/usr/include/eigen3
    else ifneq ($(wildcard /usr/local/include/eigen3),)
        EIGEN_CFLAGS := -I/usr/local/include/eigen3
    else ifneq ($(wildcard /opt/homebrew/include/eigen3),)
        EIGEN_CFLAGS := -I/opt/homebrew/include/eigen3   # macOS Homebrew
    else
        $(warning Could not find Eigen3. Set EIGEN_CFLAGS manually: make EIGEN_CFLAGS=-I/path/to/eigen3)
    endif
endif

CXXFLAGS += $(EIGEN_CFLAGS)

# ── Optional mlpack benchmark variant ────────────────────────────────────────
# Enable with: make MLPACK=1
ifdef MLPACK
    MLPACK_CFLAGS  := $(shell pkg-config --cflags mlpack 2>/dev/null) \
                      -I/usr/include/stb -I/usr/include -DWITH_MLPACK
    MLPACK_LDFLAGS := -larmadillo
    CXXFLAGS       += $(MLPACK_CFLAGS)
    LDFLAGS        += $(MLPACK_LDFLAGS)
endif

# ─────────────────────────────────────────────────────────────────────────────
# Targets
# ─────────────────────────────────────────────────────────────────────────────

.PHONY: all install uninstall clean help

all: $(TARGET)

$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) $< -o $@ $(LDFLAGS)

# ── Install ───────────────────────────────────────────────────────────────────
install: $(TARGET)
	@echo "Installing $(TARGET) to $(BINDIR)"
	@mkdir -p $(BINDIR)
	@install -m 755 $(TARGET) $(BINDIR)/$(TARGET)
	@echo "Done. Run '$(TARGET) --help' to get started."

# ── Uninstall ─────────────────────────────────────────────────────────────────
uninstall:
	@echo "Removing $(BINDIR)/$(TARGET)"
	@rm -f $(BINDIR)/$(TARGET)
	@echo "Done."

# ── Clean ─────────────────────────────────────────────────────────────────────
clean:
	rm -f $(TARGET)

# ── Help ─────────────────────────────────────────────────────────────────────
help:
	@echo ""
	@echo "Cleave — build targets"
	@echo ""
	@echo "  make              Build $(TARGET) in the current directory"
	@echo "  make install      Install to $(BINDIR)  (override: make install PREFIX=/usr)"
	@echo "  make uninstall    Remove from $(BINDIR)"
	@echo "  make clean        Remove the local build"
	@echo "  make MLPACK=1     Build with mlpack benchmark variant"
	@echo ""
	@echo "Examples:"
	@echo "  make"
	@echo "  make install"
	@echo "  sudo make install PREFIX=/usr/local"
	@echo "  make install PREFIX=~/.local"
	@echo "  make MLPACK=1 && sudo make install MLPACK=1"
	@echo ""