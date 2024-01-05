CC = clang
CXX = clang++
NVCC = nvcc

TARGET = nn
BIN = bin
INC_DIR = include
SRC_DIR = src
INCLUDES = -I $(INC_DIR)

CXXFLAGS = -Wall -MMD
CFLAGS = -Wall -MMD
NVCC_PREPEND_FLAGS = -ccbin $(CXX)
CUFLAGS = 
LDFLAGS = -L/opt/cuda/lib -lcuda -lcudart -lm

CPP_SRC = $(shell find src -type f -name "*.cpp")
C_SRC = $(shell find src -type f -name "*.c")
CUDA_SRC = $(shell find src -type f -name "*.cu")
SRC_OBJS = $(CPP_SRC:.cpp=.o) $(C_SRC:.c=.o) $(CUDA_SRC:.cu=.o)
OBJS = $(patsubst $(SRC_DIR)/%,$(BIN)/%,$(SRC_OBJS))

DEPENDS = ${TEST_OBJS:.o=.d} ${ARLG_OBJS:.o=.d} ${PLAT_OBJS:.o=.d}

all: $(TARGET)

$(BIN)/%.o: $(SRC_DIR)/%.cpp 
	$(CXX) -c $(CXXFLAGS) $(INCLUDES) -o $@ $<

$(BIN)/%.o: $(SRC_DIR)/%.c
	$(CC) -c $(CFLAGS) $(INCLUDES) -o $@ $<

$(BIN)/%.o: $(SRC_DIR)/%.cu
	$(NVCC) $(CUFLAGS) $(INCLUDES) -o $@ -c $<

$(TARGET): $(OBJS)
	$(CXX) -o $@ $^ $(LDFLAGS)

.PHONY : clean
clean :
	$(RM) $(OBJS)
	$(RM) $(TARGET)

-include ${DEPENDS}
