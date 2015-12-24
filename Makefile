INCLUDE  = -I.
#INCLUDE += -DMKL -I/opt/intel/composer_xe_2013/mkl/include/
LIBS     = -framework Accelerate
#LIBS     = -lmkl_intel_lp64 -lmkl_core -lmkl_intel_thread
#LIBS     = -mkl=parallel # works with recent Intel compilers
#LIBS    += -lm -lpthread
CXX      = g++
CXXFLAGS = -O3 -std=c++11 $(INCLUDE)
LD       = $(CXX)
LDFLAGS  = $(CXXFLAGS) $(LIBS)

all: test.x

test.x: unblocked.o test.o
#test.x: blocked.o test.o
	$(LD) $(LDFLAGS) $^ -o $@

%.o: %.cc
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	-rm -f *.o test.x
