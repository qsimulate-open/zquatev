INCLUDE  = -I.
INCLUDE += -DHAVE_ZGEMM3M
LIBS     = -lmkl_intel_lp64 -lmkl_core -lmkl_intel_thread -lm -lpthread

#INCLUDE += -DZDOT_RETURN # if your BLAS' ZDOTC returns 
#LIBS     = -framework Accelerate
#LIBS     = -mkl=parallel # works with recent Intel compilers

CXX      = g++
CXXFLAGS = -O3 -DNDEBUG -std=c++11 $(INCLUDE)
#CXXFLAGS = -O0 -g -std=c++11 $(INCLUDE)

LD       = $(CXX)
LDFLAGS  = $(CXXFLAGS) $(LIBS)

all: test.x

test.x: zquatev.o blocked.o unblocked.o transpose.o test.o
	$(LD) $(LDFLAGS) $^ -o $@

%.o: %.cc
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	-rm -f *.o test.x
