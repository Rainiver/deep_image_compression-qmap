CXX=g++
CXXFLAGS=-I. -shared -fPIC -std=c++11 -O2

all:
	$(CXX) -c ./*.cpp $(CXXFLAGS)
	$(CXX) -shared -Wl,-soname,libae.so -o libae.so  *.o
	rm -f *.o


.PHONY: clean

clean:
	rm -f *.so
	rm -f *.o
