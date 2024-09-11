CXX=g++
CXXFLAGS=-I. -shared -fPIC -std=c++11 -O2

all:
	$(CXX) -c ./*.cpp $(CXXFLAGS)
	$(CXX) -shared -Wl,-soname,libjpeg.so -o libjpeg.so  *.o
	$(CXX) -o decoder -std=c++11 -O2 *.cpp
	rm -f *.o


.PHONY: clean

clean:
	rm -f *.so
	rm -f *.o
