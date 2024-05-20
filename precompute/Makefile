all:
	g++ -O3 -Wall -shared -std=c++11 -pthread -lz -march=native -fPIC `python3 -m pybind11 --includes` push.cpp -o push.so
clean:
	rm push.so