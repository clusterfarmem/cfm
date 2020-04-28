#include <iostream>
#include <vector>
#include <algorithm>
#include <cstdlib>
#include <chrono>

const size_t MB = 1024 * 1024;
using namespace std::chrono;

void die(const char *msg, bool printErrno) {
	std::cerr << msg << "\n";
	exit(1);
}

void print_time_diff(time_point<high_resolution_clock> start,
	time_point<high_resolution_clock> end) {
	auto diff = end - start;
	std::cout << "time " << duration<double, std::nano> (diff).count() << "\n";
}

void print_time_diff_ms(time_point<high_resolution_clock> start,
	time_point<high_resolution_clock> end) {
	auto diff = end - start;
	std::cout << "time " << duration<double, std::milli> (diff).count() << " ms\n";
}

int main(int argc, char *argv[]) {
	if (argc != 2)
		die("need MB of integers to sort", false);

	long size = std::stoi(argv[1]) * MB;
	long numInts = size / sizeof(int);

	std::cout << "will sort " << numInts << " integers (" << size / MB << " MB)\n";
	std::vector<int> v(numInts);

	std::srand(std::time(0));
	time_point<high_resolution_clock> start, end;

	std::generate(v.begin(), v.end(), std::rand);
	start = high_resolution_clock::now();
	std::sort(v.begin(), v.end(), std::greater<int>());

	end = high_resolution_clock::now();
	print_time_diff_ms(start, end);

	return 0;
}
