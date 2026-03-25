#include <chrono>
#include <iostream>

#pragma once

class Timer {
public:
    using clock = std::chrono::steady_clock;

    explicit Timer(std::string name = "")
        : name(std::move(name)),
          start(clock::now()) {}

    ~Timer() { stop(); }

    void stop() {
        auto end = clock::now();
        auto duration = end - start;

        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(duration).count();

        std::cout << name << ms << " ms\n";
    }


private:
    std::string name;
    clock::time_point start;
};

