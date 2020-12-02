/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once


#include <chrono>
#include <iostream>
#include <string>
#include <vector>

#include <cuda_runtime.h>
/**
 * A very simple C++11 timer.
 */
class TimerBase
{
   public:
    // 64-bit nanoseconds
    using Time = std::chrono::duration<int64_t, std::nano>;

    TimerBase() { start(); }

    void start() { startTime = std::chrono::high_resolution_clock::now(); }

    Time stop()
    {
        auto endTime = std::chrono::high_resolution_clock::now();
        auto elapsed = endTime - startTime;
        lastTime     = std::chrono::duration_cast<Time>(elapsed);
        return lastTime;
    }


    double getTimeMicrS()
    {
        return std::chrono::duration_cast<std::chrono::duration<double, std::micro>>(lastTime).count();
    }

    double getTimeMS()
    {
        return std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(lastTime).count();
    }
    Time getTime() { return lastTime; }

   protected:
    Time lastTime = Time(0);
    std::chrono::time_point<std::chrono::high_resolution_clock> startTime;
};


template <typename T = float, typename Unit = std::chrono::milliseconds>
class ScopedTimer : public TimerBase
{
    static_assert(std::is_arithmetic<T>::value, "T must be arithmetic");

   public:
    T* target;
    explicit ScopedTimer(T* target) : target(target) { start(); }

    explicit ScopedTimer(T& target) : target(&target) { start(); }

    ScopedTimer(ScopedTimer&& other) noexcept : target(other.target) {}
    ~ScopedTimer()
    {
        T time  = std::chrono::duration_cast<std::chrono::duration<T, typename Unit::period>>(stop()).count();
        *target = time;
    }
};

class CudaScopedTimer
{
   public:
    CudaScopedTimer(float& time, cudaStream_t stream = 0) : time(time), stream(stream)
    {
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        cudaEventRecord(start, stream);
    }
    ~CudaScopedTimer()
    {
        cudaEventRecord(stop, stream);
        cudaEventSynchronize(stop);


        cudaEventElapsedTime(&time, start, stop);
    }

   private:
    float& time;
    cudaEvent_t start, stop;
    cudaStream_t stream;
};


template <typename TimerType = CudaScopedTimer, typename F>
inline float measureObject(int its, F f)
{
    std::vector<float> timings(its);
    for (int i = 0; i < its; ++i)
    {
        float time;
        {
            TimerType tim(time);
            f();
        }
        timings[i] = time;
    };

    std::sort(timings.begin(), timings.end());
    return timings[timings.size() / 2];
}
