#pragma once

#include <vector>
#include <queue>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <functional>
#include <future>
#include <atomic>

namespace emcee {

// ============================================================================
// Simple thread pool for parallel log_prob evaluation
// ============================================================================
class ThreadPool {
public:
    explicit ThreadPool(int nthreads) : stop_(false) {
        if (nthreads <= 0)
            nthreads = std::max(1u, std::thread::hardware_concurrency());
        for (int i = 0; i < nthreads; ++i)
            workers_.emplace_back([this] { worker_loop(); });
    }

    ~ThreadPool() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            stop_ = true;
        }
        cv_.notify_all();
        for (auto& t : workers_) t.join();
    }

    // Submit a task and get a future
    template<typename F>
    auto submit(F&& f) -> std::future<decltype(f())> {
        using R = decltype(f());
        auto task = std::make_shared<std::packaged_task<R()>>(std::forward<F>(f));
        auto future = task->get_future();
        {
            std::lock_guard<std::mutex> lock(mutex_);
            tasks_.emplace([task]() { (*task)(); });
        }
        cv_.notify_one();
        return future;
    }

    // Map a function over a range of indices, blocking until all done.
    // f(i) is called for i in [0, count).
    void map(std::function<void(int)> f, int count) {
        std::vector<std::future<void>> futures;
        futures.reserve(count);
        for (int i = 0; i < count; ++i)
            futures.push_back(submit([f, i]() { f(i); }));
        for (auto& fut : futures) fut.get();
    }

    int size() const { return static_cast<int>(workers_.size()); }

private:
    void worker_loop() {
        for (;;) {
            std::function<void()> task;
            {
                std::unique_lock<std::mutex> lock(mutex_);
                cv_.wait(lock, [this] { return stop_ || !tasks_.empty(); });
                if (stop_ && tasks_.empty()) return;
                task = std::move(tasks_.front());
                tasks_.pop();
            }
            task();
        }
    }

    std::vector<std::thread> workers_;
    std::queue<std::function<void()>> tasks_;
    std::mutex mutex_;
    std::condition_variable cv_;
    bool stop_;
};

} // namespace emcee
