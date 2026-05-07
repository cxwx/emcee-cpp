#pragma once

#include "types.hpp"
#include <vector>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <stdexcept>
#include <complex>

namespace emcee {
namespace autocorr {

// ============================================================================
// Autocorrelation time estimation (Sokal's windowed procedure)
//
// Ported from emcee Python: src/emcee/autocorr.py
// ============================================================================

namespace detail {

// Next power of 2 >= n
inline int next_pow2(int n) {
    int p = 1;
    while (p < n) p <<= 1;
    return p;
}

// FFT (Cooley-Tukey, in-place, radix-2 DIT)
// Expects input size to be a power of 2.
inline void fft(std::vector<std::complex<double>>& a, bool inverse) {
    int n = static_cast<int>(a.size());
    assert((n & (n - 1)) == 0);  // must be power of 2

    // Bit-reversal permutation
    for (int i = 1, j = 0; i < n; ++i) {
        int bit = n >> 1;
        for (; j & bit; bit >>= 1) j ^= bit;
        j ^= bit;
        if (i < j) std::swap(a[i], a[j]);
    }

    // Cooley-Tukey butterfly
    for (int len = 2; len <= n; len <<= 1) {
        double angle = (inverse ? 1.0 : -1.0) * 2.0 * M_PI / len;
        std::complex<double> wn(std::cos(angle), std::sin(angle));
        for (int i = 0; i < n; i += len) {
            std::complex<double> w(1.0, 0.0);
            for (int j = 0; j < len / 2; ++j) {
                auto u = a[i + j];
                auto v = a[i + j + len / 2] * w;
                a[i + j] = u + v;
                a[i + j + len / 2] = u - v;
                w *= wn;
            }
        }
    }

    if (inverse) {
        for (auto& x : a) x /= n;
    }
}

} // namespace detail

// Compute the normalized 1D autocorrelation function via FFT.
// Returns acf[0..N-1] where acf[0] = 1.0.
// NOTE: subtracts the mean before computing ACF (matches Python emcee behavior).
inline std::vector<double> function_1d(const double* x, int N) {
    int M = detail::next_pow2(2 * N);

    // Subtract mean
    double mean = 0;
    for (int i = 0; i < N; ++i) mean += x[i];
    mean /= N;

    // Zero-padded FFT
    std::vector<std::complex<double>> ft(M, 0.0);
    for (int i = 0; i < N; ++i) ft[i] = x[i] - mean;
    detail::fft(ft, false);

    // Power spectrum: |F|^2
    for (auto& c : ft) c = std::norm(c);

    // Inverse FFT → autocorrelation
    detail::fft(ft, true);

    // Normalize by acf[0]
    double norm = ft[0].real();
    if (norm == 0) return std::vector<double>(N, 0.0);

    std::vector<double> acf(N);
    for (int i = 0; i < N; ++i)
        acf[i] = ft[i].real() / norm;

    return acf;
}

// Auto-window selection: find m such that m >= c * tau[m]
inline int auto_window(const std::vector<double>& tau, double c) {
    for (int m = 0; m < static_cast<int>(tau.size()); ++m) {
        if (m >= c * tau[m])
            return m;
    }
    return static_cast<int>(tau.size()) - 1;
}

// Estimate integrated autocorrelation time for a 1D chain.
// x: chain values [N]
// c: window multiplier (default 5.0)
// Returns the estimated integrated autocorrelation time tau.
inline double integrated_time_1d(const double* x, int N, double c = 5.0) {
    auto acf = function_1d(x, N);

    // Cumulative sum: tau[m] = 2 * sum(acf[0..m]) - 1
    std::vector<double> tau(N);
    double cumsum = 0;
    for (int m = 0; m < N; ++m) {
        cumsum += acf[m];
        tau[m] = 2.0 * cumsum - 1.0;
    }

    int m = auto_window(tau, c);
    return tau[m];
}

// Estimate integrated autocorrelation time for each parameter dimension.
// chain: [nsteps x ndim] flat array
// nsteps, ndim: dimensions
// Returns vector of tau for each dimension.
inline std::vector<double> integrated_time(const double* chain,
                                            int nsteps, int ndim,
                                            double c = 5.0,
                                            int tol = 50) {
    std::vector<double> tau(ndim);

    for (int d = 0; d < ndim; ++d) {
        // Extract dimension d from row-major layout
        std::vector<double> x(nsteps);
        for (int t = 0; t < nsteps; ++t)
            x[t] = chain[t * ndim + d];

        tau[d] = integrated_time_1d(x.data(), nsteps, c);
    }

    return tau;
}

} // namespace autocorr
} // namespace emcee
