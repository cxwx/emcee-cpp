#pragma once

#include <eigen3/Eigen/Dense>
#include <vector>
#include <cmath>
#include <cassert>
#include <random>
#include <functional>

namespace emcee {

// ============================================================================
// Matrix type — Eigen row-major dynamic matrix
// ============================================================================
using Matrix = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using Vector = Eigen::VectorXd;

// ============================================================================
// Vector operations on raw pointers (for row-level arithmetic)
// ============================================================================
namespace vec {

// dst[i] = a[i] + scale * b[i]
inline void add_scaled(double* dst, const double* a, double s,
                       const double* b, int n) {
    Eigen::Map<Eigen::VectorXd>(dst, n) =
        Eigen::Map<const Eigen::VectorXd>(a, n) +
        s * Eigen::Map<const Eigen::VectorXd>(b, n);
}

// dst[i] = a[i] - b[i]
inline void sub(double* dst, const double* a, const double* b, int n) {
    Eigen::Map<Eigen::VectorXd>(dst, n) =
        Eigen::Map<const Eigen::VectorXd>(a, n) -
        Eigen::Map<const Eigen::VectorXd>(b, n);
}

// dot product
inline double dot(const double* a, const double* b, int n) {
    return Eigen::Map<const Eigen::VectorXd>(a, n).dot(
           Eigen::Map<const Eigen::VectorXd>(b, n));
}

// squared norm
inline double norm2(const double* a, int n) {
    return Eigen::Map<const Eigen::VectorXd>(a, n).squaredNorm();
}

// norm
inline double norm(const double* a, int n) {
    return Eigen::Map<const Eigen::VectorXd>(a, n).norm();
}

// dst = scale * src
inline void scale(double* dst, const double* src, double s, int n) {
    Eigen::Map<Eigen::VectorXd>(dst, n) =
        s * Eigen::Map<const Eigen::VectorXd>(src, n);
}

// dst[i] = a[i] * sa + b[i] * sb
inline void linear_comb(double* dst, const double* a, double sa,
                        const double* b, double sb, int n) {
    Eigen::Map<Eigen::VectorXd>(dst, n) =
        sa * Eigen::Map<const Eigen::VectorXd>(a, n) +
        sb * Eigen::Map<const Eigen::VectorXd>(b, n);
}

} // namespace vec

// ============================================================================
// Random number engine type alias
// ============================================================================
using RNG = std::mt19937;

// ============================================================================
// Log-probability function type
// ============================================================================
using LogProbFn = std::function<double(const double* params, int ndim)>;

} // namespace emcee
