#pragma once

#include <vector>
#include <cmath>
#include <cassert>
#include <algorithm>
#include <numeric>
#include <random>
#include <functional>

namespace emcee {

// ============================================================================
// Minimal matrix class — row-major, only operations emcee actually needs
// ============================================================================
class Matrix {
public:
    Matrix() : rows_(0), cols_(0) {}
    Matrix(int rows, int cols, double val = 0.0)
        : rows_(rows), cols_(cols), data_(rows * cols, val) {}

    int rows() const { return rows_; }
    int cols() const { return cols_; }
    int size() const { return rows_ * cols_; }
    bool empty() const { return data_.empty(); }

    double* data() { return data_.data(); }
    const double* data() const { return data_.data(); }

    double& operator()(int i, int j) { return data_[i * cols_ + j]; }
    double  operator()(int i, int j) const { return data_[i * cols_ + j]; }

    // Row pointers (for passing to functions expecting double**)
    double* row(int i) { return data_.data() + i * cols_; }
    const double* row(int i) const { return data_.data() + i * cols_; }

    void fill(double val) { std::fill(data_.begin(), data_.end(), val); }
    void resize(int rows, int cols) {
        rows_ = rows; cols_ = cols;
        data_.resize(rows * cols);
    }

    // C = A^T * B  (A is m×k, B is n×k → C is m×n)
    static Matrix AtB(const Matrix& A, const Matrix& B) {
        assert(A.cols() == B.cols());
        Matrix C(A.rows(), B.rows());
        for (int i = 0; i < A.rows(); ++i)
            for (int j = 0; j < B.rows(); ++j) {
                double s = 0;
                for (int k = 0; k < A.cols(); ++k)
                    s += A(i, k) * B(j, k);
                C(i, j) = s;
            }
        return C;
    }

    // In-place scale: A *= alpha
    void scale(double alpha) {
        for (auto& v : data_) v *= alpha;
    }

    // Cholesky decomposition: A = L * L^T (lower triangular)
    // Returns false if not positive definite
    bool cholesky(Matrix& L) const {
        assert(rows_ == cols_);
        int n = rows_;
        L = Matrix(n, n, 0.0);
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j <= i; ++j) {
                double s = 0;
                for (int k = 0; k < j; ++k)
                    s += L(i, k) * L(j, k);
                if (i == j) {
                    double diag = (*this)(i, i) - s;
                    if (diag <= 0) return false;
                    L(i, j) = std::sqrt(diag);
                } else {
                    L(i, j) = ((*this)(i, j) - s) / L(j, j);
                }
            }
        }
        return true;
    }

private:
    int rows_, cols_;
    std::vector<double> data_;
};

// ============================================================================
// Vector operations on raw pointers (for row-level arithmetic)
// ============================================================================
namespace vec {

// dst[i] = a[i] + scale * b[i]
inline void add_scaled(double* dst, const double* a, double scale,
                       const double* b, int n) {
    for (int i = 0; i < n; ++i) dst[i] = a[i] + scale * b[i];
}

// dst[i] = a[i] - b[i]
inline void sub(double* dst, const double* a, const double* b, int n) {
    for (int i = 0; i < n; ++i) dst[i] = a[i] - b[i];
}

// dot product
inline double dot(const double* a, const double* b, int n) {
    double s = 0;
    for (int i = 0; i < n; ++i) s += a[i] * b[i];
    return s;
}

// squared norm
inline double norm2(const double* a, int n) { return dot(a, a, n); }

// norm
inline double norm(const double* a, int n) { return std::sqrt(norm2(a, n)); }

// dst = scale * src
inline void scale(double* dst, const double* src, double scale, int n) {
    for (int i = 0; i < n; ++i) dst[i] = src[i] * scale;
}

// dst[i] = a[i] * scale_a + b[i] * scale_b
inline void linear_comb(double* dst, const double* a, double sa,
                         const double* b, double sb, int n) {
    for (int i = 0; i < n; ++i) dst[i] = a[i] * sa + b[i] * sb;
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
