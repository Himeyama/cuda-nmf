#ifndef CUDA_MATRIX_HPP
#define CUDA_MATRIX_HPP

#include <cublas.h>
#include <cuda-matrix.hpp>
#include <iostream>

template <class T>
class NMF{
    public:
    long n_samples, n_components, n_features;
    T *W, *H, *Y;

    void cublasCopy(int n, const float *x, int incx, float *y, int incy);

    void cublasCopy(int n, const double *x, int incx, double *y, int incy);

    NMF(long m, long n, long k, T *dat, T* rms, T* ss, T eps = 1e-4);
};

#endif
