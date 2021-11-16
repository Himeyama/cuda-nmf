/*
 *  (c) 2021 Murata Mitsuharu
 *  Licensed under the MIT License.
 */
#include <iostream>
#include <cstdlib>
#include <cublas.h>
#include <nmf-hals.hpp>
#include <ruby.h>
#include <numo/narray.h>

template <typename T>
VALUE rb_nmf_NMF(VALUE self, VALUE X, VALUE W, VALUE H, VALUE Y, VALUE rb_m, VALUE rb_n, VALUE rb_k, VALUE rb_eps){
    long m = NUM2LONG(rb_m), n = NUM2LONG(rb_n), k = NUM2LONG(rb_k);
    T eps = NUM2DBL(rb_eps);
    T *data = (T*)na_get_pointer(X);
    T rms;
    NMF<T> nmf(m, n, k, data, &rms, eps);
    rb_iv_set(self, "@rms", DBL2NUM(rms));
    T *w = (T*)na_get_pointer(W);
    T *h = (T*)na_get_pointer(H);
    T *y = (T*)na_get_pointer(Y);
    memcpy(w, nmf.W, sizeof(T) * m * k);
    memcpy(h, nmf.H, sizeof(T) * k * n);
    memcpy(y, nmf.Y, sizeof(T) * m * n);
    return self;
}

extern "C" {
    void Init_nmf() {
        VALUE rb_cCuda = rb_define_module("Cuda");
        VALUE rb_cCudaNMF = rb_define_module_under(rb_cCuda, "NMF");
        VALUE rb_cNMF = rb_define_class_under(rb_cCudaNMF, "NMF", rb_cObject);
        rb_define_method(rb_cNMF, "_NMF", rb_nmf_NMF<double>, 8);
    }
}