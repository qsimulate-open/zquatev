//
// ZQUATEV: Diagonalization of quaternionic matrices
// Copyright (c) 2013, Toru Shiozaki (shiozaki@northwestern.edu)
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice, this
//    list of conditions and the following disclaimer.
// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
// ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
// WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
// ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
// (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
// LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
// ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// The views and conclusions contained in the software and documentation are those
// of the authors and should not be interpreted as representing official policies,
// either expressed or implied, of the FreeBSD Project.
//

#ifndef __SRC_UTIL_F77_H
#define __SRC_UTIL_F77_H

#include <complex>
#include <memory>

// blas
extern "C" {

 void daxpy_(const int*, const double*, const double*, const int*, double*, const int*);
 void dsyev_(const char*, const char*, const int*, double*, const int*, double*, double*, const int*, int*);
 void dgesv_(const int* n, const int* nrhs, double* a, const int* lda, int* ipiv, double* b, const int* ldb, int* info);
 void dscal_(const int*, const double*, double*, const int*);
 double ddot_(const int*, const double*, const int*, const double*, const int*);
 void dgemv_(const char*, const int*, const int*, const double*, const double*, const int*, const double*, const int*,
             const double*, double*, const int*);
 void dgemm_(const char* transa, const char* transb, const int* m, const int* n, const int* k,
             const double* alpha, const double* a, const int* lda, const double* b, const int* ldb,
             const double* beta, double* c, const int* ldc);
 void dsysv_(const char* uplo, const int* n, const int* nrhs, double* a, const int* lda, int* ipiv,
             double* b, const int* ldb, double* work, const int* lwork, int* info);
 void drot_(const int*, const double*, const int*, const double*, const int*, const double*, const double*);
 void dger_(const int*, const int*, const double*, const double*, const int*, const double*, const int*, double*, const int*);

 void zcopy_(const int*, const std::complex<double>*, const int*, std::complex<double>*, const int*);
 void zscal_(const int*, const std::complex<double>*, std::complex<double>*, const int*);
 void zdotc_(std::complex<double>*, const int*, const std::complex<double>*, const int*, const std::complex<double>*, const int*);
 void zaxpy_(const int*, const std::complex<double>*, const std::complex<double>*, const int*, std::complex<double>*, const int*);
 void zgemv_(const char*, const int*, const int*, const std::complex<double>*, const std::complex<double>*, const int*, const std::complex<double>*, const int*,
             const std::complex<double>*, std::complex<double>*, const int*);
#ifdef MKL
 void zgemm3m_(const char* transa, const char* transb, const int* m, const int* n, const int* k,
               const std::complex<double>* alpha, const std::complex<double>* a, const int* lda, const std::complex<double>* b, const int* ldb,
               const std::complex<double>* beta, std::complex<double>* c, const int* ldc);
#else
 void zgemm_(const char* transa, const char* transb, const int* m, const int* n, const int* k,
             const std::complex<double>* alpha, const std::complex<double>* a, const int* lda, const std::complex<double>* b, const int* ldb,
             const std::complex<double>* beta, std::complex<double>* c, const int* ldc);
#endif
 void zhbev_(const char*, const char*, const int*, const int*, std::complex<double>*, const int*, double*, std::complex<double>*, const int*,
             std::complex<double>*, double*, int*);
 void zrot_(const int*, std::complex<double>*, const int*, std::complex<double>*, const int*, const double*, const std::complex<double>*);
 void zgerc_(const int*, const int*, const std::complex<double>*, const std::complex<double>*, const int*, const std::complex<double>*, const int*,
             std::complex<double>*, const int*);
 void zgeru_(const int*, const int*, const std::complex<double>*, const std::complex<double>*, const int*, const std::complex<double>*, const int*,
             std::complex<double>*, const int*);
}


// lapack
extern "C" {
 void zgeev_(const char*, const char*, const int*, std::complex<double>*, const int*, std::complex<double>*,
             std::complex<double>*, const int*, std::complex<double>*, const int*, std::complex<double>*, const int*, double*, int*);
 void zheev_(const char*, const char*, const int*, std::complex<double>*, const int*, double*, std::complex<double>*, const int*, double*, int*);
 void zhesv_(const char* uplo, const int* n, const int* nrhs, std::complex<double>* a, const int* lda, int* ipiv,
             std::complex<double>* b, const int* ldb, std::complex<double>* work, const int* lwork, int* info);
 void zgesv_(const int* n, const int* nrhs, std::complex<double>* a, const int* lda, int* ipiv,
             std::complex<double>* b, const int* ldb, int* info);
 void dgesvd_(const char*, const char*, const int*, const int*, double*, const int*, double*, double*, const int*, double*, const int*,  double*, const int*, int*);
 void zgesvd_(const char*, const char*, const int*, const int*, std::complex<double>*, const int*, double*,
              std::complex<double>*, const int*, std::complex<double>*, const int*,  std::complex<double>*, const int*, double*, int*);
 void zgesdd_(const char*, const int*, const int*, std::complex<double>*, const int*, double*,
              std::complex<double>*, const int*, std::complex<double>*, const int*,  std::complex<double>*, const int*, double*, int*, int*);
 void zlartg_(const std::complex<double>*, const std::complex<double>*, double*, std::complex<double>*, std::complex<double>*);
 void zlarfg_(const int*, const std::complex<double>*, std::complex<double>*, const int*, std::complex<double>*);
}


namespace {

 void dgemm_(const char* transa, const char* transb, const int m, const int n, const int k,
             const double alpha, const double* a, const int lda, const double* b, const int ldb,
             const double beta, double* c, const int ldc) { ::dgemm_(transa,transb,&m,&n,&k,&alpha,a,&lda,b,&ldb,&beta,c,&ldc); }
 void dgemm_(const char* transa, const char* transb, const int m, const int n, const int k,
             const double alpha, const std::unique_ptr<double []>& a, const int lda, const std::unique_ptr<double []>& b, const int ldb,
             const double beta, std::unique_ptr<double []>& c, const int ldc)
             { ::dgemm_(transa,transb,&m,&n,&k,&alpha,a.get(),&lda,b.get(),&ldb,&beta,c.get(),&ldc); }
 void dgemv_(const char* a, const int b, const int c, const double d, const double* e, const int f, const double* g, const int h,
             const double i, double* j, const int k) { ::dgemv_(a,&b,&c,&d,e,&f,g,&h,&i,j,&k); }
 void dgemv_(const char* a, const int b, const int c, const double d, const std::unique_ptr<double []>& e, const int f,
             const std::unique_ptr<double []>& g, const int h, const double i, std::unique_ptr<double []>& j, const int k)
             { ::dgemv_(a,&b,&c,&d,e.get(),&f,g.get(),&h,&i,j.get(),&k); }
 void daxpy_(const int a, const double b, const double* c, const int d, double* e, const int f) { ::daxpy_(&a,&b,c,&d,e,&f); }
 void daxpy_(const int a, const double b, const std::unique_ptr<double []>& c, const int d, std::unique_ptr<double []>& e, const int f)
             { ::daxpy_(&a,&b,c.get(),&d,e.get(),&f); }
 void dscal_(const int a, const double b, double* c, const int d) { ::dscal_(&a, &b, c, &d); }
 void dscal_(const int a, const double b, std::unique_ptr<double []>& c, const int d) { ::dscal_(&a, &b, c.get(), &d); }
 double ddot_(const int a, const double* b, const int c, const double* d, const int e) { return ::ddot_(&a,b,&c,d,&e); }
 double ddot_(const int a, const std::unique_ptr<double []>& b, const int c, const std::unique_ptr<double []>& d, const int e)
             { return ::ddot_(&a,b.get(),&c,d.get(),&e); }
 void dsyev_(const char* a, const char* b, const int c, double* d, const int e, double* f, double* g, const int h, int& i)
             { ::dsyev_(a,b,&c,d,&e,f,g,&h,&i);}
 void dsyev_(const char* a, const char* b, const int c, std::unique_ptr<double []>& d, const int e,
             std::unique_ptr<double []>& f, std::unique_ptr<double []>& g, const int h, int& i)
             { ::dsyev_(a,b,&c,d.get(),&e,f.get(),g.get(),&h,&i);}
 void dsysv_(const char* uplo, const int n, const int nrhs, double* a, const int lda, int* ipiv,
             double* b, const int ldb, double* work, const int lwork, int& info)
             { ::dsysv_(uplo, &n, &nrhs, a, &lda, ipiv, b, &ldb, work, &lwork, &info);}
 void dsysv_(const char* uplo, const int n, const int nrhs, std::unique_ptr<double[]>& a, const int lda, std::unique_ptr<int[]>& ipiv,
             std::unique_ptr<double[]>& b, const int ldb, std::unique_ptr<double[]>& work, const int lwork, int& info)
             { ::dsysv_(uplo, &n, &nrhs, a.get(), &lda, ipiv.get(), b.get(), &ldb, work.get(), &lwork, &info);}
 void dgesv_(const int n, const int nrhs, double* a, const int lda, int* ipiv, double* b, const int ldb, int& info)
             { ::dgesv_(&n, &nrhs, a, &lda, ipiv, b, &ldb, &info); }
 void dgesv_(const int n, const int nrhs, std::unique_ptr<double[]>& a, const int lda, std::unique_ptr<int[]>& ipiv,
             std::unique_ptr<double[]>& b, const int ldb, int& info) { ::dgesv_(&n, &nrhs, a.get(), &lda, ipiv.get(), b.get(), &ldb, &info); }
 void dger_(const int a, const int b, const double c, const double* d, const int e, const double* f, const int g, double* h, const int i)
            { ::dger_(&a, &b, &c, d, &e, f, &g, h, &i); }
 void dger_(const int a, const int b, const double c, const std::unique_ptr<double[]>& d, const int e, const std::unique_ptr<double[]>& f,
            const int g, std::unique_ptr<double[]>& h, const int i)
            { ::dger_(&a, &b, &c, d.get(), &e, f.get(), &g, h.get(), &i); }
 void drot_(const int a, const double* b, const int c, const double* d, const int e, const double f, const double g) { ::drot_(&a, b, &c, d, &e, &f, &g); }
 void drot_(const int a, std::unique_ptr<double[]> b, const int c, std::unique_ptr<double[]> d, const int e, const double f, const double g) { ::drot_(&a, b.get(), &c, d.get(), &e, &f, &g); }
 void dgesvd_(const char* a, const char* b, const int c, const int d, double* e, const int f, double* g, double* h, const int i, double* j, const int k,
              double* l, const int m, int& n) { ::dgesvd_(a,b,&c,&d,e,&f,g,h,&i,j,&k,l,&m,&n); }
 void zgesvd_(const char* a, const char* b, const int c, const int d, std::complex<double>* e, const int f, double* g, std::complex<double>* h,
              const int i, std::complex<double>* j, const int k, std::complex<double>* l, const int m, double* n, int& o) { ::zgesvd_(a,b,&c,&d,e,&f,g,h,&i,j,&k,l,&m,n,&o); }

 void zgemv_(const char* a, const int b, const int c, const std::complex<double> d, const std::complex<double>* e, const int f, const std::complex<double>* g, const int h,
             const std::complex<double> i, std::complex<double>* j, const int k) { ::zgemv_(a,&b,&c,&d,e,&f,g,&h,&i,j,&k); }
 void zgemv_(const char* a, const int b, const int c, const std::complex<double> d, const std::unique_ptr<std::complex<double> []>& e, const int f,
             const std::unique_ptr<std::complex<double> []>& g, const int h, const std::complex<double> i, std::unique_ptr<std::complex<double> []>& j, const int k)
             { ::zgemv_(a,&b,&c,&d,e.get(),&f,g.get(),&h,&i,j.get(),&k); }
#ifdef MKL
 void zgemm3m_(const char* transa, const char* transb, const int m, const int n, const int k,
               const std::complex<double> alpha, const std::complex<double>* a, const int lda, const std::complex<double>* b, const int ldb,
               const std::complex<double> beta, std::complex<double>* c, const int ldc) { ::zgemm3m_(transa,transb,&m,&n,&k,&alpha,a,&lda,b,&ldb,&beta,c,&ldc); }
 void zgemm3m_(const char* transa, const char* transb, const int m, const int n, const int k,
               const std::complex<double> alpha, const std::unique_ptr<std::complex<double>[]>& a, const int lda,
               const std::unique_ptr<std::complex<double>[]>& b, const int ldb,
               const std::complex<double> beta, std::unique_ptr<std::complex<double>[]>& c, const int ldc)
               { ::zgemm3m_(transa,transb,&m,&n,&k,&alpha,a.get(),&lda,b.get(),&ldb,&beta,c.get(),&ldc); }
#else
 void zgemm_(const char* transa, const char* transb, const int m, const int n, const int k,
             const std::complex<double> alpha, const std::complex<double>* a, const int lda, const std::complex<double>* b, const int ldb,
             const std::complex<double> beta, std::complex<double>* c, const int ldc) { ::zgemm_(transa,transb,&m,&n,&k,&alpha,a,&lda,b,&ldb,&beta,c,&ldc); }
 void zgemm_(const char* transa, const char* transb, const int m, const int n, const int k,
             const std::complex<double> alpha, const std::unique_ptr<std::complex<double>[]>& a, const int lda,
             const std::unique_ptr<std::complex<double>[]>& b, const int ldb,
             const std::complex<double> beta, std::unique_ptr<std::complex<double>[]>& c, const int ldc)
             { ::zgemm_(transa,transb,&m,&n,&k,&alpha,a.get(),&lda,b.get(),&ldb,&beta,c.get(),&ldc); }
#endif

 void zaxpy_(const int a, const std::complex<double> b, const std::complex<double>* c, const int d, std::complex<double>* e, const int f) { ::zaxpy_(&a,&b,c,&d,e,&f); }
 void zaxpy_(const int a, const std::complex<double> b, const std::unique_ptr<std::complex<double>[]>& c, const int d, std::unique_ptr<std::complex<double>[]>& e,
             const int f) { ::zaxpy_(&a,&b,c.get(),&d,e.get(),&f); }
 void zscal_(const int a, const std::complex<double> b, std::complex<double>* c, const int d) { ::zscal_(&a, &b, c, &d); }
 void zscal_(const int a, const std::complex<double> b, std::unique_ptr<std::complex<double> []>& c, const int d) { ::zscal_(&a, &b, c.get(), &d); }

 std::complex<double> zdotc_(const int b, const std::complex<double>* c, const int d, const std::complex<double>* e, const int f) {
   std::complex<double> a;
   ::zdotc_(&a,&b,c,&d,e,&f);
   return a;
 }
 std::complex<double> zdotc_(const int b, const std::unique_ptr<std::complex<double> []>& c, const int d, const std::unique_ptr<std::complex<double> []>& e, const int f) {
   std::complex<double> a;
   ::zdotc_(&a,&b,c.get(),&d,e.get(),&f);
   return a;
 }
 void zgesv_(const int n, const int nrhs, std::complex<double>* a, const int lda, int* ipiv, std::complex<double>* b, const int ldb, int& info)
             { ::zgesv_(&n, &nrhs, a, &lda, ipiv, b, &ldb, &info); }
 void zgesv_(const int n, const int nrhs, std::unique_ptr<std::complex<double>[]>& a, const int lda, std::unique_ptr<int[]>& ipiv,
             std::unique_ptr<std::complex<double>[]>& b, const int ldb, int& info) { ::zgesv_(&n, &nrhs, a.get(), &lda, ipiv.get(), b.get(), &ldb, &info); }
 void zheev_(const char* a, const char* b, const int c, std::complex<double>* d, const int e, double* f, std::complex<double>* g, const int h, double* i, int& j)
             { ::zheev_(a,b,&c,d,&e,f,g,&h,i,&j); }
 void zheev_(const char* a, const char* b, const int c, std::unique_ptr<std::complex<double> []>& d, const int e,
             std::unique_ptr<double []>& f, std::unique_ptr<std::complex<double> []>& g, const int h, std::unique_ptr<double[]>& i, int& j)
             { ::zheev_(a,b,&c,d.get(),&e,f.get(),g.get(),&h,i.get(),&j); }
 void zgeev_(const char* a, const char* b, const int c, std::complex<double>* d, const int e, std::complex<double>* f,
             std::complex<double>* g, const int h, std::complex<double>* i, const int j, std::complex<double>* k, const int l, double* m, int& n)
             { ::zgeev_(a,b,&c,d,&e,f,g,&h,i,&j,k,&l,m,&n); }
 void zhbev_(const char* a, const char* b, const int c, const int d, std::complex<double>* e, const int f, double* g, std::complex<double>* h, const int i,
             std::complex<double>* j, double* k, int& l) { ::zhbev_(a, b, &c, &d, e, &f, g, h, &i, j, k, &l); }
 void zrot_(const int a, std::complex<double>* b, const int c, std::complex<double>* d, const int e, const double f, const std::complex<double> g) {
            ::zrot_(&a, b, &c, d, &e, &f, &g); }
 void zgerc_(const int a, const int b, const std::complex<double> c, const std::complex<double>* d, const int e, const std::complex<double>* f, const int g,
             std::complex<double>* h, const int i) { ::zgerc_(&a, &b, &c, d, &e, f, &g, h, &i); }
 void zgeru_(const int a, const int b, const std::complex<double> c, const std::complex<double>* d, const int e, const std::complex<double>* f, const int g,
             std::complex<double>* h, const int i) { ::zgeru_(&a, &b, &c, d, &e, f, &g, h, &i); }
 void zlartg_(const std::complex<double> a, const std::complex<double> b, double& c, std::complex<double>& d, std::complex<double>& e) { ::zlartg_(&a, &b, &c, &d, &e); }
 void zlarfg_(const int a, std::complex<double>& b, std::complex<double>* c, const int d, std::complex<double>& e) { ::zlarfg_(&a, &b, c, &d, &e); }

}

#endif
