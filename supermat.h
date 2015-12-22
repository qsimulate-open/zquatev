//
// Copyright (c) 2016, Toru Shiozaki (shiozaki@northwestern.edu)
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


#ifndef __SUPERMATRIX_H
#define __SUPERMATRIX_H

#include <complex>
#include <iostream>
#include <iomanip>
#include <array>
#include "f77.h"

template <size_t NB, size_t MB>
class SuperMatrix {
  protected:
    // raw pointer. Efficiency is everything!
    std::complex<double>* data_;

    // size information
    const int nmax_;
    const int mmax_;
    int area() const { return nmax_*mmax_; }

    // size pointer
    std::array<int, NB> nptr_;
    std::array<int, MB> mptr_;

  private:
    void conj_n(std::complex<double>* p, const size_t n) {
      double* dp = reinterpret_cast<double*>(p) + 1;
      for (double* i = dp; i <= dp + 2*n-2; i += 2) *i = -*i;
    }

  public:
    SuperMatrix(std::complex<double>* d, const int nm, const int mm, const int nst = 1, const int mst = 1)
        : data_(d), nmax_(nm), mmax_(mm) {
      std::fill(nptr_.begin(), nptr_.end(), nst);
      std::fill(mptr_.begin(), mptr_.end(), mst);
      std::fill_n(data_, nmax_*mmax_*NB*MB, 0.0);
    }

    template<size_t I, size_t J, class = typename std::enable_if<(I < NB and J < MB)>::type>
    std::complex<double>& data(const int n, const int m) { return *(block(I, J) + n + nmax_*m); }

    std::complex<double>* block(const int i, const int j) { return data_+area()*(i+NB*j); }
    const std::complex<double>* block(const int i, const int j) const { return data_+area()*(i+NB*j); }

    void conj() {
      for (int m = 0; m != MB; ++m)
        for (int n = 0; n != NB; ++n)
          for (int j = 0; j != mptr_[m]; ++j)
            conj_n(block(n,m)+j*nmax_, nptr_[n]);
    }

    void scale(const std::complex<double> a) {
      for (int m = 0; m != MB; ++m)
        for (int n = 0; n != NB; ++n)
          for (int j = 0; j != mptr_[m]; ++j)
            zscal_(nptr_[n], a, block(n,m)+j*nmax_, 1);
    }

    template<size_t iblock, class = typename std::enable_if<(iblock < MB)>::type>
    void add_column(const std::complex<double>* d, const int ld, const int off = 0) {
      assert(mptr_[iblock]+1 <= mmax_);
      for (int nblock = 0; nblock != NB; ++nblock)
        copy_n(d+ld*nblock, nptr_[nblock]-off, block(nblock, iblock)+mptr_[iblock]*nmax_+off);
      mptr_[iblock]++;
    }

    template<size_t iblock, class = typename std::enable_if<(iblock < MB)>::type>
    void add_column(const SuperMatrix<NB,1>& d) {
      add_column<iblock>(d.block(0,0), d.nmax());
    }

    template<size_t iblock, class = typename std::enable_if<(iblock < MB)>::type>
    void add_column_unit(const int off) {
      assert(mptr_[iblock]+1 <= mmax_);
      for (int nblock = 0; nblock != NB; ++nblock)
        *(block(nblock, iblock)+mptr_[iblock]*nmax_+off) = 1.0;
      mptr_[iblock]++;
    }

    template<size_t iblock, class = typename std::enable_if<(iblock < MB)>::type>
    void write_lastcolumn(std::complex<double>* d, const int ld, const int off = 0) {
      for (int nblock = 0; nblock != NB; ++nblock)
        copy_n(d+ld*nblock, nptr_[nblock]-off, block(nblock, iblock)+(mptr_[iblock]-1)*nmax_+off);
    }

    template<size_t iblock, class = typename std::enable_if<(iblock < MB)>::type>
    void write_lastcolumn(const int nblock, const int off) {
      for (int nblock = 0; nblock != NB; ++nblock)
        *(block(nblock, iblock)+(mptr_[iblock]-1)*nmax_+off) = 1.0;
    }

    template<size_t iblock, class = typename std::enable_if<(iblock < NB)>::type>
    void add_row(std::complex<double>* d, const int ld) {
      assert(nptr_[iblock]+1 <= nmax_);
      for (int mblock = 0; mblock != NB; ++mblock)
        for (int m = 0; m != mptr_[mblock]; ++m)
          *(block(iblock, mblock)+m*nmax_+nptr_[iblock]) = *(d+m+ld*mblock);
      nptr_[iblock]++;
    }

    template<size_t iblock, class = typename std::enable_if<(iblock < NB)>::type>
    void add_row(const int mblock = 0, const int off = 0, const std::complex<double> a = 0.0) {
      assert(nptr_[iblock]+1 <= nmax_);
      *(block(iblock, mblock)+off*nmax_+nptr_[iblock]) = a;
      nptr_[iblock]++;
    }

    template<size_t nblock>
    void cut_row(const int off, SuperMatrix<MB,1>& result) const {
      for (int i = 0; i != MB; ++i) {
        std::complex<double>* out = result.block(i,0);
        result.nptr(i) = mptr(i);
        for (int m = 0; m != mptr_[i]; ++m)
          *out++ = std::conj(*(block(nblock, i)+m*nmax_+off));
      }
    }

    int& nptr(const int i) { return nptr_[i]; }
    int& mptr(const int i) { return mptr_[i]; }
    const int& nptr(const int i) const { return nptr_[i]; }
    const int& mptr(const int i) const { return mptr_[i]; }
    int nmax() const { return nmax_; }
    int mmax() const { return mmax_; }

    void reset() {
      std::fill(nptr_.begin(), nptr_.end(), 1);
      std::fill(mptr_.begin(), mptr_.end(), 1);
      std::fill_n(data_, nmax_*mmax_*NB*MB, 0.0);
    }

    void print() {
      std::cout << std::setprecision(4);
      for (int n = 0; n != NB; ++n)
        for (int m = 0; m != MB; ++m) {
          std::cout << n << " " << m << ":" << std::endl;
          for (int i = 0; i != nptr_[n]; ++i) {
            for (int j = 0; j != mptr_[m]; ++j)
              std::cout << *(block(n, m)+i+nmax_*j);
            std::cout << std::endl;
          }
        }
    }
};

namespace {
template<bool transA, bool transB, size_t N, size_t M, size_t K, size_t L, size_t X, size_t Y>
void contract(const char* c1, const char* c2, std::complex<double> a, const SuperMatrix<N,M>& A, const SuperMatrix<K,L>& B, SuperMatrix<X,Y>& C) {
  const constexpr int loopblock = transA ? N : M;
  static_assert((transA ? M : N) == X, "A dim wrong");
  static_assert((transB ? K : L) == Y, "B dim wrong");
  static_assert((transA ? N : M) == (transB ? L : K), "AB dim wrong");
  for (int y = 0; y != Y; ++y)
    for (int x = 0; x != X; ++x)
      for (int l = 0; l != loopblock; ++l) {
        assert((transA ? A.nptr(l) : A.mptr(l)) == (transB ? B.mptr(l) : B.nptr(l)));
        zgemm3m_(c1, c2, (transA ? A.mptr(x) : A.nptr(x)), (transB ? B.nptr(y) : B.mptr(y)), (transA ? A.nptr(l) : A.mptr(l)),
                 a, (transA ? A.block(l, x) : A.block(x, l)), A.nmax(), (transB ? B.block(y, l) : B.block(l, y)), B.nmax(),
                 1.0, C.block(x,y), C.nmax());
        C.nptr(x) = (transA ? A.mptr(x) : A.nptr(x));
        C.mptr(y) = (transB ? B.nptr(y) : B.mptr(y));
      }
}
}

#endif
