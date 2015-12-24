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

#include "zquatev.h"
#include "f77.h"
#include <cassert>
#include <algorithm>

using namespace std;

namespace ts {

// implementation...

void zquatev(const int n2, complex<double>* const D, double* const eig) {
  assert(n2 % 2 == 0);
  const int n = n2/2;

  // rearrange data
  complex<double>* const D0 = D;
  complex<double>* const D1 = D + n*n;
  copy_n(D, n2*n, D+n2*n);
  for (int i = 0; i != n; ++i) {
    copy_n(D+n2*n+i*n2, n, D0+i*n);
    copy_n(D+n2*n+i*n2+n, n, D1+i*n);
  }

  // identity matrix of n2 dimension
  complex<double>* const Q0 = D + n2*n;
  complex<double>* const Q1 = D + n2*n + n*n;
  fill_n(Q0, n*n, 0.0);
  fill_n(Q1, n*n, 0.0);
  for (int i = 0; i != n; ++i) Q0[i+n*i] = 1.0;

  unique_ptr<complex<double>[]> buf(new complex<double>[n2]);
  unique_ptr<complex<double>[]> hout(new complex<double>[n2]);
  unique_ptr<complex<double>[]> choutf(new complex<double>[n2]);


  // Reference - arXiv:1203.6151v4
  for (int k = 0; k != n-1; ++k) {
    const int len = n-k-1;
    if (len > 1) {
      copy_n(D1+n*k+k+2, len-1, hout.get()+1);
      complex<double> tau;
      complex<double> alpha = D1[n*k+k+1];
      hout[0] = 1.0;
      zlarfg_(len, alpha, hout.get()+1, 1, tau);
      tau = conj(tau);

      for (int i = 0; i != len; ++i) choutf[i] = conj(hout[i]);

      // 00
      zgemv_("C", len, len+1, 1.0, D0+k+1+(k)*n, n, choutf.get(), 1, 0.0, buf.get(), 1);
      zaxpy_(len, -conj(tau)*0.5*zdotc_(len, buf.get()+1, 1, choutf.get(), 1), choutf.get(), 1, buf.get()+1, 1);
      zgerc_(len, len+1, -conj(tau), choutf.get(), 1, buf.get(), 1, D0+k+1+(k)*n, n);
      zgeru_(len+1, len, -tau, buf.get(), 1, hout.get(), 1, D0+(k+1)*n+(k), n);

      // 10
      zgemv_("N", len+1, len, 1.0, D1+k+(k+1)*n, n, choutf.get(), 1, 0.0, buf.get(), 1);
      zgeru_(len, len+1, tau, hout.get(), 1, buf.get(), 1, D1+k+1+(k)*n, n);
      zgeru_(len+1, len, -tau, buf.get(), 1, hout.get(), 1, D1+(k+1)*n+(k), n);

      // 00-2
      zgemv_("N", n, len, 1.0, Q0+(k+1)*n, n, choutf.get(), 1, 0.0, buf.get(), 1);
      zgeru_(n, len, -tau, buf.get(), 1, hout.get(), 1, Q0+(k+1)*n, n);

      // 10-2
      zgemv_("N", n, len, 1.0, Q1+(k+1)*n, n, choutf.get(), 1, 0.0, buf.get(), 1);
      zgeru_(n, len, -tau, buf.get(), 1, hout.get(), 1, Q1+(k+1)*n, n);

      // lapack routine returns transformed subdiagonal element
      assert(abs(alpha - D1[n*k+k+1]) < 1.0e-10);
    }

    // symplectic Givens rotation to clear out D(k+n, k)
    double c;
    complex<double> s, dum;
    zlartg_(D0[k+1+k*n], D1[k+1+k*n], c, s, dum);

    zrot_(len+1, D0+k+1+k*n, n, D1+k+1+k*n, n, c, s);

    for (int i = 0; i != len+1; ++i)
      D1[(k+1)*n+k+i] = -conj(D1[(k+1)*n+k+i]);
    zrot_(len+1, D0+(k+1)*n+k, 1, D1+(k+1)*n+k, 1, c, conj(s));
    for (int i = 0; i != len+1; ++i)
      D1[(k+1)*n+k+i] = -conj(D1[(k+1)*n+k+i]);

    for (int i = 0; i != n; ++i)
      Q1[(k+1)*n+i] = -conj(Q1[(k+1)*n+i]);
    zrot_(n, Q0+(k+1)*n, 1, Q1+(k+1)*n, 1, c, conj(s));
    for (int i = 0; i != n; ++i)
      Q1[(k+1)*n+i] = -conj(Q1[(k+1)*n+i]);

    // Householder to fix top half in column k
    if (len > 1) {
      copy_n(D0+n*k+k+2, len-1, hout.get()+1);
      complex<double> tau;
      complex<double> alpha = D0[n*k+k+1];
      hout[0] = 1.0;
      zlarfg_(len, alpha, hout.get()+1, 1, tau);
      tau = conj(tau);

      for (int i = 0; i != len; ++i) choutf[i] = conj(hout[i]);

      // 00
      zgemv_("C", len, len+1, 1.0, D0+k+1+(k)*n, n, hout.get(), 1, 0.0, buf.get(), 1);
      zaxpy_(len, -tau*0.5*zdotc_(len, hout.get(), 1, buf.get()+1, 1), hout.get(), 1, buf.get()+1, 1);
      zgerc_(len, len+1, -tau, hout.get(), 1, buf.get(), 1, D0+k+1+(k)*n, n);
      zgerc_(len+1, len, -conj(tau), buf.get(), 1, hout.get(), 1, D0+(k+1)*n+(k), n);

      // 01-1
      zgemv_("T", len, len+1, 1.0, D1+k+1+(k)*n, n, hout.get(), 1, 0.0, buf.get(), 1);
      zgeru_(len, len+1, -conj(tau), choutf.get(), 1, buf.get(), 1, D1+k+1+(k)*n, n);
      zgerc_(len+1, len, conj(tau), buf.get(), 1, hout.get(), 1, D1+(k+1)*n+(k), n);

      // 00-2
      zgemv_("N", n, len, 1.0, Q0+(k+1)*n, n, hout.get(), 1, 0.0, buf.get(), 1);
      zgerc_(n, len, -conj(tau), buf.get(), 1, hout.get(), 1, Q0+(k+1)*n, n);

      // 01-2
      zgemv_("N", n, len, -1.0, Q1+(k+1)*n, n, hout.get(), 1, 0.0, buf.get(), 1);
      zgerc_(n, len, conj(tau), buf.get(), 1, hout.get(), 1, Q1+(k+1)*n, n);

      // lapack routine returns transformed subdiagonal element
      assert(abs(alpha - D0[n*k+k+1]) < 1.0e-10);
    }

  }

  // diagonalize this tri-diagonal matrix (this step is much cheaper than
  // the Householder transformation above).
  unique_ptr<complex<double>[]> Cmat(new complex<double>[n*n]);
  unique_ptr<complex<double>[]> Work(new complex<double>[n]);
  int info;
  unique_ptr<double[]> rwork(new double[n*3]);
  for (int i = 0; i != n; ++i)
    for (int j = 0; j <= i; ++j)
      D0[i-j+j*n] = D0[i+j*n];
  zhbev_("V", "L", n, 1, D0, n, eig, Cmat.get(), n, Work.get(), rwork.get(), info);
  if (info) throw runtime_error("zhbev failed in quaternion diagonalization");

  // form the coefficient matrix in D
#ifdef MKL
  zgemm3m_("N", "N", n, n, n, 1.0, Q0, n, Cmat.get(), n, 0.0, D, n2);
  zgemm3m_("N", "N", n, n, n, 1.0, Q1, n, Cmat.get(), n, 0.0, D+n, n2);
#else
  zgemm_("N", "N", n, n, n, 1.0, Q0, n, Cmat.get(), n, 0.0, D, n2);
  zgemm_("N", "N", n, n, n, 1.0, Q1, n, Cmat.get(), n, 0.0, D+n, n2);
#endif

  // eigen vectors using symmetry
  for (int i = 0; i != n; ++i) {
    for (int j = 0; j != n; ++j) {
       D[j+n2*(i+n)] = -conj(D[j+n+n2*i]);
       D[j+n+n2*(i+n)] = conj(D[j+n2*i]);
    }
  }
}

}
