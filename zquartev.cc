//
// Filename: zquartev.cc
// Copyright (C) 2013 Toru Shiozaki
//
// Author: Toru Shiozaki <shiozaki@northwestern.edu> 
// Maintainer: TS 
//
// You can redistribute this program and/or modify
// it under the terms of the GNU Library General Public License as published by
// the Free Software Foundation; either version 3, or (at your option)
// any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU Library General Public License for more details.
//

#include "zquartev.h"
#include "f77.h"
#include <cassert>

using namespace std;

namespace ts {

// some local functions..
static auto givens = [](const complex<double> a, const complex<double> b) {
  const double absa = abs(a);
  const double c = absa / sqrt(absa*absa + norm(b));
  const complex<double> s = absa == 0.0 ? 1.0 : (a / absa * conj(b) / sqrt(absa*absa + norm(b)));
  return make_pair(c, s); 
};  

static auto householder = [](const complex<double>* const hin, complex<double>* const out, const int len) {
  const double norm = sqrt(real(zdotc_(len, hin, 1, hin, 1)));
  const double sign = abs(real(hin[0])) == 0.0 ? 0.0 : real(hin[0])/abs(real(hin[0]));
  out[0] = hin[0] + sign*norm;
  for (int i = 1; i < len; ++i) out[i] = hin[i];
  return conj(1.0 / (conj(out[0]) * (sign*norm)));
};  

static auto conjarray = [](unique_ptr<complex<double>[]>& a, const int n) {
  for (int i = 0; i != n; ++i) a[i] = conj(a[i]);
};


// implementation...

void zquatev(const int n2, complex<double>* const D, double* const eig) {
  assert(n2 % 2 == 0); 
  const int n = n2/2;

  // identity matrix of n2 dimension
  unique_ptr<complex<double>[]> Qbuf(new complex<double>[n2*n2]);
  complex<double>* const Q = Qbuf.get();
  for (int i = 0; i != n2; ++i) Q[i+n2*i] = 1.0;

  unique_ptr<complex<double>[]> buf(new complex<double>[n2]);
  unique_ptr<complex<double>[]> hout(new complex<double>[n2]);
  unique_ptr<complex<double>[]> choutf(new complex<double>[n2]);


  // Reference - arXiv:1203.6151v4
  for (int k = 0; k != n-1; ++k) {
    const int len = n-k-1;
    if (len > 1) {
      complex<double>* const hin = D+n2*k+k+1+n;
      complex<double> tau = householder(hin, hout.get(), len);

      for (int i = 0; i != len; ++i) choutf[i] = conj(hout[i]);

      // 00-1
      zgemv_("T", len, len+1, 1.0, D+k+1+(k)*n2, n2, hout.get(), 1, 0.0, buf.get(), 1); 
      zgeru_(len, len+1, -conj(tau), choutf.get(), 1, buf.get(), 1, D+k+1+(k)*n2, n2);

      // 00-2
      zgemv_("N", len+1, len, 1.0, D+(k+1)*n2+(k), n2, choutf.get(), 1, 0.0, buf.get(), 1); 
      zgeru_(len+1, len, -tau, buf.get(), 1, hout.get(), 1, D+(k+1)*n2+(k), n2);

      // 10-1
      zgemv_("N", len+1, len, 1.0, D+k+n+(k+1)*n2, n2, choutf.get(), 1, 0.0, buf.get(), 1); 
      conjarray(buf, len+1);
      zgerc_(len, len+1, tau, hout.get(), 1, buf.get(), 1, D+k+n+1+(k)*n2, n2);

      // 10-2
      zgemv_("N", len+1, len, 1.0, D+k+n+(k+1)*n2, n2, choutf.get(), 1, 0.0, buf.get(), 1); 
      zgeru_(len+1, len, -tau, buf.get(), 1, hout.get(), 1, D+(k+1)*n2+(k+n), n2);

      // 00-2
      zgemv_("N", len+1, len, 1.0, Q+(k+1)*n2+(k), n2, choutf.get(), 1, 0.0, buf.get(), 1); 
      zgeru_(len+1, len, -tau, buf.get(), 1, hout.get(), 1, Q+(k+1)*n2+(k), n2);

      // 10-2
      zgemv_("N", len+1, len, 1.0, Q+k+n+(k+1)*n2, n2, choutf.get(), 1, 0.0, buf.get(), 1); 
      zgeru_(len+1, len, -tau, buf.get(), 1, hout.get(), 1, Q+(k+1)*n2+(k+n), n2);

    }   

    // symplectic Givens rotation to clear out D(k+n, k)
    pair<double,complex<double>> gr = givens(D[k+1+k*n2], D[k+n+1+k*n2]);

    zrot_(len+1, D+k+1+k*n2, n2, D+k+n+1+k*n2, n2, gr.first, gr.second);

    for (int i = 0; i != len+1; ++i)
      D[(k+1)*n2+k+n+i] = -conj(D[(k+1)*n2+k+n+i]);
    zrot_(len+1, D+(k+1)*n2+k, 1, D+(k+1)*n2+k+n, 1, gr.first, conj(gr.second));
    for (int i = 0; i != len+1; ++i)
      D[(k+1)*n2+k+n+i] = -conj(D[(k+1)*n2+k+n+i]);

    for (int i = 0; i != len+1; ++i)
      Q[(k+1)*n2+k+n+i] = -conj(Q[(k+1)*n2+k+n+i]);
    zrot_(len+1, Q+(k+1)*n2+k, 1, Q+(k+1)*n2+k+n, 1, gr.first, conj(gr.second));
    for (int i = 0; i != len+1; ++i)
      Q[(k+1)*n2+k+n+i] = -conj(Q[(k+1)*n2+k+n+i]);

    // Householder to fix top half in column k
    if (len > 1) {
      complex<double>* const hin = D+n2*k+k+1;
      complex<double> tau = householder(hin, hout.get(), len);

      for (int i = 0; i != len; ++i) choutf[i] = conj(hout[i]);

      // 00-1
      zgemv_("C", len, len+1, 1.0, D+k+1+(k)*n2, n2, hout.get(), 1, 0.0, buf.get(), 1); 
      zgerc_(len, len+1, -tau, hout.get(), 1, buf.get(), 1, D+k+1+(k)*n2, n2);

      // 00-2
      zgemv_("N", len+1, len, 1.0, D+(k+1)*n2+(k), n2, hout.get(), 1, 0.0, buf.get(), 1); 
      zgerc_(len+1, len, -conj(tau), buf.get(), 1, hout.get(), 1, D+(k+1)*n2+(k), n2);

      // 01-1
      zgemv_("T", len, len+1, 1.0, D+k+n+1+(k)*n2, n2, hout.get(), 1, 0.0, buf.get(), 1); 
      zgeru_(len, len+1, -conj(tau), choutf.get(), 1, buf.get(), 1, D+k+n+1+(k)*n2, n2);

      // 01-2 
      zgemv_("N", len+1, len, -1.0, D+(k+1)*n2+(k+n), n2, hout.get(), 1, 0.0, buf.get(), 1); 
      zgerc_(len+1, len, conj(tau), buf.get(), 1, hout.get(), 1, D+(k+1)*n2+(k+n), n2);

      // 00-2
      zgemv_("N", len+1, len, 1.0, Q+(k+1)*n2+(k), n2, hout.get(), 1, 0.0, buf.get(), 1); 
      zgerc_(len+1, len, -conj(tau), buf.get(), 1, hout.get(), 1, Q+(k+1)*n2+(k), n2);

      // 01-2 
      zgemv_("N", len+1, len, -1.0, Q+(k+1)*n2+(k+n), n2, hout.get(), 1, 0.0, buf.get(), 1); 
      zgerc_(len+1, len, conj(tau), buf.get(), 1, hout.get(), 1, Q+(k+1)*n2+(k+n), n2);

    }   

  }

  // diagonalize this tri-diagonal matrix (this step is much cheaper than
  // the Householder transformation above).
  unique_ptr<complex<double>[]> Cmat(new complex<double>[n2*n2]);
  complex<double>* Work = D+n*n2;
  int info;
  unique_ptr<double[]> rwork(new double[n*3]);
  for (int i = 0; i != n; ++i)
    for (int j = 0; j <= i; ++j)
      D[i-j+j*n2] = D[i+j*n2];
  zhbev_("V", "L", n, 1, D, n2, eig, Cmat.get(), n, Work, rwork.get(), info);
  if (info) throw runtime_error("zhbev failed in quaternion diagonalization");

  for (int i = 0; i != n; ++i) {
    for (int j = 0; j != n; ++j) {
      Q[(n+i)*n2+n+j] = conj(Q[i*n2+j]);
      Q[(n+i)*n2+j] = -conj(Q[i*n2+n+j]);
    }
  }

  // form the coefficient matrix in D
#ifdef MKL
  zgemm3m_("N", "N", n2, n, n, 1.0, Q, n2, Cmat.get(), n, 0.0, D, n2);
#else
  zgemm_("N", "N", n2, n, n, 1.0, Q, n2, Cmat.get(), n, 0.0, D, n2);
#endif

  for (int i = 0; i != n*n; ++i) Cmat[i] = conj(Cmat[i]);
#ifdef MKL
  zgemm3m_("N", "N", n2, n, n, 1.0, Q+n*n2, n2, Cmat.get(), n, 0.0, D+n*n2, n2);
#else
  zgemm_("N", "N", n2, n, n, 1.0, Q+n*n2, n2, Cmat.get(), n, 0.0, D+n*n2, n2);
#endif
}

}
