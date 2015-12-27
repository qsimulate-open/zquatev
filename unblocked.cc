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

void unblocked_update(const int n, complex<double>* const D0, complex<double>* const D1, complex<double>* const Q0, complex<double>* const Q1, const int ld,
                      const int norig, complex<double>* const work) {

  complex<double>* tmp =  work;
  complex<double>* vec =  work + n;
  complex<double>* cvec = work + n*2;

  for (int k = 0; k != n-1; ++k) {
    const int len = n-k-1;
    if (len > 1) {
      copy_n(D1+k*ld+k+2, len-1, vec+1);
      complex<double> tau;
      complex<double> alpha = D1[k*ld+k+1];
      vec[0] = 1.0;
      zlarfg_(len, alpha, vec+1, 1, tau);
      tau = conj(tau);

      for (int i = 0; i != len; ++i) cvec[i] = conj(vec[i]);

      // 00
      zgemv_("C", len, len+1, 1.0, D0+k+1+(k)*ld, ld, cvec, 1, 0.0, tmp, 1);
      zaxpy_(len, -conj(tau)*0.5*zdotc_(len, tmp+1, 1, cvec, 1), cvec, 1, tmp+1, 1);
      zgerc_(len, len+1, -conj(tau), cvec, 1, tmp, 1, D0+k+1+(k)*ld, ld);
      zgeru_(len+1, len, -tau, tmp, 1, vec, 1, D0+(k+1)*ld+(k), ld);

      // 10
      zgemv_("N", len+1, len, 1.0, D1+k+(k+1)*ld, ld, cvec, 1, 0.0, tmp, 1);
      zgeru_(len, len+1, tau, vec, 1, tmp, 1, D1+k+1+(k)*ld, ld);
      zgeru_(len+1, len, -tau, tmp, 1, vec, 1, D1+(k+1)*ld+(k), ld);

      // 00-2
      zgemv_("N", n, len, 1.0, Q0+(k+1)*n, n, cvec, 1, 0.0, tmp, 1);
      zgeru_(n, len, -tau, tmp, 1, vec, 1, Q0+(k+1)*n, n);

      // 10-2
      zgemv_("N", n, len, 1.0, Q1+(k+1)*n, n, cvec, 1, 0.0, tmp, 1);
      zgeru_(n, len, -tau, tmp, 1, vec, 1, Q1+(k+1)*n, n);
    }

    // symplectic Givens rotation to clear out D(k+n, k)
    double c;
    complex<double> s, dum;
    zlartg_(D0[k+1+k*ld], D1[k+1+k*ld], c, s, dum);

    zrot_(len+1, D0+k+1+k*ld, ld, D1+k+1+k*ld, ld, c, s);

    for (int i = 0; i != len+1; ++i)
      D1[(k+1)*ld+k+i] = -conj(D1[(k+1)*ld+k+i]);
    zrot_(len+1, D0+(k+1)*ld+k, 1, D1+(k+1)*ld+k, 1, c, conj(s));
    for (int i = 0; i != len+1; ++i)
      D1[(k+1)*ld+k+i] = -conj(D1[(k+1)*ld+k+i]);

    for (int i = 0; i != n; ++i)
      Q1[(k+1)*n+i] = -conj(Q1[(k+1)*n+i]);
    zrot_(n, Q0+(k+1)*n, 1, Q1+(k+1)*n, 1, c, conj(s));
    for (int i = 0; i != n; ++i)
      Q1[(k+1)*n+i] = -conj(Q1[(k+1)*n+i]);

    // Householder to fix top half in column k
    if (len > 1) {
      copy_n(D0+k*ld+k+2, len-1, vec+1);
      complex<double> tau;
      complex<double> alpha = D0[k*ld+k+1];
      vec[0] = 1.0;
      zlarfg_(len, alpha, vec+1, 1, tau);
      tau = conj(tau);

      for (int i = 0; i != len; ++i) cvec[i] = conj(vec[i]);

      // 00
      zgemv_("C", len, len+1, 1.0, D0+k+1+(k)*ld, ld, vec, 1, 0.0, tmp, 1);
      zaxpy_(len, -tau*0.5*zdotc_(len, vec, 1, tmp+1, 1), vec, 1, tmp+1, 1);
      zgerc_(len, len+1, -tau, vec, 1, tmp, 1, D0+k+1+(k)*ld, ld);
      zgerc_(len+1, len, -conj(tau), tmp, 1, vec, 1, D0+(k+1)*ld+(k), ld);

      // 01-1
      zgemv_("T", len, len+1, 1.0, D1+k+1+(k)*ld, ld, vec, 1, 0.0, tmp, 1);
      zgeru_(len, len+1, -conj(tau), cvec, 1, tmp, 1, D1+k+1+(k)*ld, ld);
      zgerc_(len+1, len, conj(tau), tmp, 1, vec, 1, D1+(k+1)*ld+(k), ld);

      // 00-2
      zgemv_("N", n, len, 1.0, Q0+(k+1)*n, n, vec, 1, 0.0, tmp, 1);
      zgerc_(n, len, -conj(tau), tmp, 1, vec, 1, Q0+(k+1)*n, n);

      // 01-2
      zgemv_("N", n, len, -1.0, Q1+(k+1)*n, n, vec, 1, 0.0, tmp, 1);
      zgerc_(n, len, conj(tau), tmp, 1, vec, 1, Q1+(k+1)*n, n);

    }

  }
}

}
