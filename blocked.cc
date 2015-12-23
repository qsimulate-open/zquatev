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

#include "zquartev.h"
#include "f77.h"
#include <cassert>
#include "supermat.h"
#include <algorithm>

using namespace std;

namespace ts {

// some local functions..
namespace {
pair<double,complex<double>> givens(const complex<double> a, const complex<double> b) {
  const double absa = abs(a);
  const double c = absa == 0.0 ? 0.0 : absa / sqrt(absa*absa + norm(b));
  const complex<double> s = absa == 0.0 ? 1.0 : (a / absa * conj(b) / sqrt(absa*absa + norm(b)));
  return make_pair(c, s);
}

complex<double> householder(const complex<double>* const hin, complex<double>* const out, const int len) {
  const bool trivial = abs(real(hin[0])) == 0.0;
  complex<double> fac = 0.0;
  copy_n(hin, len, out);
  if (!trivial) {
    const double norm = sqrt(real(zdotc_(len, hin, 1, hin, 1)));
    const double sign = real(hin[0])/abs(real(hin[0]));
    out[0] = hin[0] + sign*norm;
    fac = 1.0 / (out[0] * (sign*norm));
  }
  return fac;
}
}


// TODO Debug
void print(string label, const complex<double>* Q0, const complex<double>* Q1,
           const SuperMatrix<3,3>& T, const SuperMatrix<1,3>& W,
           const SuperMatrix<3,1>& R, const SuperMatrix<1,3>& S,
           SuperMatrix<3,1>& W2, SuperMatrix<1,1>& Y1, SuperMatrix<1,1>& Y2, SuperMatrix<1,1>& X, const int n) {
  {
    cout << setprecision(4) << "------ Q0 prelim " << label << endl;
    for (int i = 0; i != n; ++i) {
      for (int j = 0; j != n; ++j)
        cout << Q0[i+j*n] - (i == j ? 1.0 : 0.0);
      cout << endl;
    }
    cout << setprecision(4) << "------ Q0 reconst " << label << endl;
    W2.reset();
    X.reset();
    contract<false,true> ("N", "C", 1.0, T, W, W2);
    contract<false,false>("N", "N", 1.0, W, W2, X);
    X.print();
  }
#if 1
  {
    cout << setprecision(4) << "------ Q1 prelim " << label << endl;
    for (int i = 0; i != n; ++i) {
      for (int j = 0; j != n; ++j)
        cout << Q1[i+j*n];
      cout << endl;
    }
    cout << setprecision(4) << "------ Q1 reconst " << label << endl;
    Y1.reset();
    Y2.reset();
    X.reset();
    contract<false,false>("N", "N", 1.0, W, R, Y1);
    contract<false,true >("N", "C", 1.0, W, S, Y2);
    Y1.conj();
    contract<false,true >("N", "C",-1.0, Y1, Y2, X);
    X.print();
  }
#endif
}

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

  // will be updated
  const int nb = n;

  unique_ptr<complex<double>[]> buf(new complex<double>[n*3]);
  unique_ptr<complex<double>[]> buf2(new complex<double>[n*3]);
  unique_ptr<complex<double>[]> buf3(new complex<double>[nb*3]);
  unique_ptr<complex<double>[]> buf4(new complex<double>[nb*3]);
  unique_ptr<complex<double>[]> hout(new complex<double>[n2]);
  unique_ptr<complex<double>[]> choutf(new complex<double>[n2]);

  unique_ptr<complex<double>[]> _T(new complex<double>[9*n*n]);
  unique_ptr<complex<double>[]> _R(new complex<double>[3*n*n]);
  unique_ptr<complex<double>[]> _S(new complex<double>[3*n*n]);
  unique_ptr<complex<double>[]> _Y1(new complex<double>[n*n]);
  unique_ptr<complex<double>[]> _Y2(new complex<double>[n*n]);
  unique_ptr<complex<double>[]> _W(new complex<double>[3*n*n]);
  unique_ptr<complex<double>[]> _X(new complex<double>[n*n]);
  unique_ptr<complex<double>[]> _W2(new complex<double>[3*n*n]);
  SuperMatrix<3,3> T(_T.get(), nb, nb);
  SuperMatrix<3,1> R(_R.get(), nb, nb);
  SuperMatrix<1,3> S(_S.get(), nb, nb);
  SuperMatrix<1,3> W(_W.get(), n, nb, n, 1);
  SuperMatrix<1,1> X(_X.get(), n, n, n, n);

  SuperMatrix<3,1> W2(_W2.get(), n, n);
  SuperMatrix<1,1> Y1(_Y1.get(), n, n);
  SuperMatrix<1,1> Y2(_Y2.get(), n, n);

  // Reference - arXiv:1203.6151v4
  for (int k = 0; k != n-1; ++k) {
    const int len = n-k-1;
    if (len > 1) {
      complex<double>* const hin = D1+n*k+k+1;
      complex<double> tau = householder(hin, hout.get(), len);

      for (int i = 0; i != len; ++i) choutf[i] = conj(hout[i]);

      if (k == 0) {
        W.write_lastcolumn<0>(choutf.get(), len, 1);
        T.data<0,0>(0,0) = -tau;
      } else {
        R.append_row<0>();
        SuperMatrix<3,1> x(buf2.get(), nb, 1);
        {
          SuperMatrix<1,1> v(buf.get(), n, 1, n, 1);
          v.write_lastcolumn<0>(choutf.get(), len, k+1);
          contract<true, false>("C", "N", -tau, W, v, x);
        } {
          SuperMatrix<3,1> v(buf.get(), nb, 1);
          contract<false,false>("N", "N", 1.0, T, x, v);
          T.append_column<0>(v);
          T.append_row<0>(0, k, -tau);
        } {
          SuperMatrix<1,1> v(buf.get(), nb, 1);
          contract<false,false>("N", "N", 1.0, S, x, v);
          S.append_column<0>(v);
        }
        W.append_column<0>(choutf.get(), len, k+1);
      }

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

    }

    // symplectic Givens rotation to clear out D(k+n, k)
    pair<double,complex<double>> gr = givens(D0[k+1+k*n], D1[k+1+k*n]);
    zrot_(len+1, D0+k+1+k*n, n, D1+k+1+k*n, n, gr.first, gr.second);

    for (int i = 0; i != len+1; ++i)
      D1[(k+1)*n+k+i] = -conj(D1[(k+1)*n+k+i]);
    zrot_(len+1, D0+(k+1)*n+k, 1, D1+(k+1)*n+k, 1, gr.first, conj(gr.second));
    for (int i = 0; i != len+1; ++i)
      D1[(k+1)*n+k+i] = -conj(D1[(k+1)*n+k+i]);

    for (int i = 0; i != n; ++i)
      Q1[(k+1)*n+i] = -conj(Q1[(k+1)*n+i]);
    zrot_(n, Q0+(k+1)*n, 1, Q1+(k+1)*n, 1, gr.first, conj(gr.second));
    for (int i = 0; i != n; ++i)
      Q1[(k+1)*n+i] = -conj(Q1[(k+1)*n+i]);


    const double cbar = -(1.0-gr.first);
    const complex<double> sbar = conj(gr.second);
    if (k == 0) {
      T.data<0,1>(0,0) = T.data<0,0>(0,0)*cbar*conj(W.data<0,0>(1,0));
      T.data<1,1>(0,0) = cbar;
      W.write_lastcolumn<1>(0,1);
      R.data<0,0>(0,0) = T.data<0,0>(0,0)*conj(W.data<0,0>(1,0));
      R.data<1,0>(0,0) = 1.0;
      S.data<0,1>(0,0) = -sbar;
    } else {
      SuperMatrix<3,1> x(buf2.get(), nb, 1); // holds W^+ e_j
      SuperMatrix<3,1> v(buf.get(), nb, 1); // holds TW^+ e_j
      W.cut_row<0>(k+1, x);
      contract<false,false>("N", "N", 1.0, T, x, v);

      // update part of T
      SuperMatrix<1,1> y(buf3.get(), nb, 1); // holds SW^+ e_j
      SuperMatrix<3,1> z(buf4.get(), nb, 1); // holds RSW^+ e_j
      contract<false,false>("N", "N", 1.0, S, x, y);
      y.conj();
      contract<false,false>("N", "N", sbar, R, y, z);
      y.conj();
      T.append_column<1>(z);

      // update R and S -- ok
      R.append_column<0>(v);
      R.append_row<1>(0,k,1.0);
      y.scale(cbar);
      S.append_column<1>(y);
      S.append_row<0>(1, k, -sbar);

      // update the rest of T
      T.add_lastcolumn<1>(v, cbar);
      T.append_row<1>(1, k, cbar);
      // update W
      W.append_column_unit<1>(k+1);
    }

    // Householder to fix top half in column k
    if (len > 1) {
      complex<double>* const hin = D0+n*k+k+1;
      complex<double> tau = householder(hin, hout.get(), len);

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

      if (k == 0) {
        W.write_lastcolumn<2>(hout.get(), len, 1);
        T.data<0,2>(0,0) = -conj(tau)*T.data<0,0>(0,0)*zdotc_(len, W.block(0,0)+1, 1, hout.get(), 1)
                           -conj(tau)*T.data<0,1>(0,0)*hout[0];
        T.data<1,2>(0,0) = -conj(tau)*T.data<1,1>(0,0)*hout[0];
        T.data<2,2>(0,0) = -conj(tau);
        S.data<0,2>(0,0) = -conj(tau)*S.data<0,1>(0,0)*hout[0];
      } else {
        R.append_row<2>();
        SuperMatrix<3,1> x(buf2.get(), nb, 1);
        {
          SuperMatrix<1,1> v(buf.get(), n, 1, n, 1);
          v.write_lastcolumn<0>(hout.get(), len, k+1);
          contract<true, false>("C", "N", -conj(tau), W, v, x);
        } {
          SuperMatrix<3,1> v(buf.get(), nb, 1);
          contract<false,false>("N", "N", 1.0, T, x, v);
          T.append_column<2>(v);
          T.append_row<2>(2, k, -conj(tau));
        } {
          SuperMatrix<1,1> v(buf.get(), nb, 1);
          contract<false,false>("N", "N", 1.0, S, x, v);
          S.append_column<2>(v);
        }
        W.append_column<2>(hout.get(), len, k+1);
      }
    }
  }
  print("tridiagonalization", Q0, Q1, T, W, R, S, W2, Y1, Y2, X, n);

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
