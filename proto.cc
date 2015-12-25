//
// ZQUATEV: Diagonalization of quaternionic matrices
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

#include "zquatev.h"
#include "f77.h"
#include "supermat.h"
#include <cassert>

using namespace std;

namespace ts {

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
  unique_ptr<complex<double>[]> buf2(new complex<double>[nb*3]);
  unique_ptr<complex<double>[]> buf3(new complex<double>[nb]);
  unique_ptr<complex<double>[]> buf4(new complex<double>[nb]);

  auto work1_3n  = buf.get();
  auto work2_3nb = buf2.get();
  auto work3_nb  = buf3.get();
  auto work4_nb  = buf4.get();

  unique_ptr<complex<double>[]> _T(new complex<double>[9*nb*nb]);
  unique_ptr<complex<double>[]> _R(new complex<double>[3*nb*nb]);
  unique_ptr<complex<double>[]> _S(new complex<double>[3*nb*nb]);
  unique_ptr<complex<double>[]> _W(new complex<double>[3*nb*n]);
  unique_ptr<complex<double>[]> _YD(new complex<double>[3*nb*n]);
  unique_ptr<complex<double>[]> _YE(new complex<double>[3*nb*n]);
  unique_ptr<complex<double>[]> _ZD(new complex<double>[3*nb*n]);
  unique_ptr<complex<double>[]> _ZE(new complex<double>[3*nb*n]);
  SuperMatrix<3,3> T(_T.get(), nb, nb);
  SuperMatrix<3,1> R(_R.get(), nb, nb);
  SuperMatrix<1,3> S(_S.get(), nb, nb);
  SuperMatrix<1,3> W(_W.get(), n, nb, n, 1);
  SuperMatrix<1,3> YD(_YD.get(), n, nb, n, 1);
  SuperMatrix<1,3> ZD(_ZD.get(), n, nb, n, 1);
  SuperMatrix<1,3> YE(_YE.get(), n, nb, n, 1);
  SuperMatrix<1,3> ZE(_ZE.get(), n, nb, n, 1);

  // TODO the following matrices are only used for printing so far
  unique_ptr<complex<double>[]> _Y1(new complex<double>[n*n]);
  unique_ptr<complex<double>[]> _Y2(new complex<double>[n*n]);
  unique_ptr<complex<double>[]> _X(new complex<double>[n*n]);
  unique_ptr<complex<double>[]> _W2(new complex<double>[3*n*n]);
  SuperMatrix<1,1> X(_X.get(), n, n, n, n);
  SuperMatrix<3,1> W2(_W2.get(), n, n);
  SuperMatrix<1,1> Y1(_Y1.get(), n, n);
  SuperMatrix<1,1> Y2(_Y2.get(), n, n);

  for (int k = 0; k != n-1; ++k) {
    const int len = n-k-1;

    // prepare the k-th column using compact WY-like representation
    if (k > 0) {
      SuperMatrix<1,1> d(D0+k*n, n, 1, n, 1, false);
      SuperMatrix<1,1> e(D1+k*n, n, 1, n, 1, false);

      SuperMatrix<3,1> x(work2_3nb, nb, 1, nb, 1);
      W.cut_row<0>(k, x);
      contract<false>("N", 1.0, YD, x, d);
      contract<false>("N", 1.0, YE, x, e);

      x.conj();
      SuperMatrix<1,1> y(work1_3n, n, 1);
      contract<false>("N",  1.0, ZE, x, y);
      y.conj();
      d.add_lastcolumn<0>(y);
      y.reset();
      contract<false>("N", -1.0, ZD, x, y);
      y.conj();
      e.add_lastcolumn<0>(y);

      SuperMatrix<3,1> y1(work1_3n, nb, 1);
      contract<true> ("C", 1.0, W, d, y1);
      SuperMatrix<3,1> y2(work2_3nb, nb, 1);
      contract<true> ("C", 1.0, T, y1, y2);
      contract<false>("N", 1.0, W, y2, d);
      SuperMatrix<1,1> y3(work3_nb, nb, 1);
      contract<true> ("C", 1.0, R, y1, y3);
      y3.conj();
      y2.reset();
      contract<true> ("C", 1.0, S, y3, y2);
      SuperMatrix<1,1> y4(work1_3n, n, 1);
      contract<false>("N", 1.0, W, y2, y4);
      y4.conj();

      SuperMatrix<3,1> y5(work2_3nb, nb, 1);
      contract<true> ("T", 1.0, W, e, y5);
      e.add_lastcolumn<0>(y4);

      SuperMatrix<3,1> y5x(work1_3n, y5);
      SuperMatrix<1,1> y6(work3_nb, nb, 1);
      contract<true> ("T", 1.0, R, y5x, y6);
      SuperMatrix<3,1> y7(work2_3nb, nb, 1);
      contract<true> ("C", 1.0, S, y6, y7);
      contract<false>("N", -1.0, W, y7, d);
      y7.reset();
      contract<true> ("T", 1.0, T, y5x, y7);
      y7.conj();
      SuperMatrix<1,1> y8(work1_3n, n, 1);
      contract<false>("N", 1.0, W, y7, y8);
      y8.conj();
      e.add_lastcolumn<0>(y8);

cout << "print d" << endl;
d.print();
cout << "print e" << endl;
e.print();
assert(false);
    }

    complex<double> alpha = D1[n*k+k+1];
    if (len > 1) {
      complex<double>* dnow = D1+n*k+k+1;
      complex<double> tau;
      dnow[0] = 1.0;
      zlarfg_(len, alpha, dnow+1, 1, tau);
      tau = conj(tau);
      conj_n(dnow, len);

      if (k == 0) {
        W.write_lastcolumn<0>(dnow, len, 1);
        T.data<0,0>(0,0) = -tau;
        zgemv_("N", n, n-1, -tau, D0+n, n, dnow, 1, 0.0, YD.block(0,0), 1);
        zgemv_("N", n, n-1, -tau, D1+n, n, dnow, 1, 0.0, YE.block(0,0), 1);
        // update D0
        zaxpy_(n-1, conj(YD.data<0,0>(0,0)), dnow, 1, D0+1, 1);
      } else {
        R.append_row<0>();
        SuperMatrix<3,1> x(work2_3nb, nb, 1);

        SuperMatrix<1,1> v(work1_3n, n, 1, n, 1);
        v.write_lastcolumn<0>(dnow, len, k+1);
        contract<true>("C", -tau, W, v, x);

        SuperMatrix<3,1> v2(work1_3n, nb, 1);
        contract_tr<false>("N", 1.0, T, x, v2, work4_nb);
        T.append_column<0>(v2);
        T.append_row<0>(0, k, -tau);

        SuperMatrix<1,1> v3(work3_nb, nb, 1);
        contract_tr<false>("N", 1.0, S, x, v3, work4_nb);
        S.append_column<0>(v3);

        W.append_column<0>(dnow, len, k+1);
      }
    }

    double c;
    complex<double> s, dum;
    zlartg_(D0[k+1+k*n], alpha, c, s, dum);
    assert(abs(-conj(s)*D0[k+1+k*n]+c*alpha) < 1.0e-10);
    D0[k+1+k*n] = c*D0[k+1+k*n] + s*alpha;

    const double cbar = c-1.0;
    const complex<double> sbar = conj(s);
    if (k == 0) {
      assert(abs(W.data<0,0>(1,0)-1.0)<1.0e-10);
      T.data<0,1>(0,0) = T.data<0,0>(0,0)*cbar;
      T.data<1,1>(0,0) = cbar;
      W.write_lastcolumn<1>(0,1);
      R.data<0,0>(0,0) = T.data<0,0>(0,0);
      R.data<1,0>(0,0) = 1.0;
      S.data<0,1>(0,0) = -sbar;

      auto YD0 = YD.slice_column<0>();
      auto YE0 = YE.slice_column<0>();
      YD.add_lastcolumn<1>(YD0,  cbar);
      YE.add_lastcolumn<1>(YE0,  cbar);
      ZD.add_lastcolumn<1>(YD0, -conj(sbar));
      ZE.add_lastcolumn<1>(YE0, -conj(sbar));
      zaxpy_(n,        cbar, D0+n, 1, YD.block(0,1), 1);
      zaxpy_(n,        cbar, D1+n, 1, YE.block(0,1), 1);
      zaxpy_(n, -conj(sbar), D0+n, 1, ZD.block(0,1), 1);
      zaxpy_(n, -conj(sbar), D1+n, 1, ZE.block(0,1), 1);
//debug
X.reset();
contract<false,true>("N", "C", 1.0, YD, W, X);
ZE.conj();
contract<false,true>("N", "C", 1.0, ZE, W, X);
ZE.conj();
cout << "aa" << endl;
X.print();
X.reset();
contract<false,true>("N", "C", 1.0, YE, W, X);
ZD.conj();
contract<false,true>("N", "C", -1.0, ZD, W, X);
ZD.conj();
cout << "aa2" << endl;
X.print();
    } else {
      SuperMatrix<3,1> x(work2_3nb, nb, 1);
      SuperMatrix<3,1> v(work4_nb, nb, 1);
      W.cut_row<0>(k+1, x);
      contract_tr<false>("N", 1.0, T, x, v, work4_nb);

      SuperMatrix<1,1> y(work3_nb, nb, 1);
      contract_tr<false>("N", 1.0, S, x, y, work4_nb);
      y.conj();
      x.reset();
      contract_tr<false>("N", sbar, R, y, x, work4_nb);
      y.conj();
      T.append_column<1>(x);
      T.add_lastcolumn<1>(v, cbar);
      T.append_row<1>(1, k, cbar);

      R.append_column<0>(v);
      R.append_row<1>(0,k,1.0);
      y.scale(cbar);
      S.append_column<1>(y);
      S.append_row<0>(1, k, -sbar);

      W.append_column_unit<1>(k+1);
    }

    // Householder to fix top half in column k
    if (len > 1) {
      complex<double>* dnow = D0+n*k+k+1;
      complex<double> ctau;
      complex<double> alpha2 = dnow[0];
      dnow[0] = 1.0;
      zlarfg_(len, alpha2, dnow+1, 1, ctau);

      if (k == 0) {
        W.write_lastcolumn<2>(dnow, len, 1);
        const complex<double> zz = -ctau*zdotc_(len, W.block(0,0)+1, 1, dnow, 1);
        T.data<0,2>(0,0) = zz*T.data<0,0>(0,0) -ctau*T.data<0,1>(0,0);
        T.data<1,2>(0,0) = -ctau*T.data<1,1>(0,0);
        T.data<2,2>(0,0) = -ctau;
        S.data<0,2>(0,0) = -ctau*S.data<0,1>(0,0);

        zgemv_("N", n, n-1, -ctau, D0+n, n, dnow, 1, 0.0, YD.block(0,2), 1);
        zgemv_("N", n, n-1, -ctau, D1+n, n, dnow, 1, 0.0, YE.block(0,2), 1);
        YD.add_lastcolumn<2>(YD.slice_column<0>(), zz);
        YE.add_lastcolumn<2>(YE.slice_column<0>(), zz);
        YD.add_lastcolumn<2>(YD.slice_column<1>(), -ctau);
        YE.add_lastcolumn<2>(YE.slice_column<1>(), -ctau);
        ZD.add_lastcolumn<2>(ZD.slice_column<1>(), -conj(ctau));
        ZE.add_lastcolumn<2>(ZE.slice_column<1>(), -conj(ctau));
//debug
X.reset();
contract<false,true>("N", "C", 1.0, YD, W, X);
ZE.conj();
contract<false,true>("N", "C", 1.0, ZE, W, X);
ZE.conj();
cout << "bb" << endl;
X.print();
X.reset();
contract<false,true>("N", "C", 1.0, YE, W, X);
ZD.conj();
contract<false,true>("N", "C", -1.0, ZD, W, X);
ZD.conj();
cout << "bb2" << endl;
X.print();
      } else {
        R.append_row<2>();
        SuperMatrix<3,1> x(work2_3nb, nb, 1);

        SuperMatrix<1,1> v(buf.get(), n, 1, n, 1);
        v.write_lastcolumn<0>(dnow, len, k+1);
        contract<true>("C", -ctau, W, v, x);

        SuperMatrix<3,1> v2(buf.get(), nb, 1);
        contract_tr<false>("N", 1.0, T, x, v2, work4_nb);
        T.append_column<2>(v2);
        T.append_row<2>(2, k, -ctau);

        SuperMatrix<1,1> v3(buf.get(), nb, 1);
        contract_tr<false>("N", 1.0, S, x, v3, work4_nb);
        S.append_column<2>(v3);

        W.append_column<2>(dnow, len, k+1);
      }
      dnow[0] = alpha2;
    }
  }
  print("tridiagonalization", Q0, Q1, T, W, R, S, W2, Y1, Y2, X, n);

  // diagonalize this tri-diagonal matrix (this step is much cheaper than
  // the Householder transformation above).
  unique_ptr<complex<double>[]> Cmat(new complex<double>[n*n]);
  int info;
  for (int i = 0; i != n; ++i)
    for (int j = 0; j <= i; ++j)
      D0[i-j+j*n] = D0[i+j*n];
  zhbev_("V", "L", n, 1, D0, n, eig, Cmat.get(), n, work1_3n, reinterpret_cast<double*>(work1_3n+n), info);
  if (info) throw runtime_error("zhbev failed in quaternion diagonalization");

  // form the coefficient matrix in D
  zgemm3m_("N", "N", n, n, n, 1.0, Q0, n, Cmat.get(), n, 0.0, D, n2);
  zgemm3m_("N", "N", n, n, n, 1.0, Q1, n, Cmat.get(), n, 0.0, D+n, n2);

  // eigen vectors using symmetry
  for (int i = 0; i != n; ++i) {
    for (int j = 0; j != n; ++j) {
       D[j+n2*(i+n)] = -conj(D[j+n+n2*i]);
       D[j+n+n2*(i+n)] = conj(D[j+n2*i]);
    }
  }
}

}
