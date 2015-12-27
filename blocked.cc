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
void print(string label, const SuperMatrix<3,3>& T, const SuperMatrix<1,3>& W,
                         const SuperMatrix<3,1>& R, const SuperMatrix<1,3>& S, const int n) {

  unique_ptr<complex<double>[]> xY1(new complex<double>[n*n]);
  unique_ptr<complex<double>[]> xY2(new complex<double>[n*n]);
  unique_ptr<complex<double>[]> xX(new complex<double>[n*n]);
  unique_ptr<complex<double>[]> xW2(new complex<double>[3*n*n]);
  SuperMatrix<1,1> X(xX.get(), n, n, n, n);
  SuperMatrix<3,1> W2(xW2.get(), n, n);
  SuperMatrix<1,1> Y1(xY1.get(), n, n);
  SuperMatrix<1,1> Y2(xY2.get(), n, n);

  {
    cout << setprecision(4) << "------ Q0 reconst " << label << endl;
    contract<_N,_C> (1.0, T, W, W2);
    contract<_N,_N>(1.0, W, W2, X);
    X.print();
  }
  {
    cout << setprecision(4) << "------ Q1 reconst " << label << endl;
    X.reset();
    contract<_N,_N>(1.0, W, R, Y1);
    contract<_N,_C>(1.0, W, S, Y2);
    Y1.conj();
    contract<_N,_C>(-1.0, Y1, Y2, X);
    X.print();
  }
}

// implementation...

void zquatev(const int n2, complex<double>* const D, const int ld2, double* const eig) {
  assert(n2 % 2 == 0);
  const int n = n2/2;
  const int ld = ld2/2;

  // rearrange data
  complex<double>* const D0 = D;
  complex<double>* const D1 = D + n*ld;
  copy_n(D, ld2*n, D+ld2*n);
  for (int i = 0; i != n; ++i) {
    copy_n(D+ld2*n+i*ld2, n, D0+i*ld);
    copy_n(D+ld2*n+i*ld2+n, n, D1+i*ld);
  }

  // identity matrix of n2 dimension
  complex<double>* const Q0 = D + ld2*n;
  complex<double>* const Q1 = D + ld2*n + ld*n;
  fill_n(Q0, ld*n, 0.0);
  fill_n(Q1, ld*n, 0.0);
  for (int i = 0; i != n; ++i) Q0[i+ld*i] = 1.0;

  // will be updated
  const int nb = n;

  unique_ptr<complex<double>[]> buf(new complex<double>[max(n-1,nb)*3]);
  unique_ptr<complex<double>[]> buf2(new complex<double>[nb*3]);
  unique_ptr<complex<double>[]> buf3(new complex<double>[nb]);
  unique_ptr<complex<double>[]> buf4(new complex<double>[nb]);

  auto work1_3n  = buf.get();
  auto work2_3nb = buf2.get();
  auto work3_nb  = buf3.get();
  auto work4_nb  = buf4.get();

  unique_ptr<complex<double>[]> xT(new complex<double>[9*nb*nb]);
  unique_ptr<complex<double>[]> xR(new complex<double>[3*nb*nb]);
  unique_ptr<complex<double>[]> xS(new complex<double>[3*nb*nb]);
  unique_ptr<complex<double>[]> xW(new complex<double>[2*nb*n]);
  unique_ptr<complex<double>[]> xYD(new complex<double>[(1+2*nb)*n]);
  unique_ptr<complex<double>[]> xYE(new complex<double>[(1+2*nb)*n]);
  unique_ptr<complex<double>[]> xZD(new complex<double>[(1+2*nb)*n]);
  unique_ptr<complex<double>[]> xZE(new complex<double>[(1+2*nb)*n]);
  SuperMatrix<3,3> T(xT.get(), nb, nb);
  SuperMatrix<3,1> R(xR.get(), nb, nb);
  SuperMatrix<1,3> S(xS.get(), nb, nb);
  SuperMatrix<1,2> W(xW.get(), n-1, nb, n-1, 1);
  SuperMatrix<1,3> YD(xYD.get(), n-1, nb, n-1, 1, true, true);
  SuperMatrix<1,3> ZD(xZD.get(), n-1, nb, n-1, 1, true, true);
  SuperMatrix<1,3> YE(xYE.get(), n-1, nb, n-1, 1, true, true);
  SuperMatrix<1,3> ZE(xZE.get(), n-1, nb, n-1, 1, true, true);

  // TODO once the paper is out, add comments with equation numbers.
  for (int k = 0; k != n; ++k) {
    // prepare the k-th column using compact WY-like representation
    if (k > 0) {
      SuperMatrix<1,1> d(D0+k*ld+1, n-1, 1, n-1, 1, false);
      SuperMatrix<1,1> e(D1+k*ld+1, n-1, 1, n-1, 1, false);

      SuperMatrix<3,1> x(work2_3nb, W.mptr(0), 1, W.mptr(0), 1);
      x.nptr(2) = 1;
      SuperMatrix<2,1> x2 = x.trunc<2>();
      W.cut_row<0>(k-1, x2);
      x.data<2,0>(0,0) = 1.0;
      contract<_N>(1.0, YD, x, d);
      contract<_N>(1.0, YE, x, e);

      SuperMatrix<1,1> y(work1_3n, n-1, 1);
      contract<_N>(1.0, ZE, x, y);
      d.add_lastcolumn<0>(y);
      y.reset();
      contract<_N>(-1.0, ZD, x, y);
      e.add_lastcolumn<0>(y);

      const int sf = k-1;
      auto ds = d.shift(sf);
      auto es = e.shift(sf);
      auto Ws = W.shift(sf);
      SuperMatrix<3,1> y1(work1_3n, k, 1, k, 1);
      auto y1s = y1.trunc<2>();
      contract<_C>(1.0, W, d, y1s);
      zaxpy_(k, 1.0, d.block(0,0), 1, y1.ptr<2,0>(0,0), 1);
      SuperMatrix<3,1> y2(work2_3nb, k, 1);
      contract_tr<_C>(1.0, T, y1, y2, work4_nb);
      contract<_N>(1.0, Ws, y2.trunc<2>(), ds);
      ds.data<0,0>(0,0) += y2.data<2,0>(k-1,0);
      SuperMatrix<1,1> y3(work3_nb, k, 1);
      contract_tr<_C>(1.0, R, y1, y3, work4_nb);
      y3.conj();
      y2.reset();
      contract_tr<_C>(1.0, S, y3, y2, work4_nb);
      SuperMatrix<1,1> y4(work1_3n, n-sf, 1);
      contract<_N>(1.0, Ws, y2.trunc<2>(), y4);
      y4.data<0,0>(0,0) += y2.data<2,0>(k-1,0);
      y4.conj();

      SuperMatrix<3,1> y5(work2_3nb, k, 1, k, 1);
      auto y5s = y5.trunc<2>();
      contract<_T>(1.0, W, e, y5s);
      zaxpy_(k, 1.0, e.block(0,0), 1, y5.ptr<2,0>(0,0), 1);
      es.add_lastcolumn<0>(y4);

      SuperMatrix<3,1> y5x(work1_3n, y5);
      SuperMatrix<1,1> y6(work3_nb, k, 1);
      contract_tr<_T>(1.0, R, y5x, y6, work4_nb);
      SuperMatrix<3,1> y7(work2_3nb, k, 1);
      contract_tr<_C>(1.0, S, y6, y7, work4_nb);
      contract<_N>(-1.0, Ws, y7.trunc<2>(), ds);
      ds.data<0,0>(0,0) -= y7.data<2,0>(k-1,0);
      y7.reset();
      contract_tr<_T>(1.0, T, y5x, y7, work4_nb);
      y7.conj();
      SuperMatrix<1,1> y8(work1_3n, n-sf, 1);
      contract<_N>(1.0, Ws, y7.trunc<2>(), y8);
      y8.data<0,0>(0,0) += y7.data<2,0>(k-1,0);
      y8.conj();
      es.add_lastcolumn<0>(y8);
    }
    if (k == n-1) break;

    const int len = n-k-1;
    complex<double> alpha = D1[k*ld+k+1];
    if (len > 1) {
      complex<double>* dnow = D1+k*ld+k+1;
      complex<double> tau;
      dnow[0] = 1.0;
      zlarfg_(len, alpha, dnow+1, 1, tau);
      tau = conj(tau);
      conj_n(dnow, len);

      if (k == 0) {
        W.write_lastcolumn<0>(dnow, len);
        T.data<0,0>(0,0) = -tau;
        zgemv_("N", len, len, -tau, D0+ld+1, ld, dnow, 1, 0.0, YD.block(0,0), 1);
        zgemv_("N", len, len, -tau, D1+ld+1, ld, dnow, 1, 0.0, YE.block(0,0), 1);

        zaxpy_(len, -conj(tau)*zdotc_(len, dnow, 1, D0+1, 1), dnow, 1, D0+1, 1);
      } else {
        R.append_row<0>();
        SuperMatrix<2,1> x(work2_3nb, k, 1);

        SuperMatrix<1,1> v(work1_3n, n-k-1, 1, n-k-1, 1);
        v.write_lastcolumn<0>(dnow, len);
        contract<_C>(-tau, W.shift(k), v, x);

        SuperMatrix<3,1> v2(work1_3n, k, 1);
        contract_tr<_N>(1.0, T.slice<0,2>(), x, v2, work4_nb);
        T.append_column<0>(v2);
        T.append_row<0,0>(k, -tau);

        SuperMatrix<1,1> v3(work3_nb, k, 1);
        contract_tr<_N>(1.0, S.slice<0,2>(), x, v3, work4_nb);
        S.append_column<0>(v3);
        W.append_column<0>(dnow, len, k);

        SuperMatrix<1,1> yx(work1_3n, n-1, 1);
        contract<_N>(1.0, YD.slice<0,2>(), x, yx);
        YD.append_column<0>(yx);
        zgemv_("N", n-1, len, -tau, D0+(k+1)*ld+1, ld, dnow, 1, 1.0, YD.ptr<0,0>(0,k), 1);
        yx.reset();
        contract<_N>(1.0, YE.slice<0,2>(), x, yx);
        YE.append_column<0>(yx);
        zgemv_("N", n-1, len, -tau, D1+(k+1)*ld+1, ld, dnow, 1, 1.0, YE.ptr<0,0>(0,k), 1);
        yx.reset();
        contract<_N>(1.0, ZD.slice<0,2>(), x, yx);
        ZD.append_column<0>(yx);
        yx.reset();
        contract<_N>(1.0, ZE.slice<0,2>(), x, yx);
        ZE.append_column<0>(yx);

        zaxpy_(len, -conj(tau)*zdotc_(len, dnow, 1, D0+k*ld+k+1, 1), dnow, 1, D0+k*ld+k+1, 1);
      }
    }

    double c;
    complex<double> s, dum;
    zlartg_(D0[k+1+k*ld], alpha, c, s, dum);
    assert(abs(-conj(s)*D0[k+1+k*ld]+c*alpha) < 1.0e-10);
    D0[k+1+k*ld] = c*D0[k+1+k*ld] + s*alpha;

    const double cbar = c-1.0;
    const complex<double> sbar = conj(s);
    if (k == 0) {
      assert(abs(W.data<0,0>(0,0)-1.0)<1.0e-10);
      T.data<0,2>(0,0) = T.data<0,0>(0,0)*cbar;
      T.data<2,2>(0,0) = cbar;
      R.data<0,0>(0,0) = T.data<0,0>(0,0);
      R.data<2,0>(0,0) = 1.0;
      S.data<0,2>(0,0) = -sbar;

      auto YD0 = YD.slice<0>();
      auto YE0 = YE.slice<0>();
      YD.add_lastcolumn<2>(YD0,  cbar);
      YE.add_lastcolumn<2>(YE0,  cbar);
      zaxpy_(n-1, cbar, D0+ld+1, 1, YD.block(0,2), 1);
      zaxpy_(n-1, cbar, D1+ld+1, 1, YE.block(0,2), 1);
      SuperMatrix<1,1> yd0b(work1_3n,     YD0);
      SuperMatrix<1,1> ye0b(work1_3n+n-1, YE0);
      zaxpy_(n-1, 1.0, D0+ld+1, 1, yd0b.block(0,0), 1);
      zaxpy_(n-1, 1.0, D1+ld+1, 1, ye0b.block(0,0), 1);
      yd0b.conj();
      ye0b.conj();
      ZD.add_lastcolumn<2>(yd0b, -sbar);
      ZE.add_lastcolumn<2>(ye0b, -sbar);
    } else {
      SuperMatrix<2,1> x(work2_3nb, k+1, 1);
      W.cut_row<0>(k, x);

      SuperMatrix<1,1> yx(work1_3n,     n-1, 1);
      SuperMatrix<1,1> zx(work1_3n+n-1, n-1, 1);
      contract<_N>(1.0, YD.slice<0,2>(), x, yx);
      contract<_N>(1.0, ZD.slice<0,2>(), x, zx);
      fill_n(YD.block(0,2), n-1, 0.0);
      fill_n(ZD.block(0,2), n-1, 0.0);
      ZD.add_lastcolumn<2>(zx, cbar);
      zx.conj();
      YD.add_lastcolumn<2>(zx, sbar);
      zaxpy_(n-1, 1.0, D0+(k+1)*ld+1, 1, yx.block(0,0), 1);
      YD.add_lastcolumn<2>(yx, cbar);
      yx.conj();
      ZD.add_lastcolumn<2>(yx, -sbar);

      yx.reset();
      zx.reset();
      contract<_N>(1.0, YE.slice<0,2>(), x, yx);
      contract<_N>(1.0, ZE.slice<0,2>(), x, zx);
      fill_n(YE.block(0,2), n-1, 0.0);
      fill_n(ZE.block(0,2), n-1, 0.0);
      ZE.add_lastcolumn<2>(zx, cbar);
      zx.conj();
      YE.add_lastcolumn<2>(zx, sbar);
      zaxpy_(n-1, 1.0, D1+(k+1)*ld+1, 1, yx.block(0,0), 1);
      YE.add_lastcolumn<2>(yx, cbar);
      yx.conj();
      ZE.add_lastcolumn<2>(yx, -sbar);

      SuperMatrix<3,1> v(work1_3n, k+1, 1);
      contract_tr<_N>(1.0, T.slice<0,2>(), x, v, work4_nb);
      SuperMatrix<1,1> y(work3_nb, k+1, 1);
      contract_tr<_N>(1.0, S.slice<0,2>(), x, y, work4_nb);
      y.conj();
      SuperMatrix<3,1> z(work2_3nb, k+1, 1);
      contract_tr<_N>(sbar, R, y, z, work4_nb);
      y.conj();
      T.append_column<2>(z);
      T.add_lastcolumn<2>(v, cbar);
      T.append_row<2,2>(k, cbar);

      R.append_column<0>(v);
      R.append_row<2,0>(k,1.0);
      y.scale(cbar);
      S.append_column<2>(y);
      S.append_row<0,2>(k, -sbar);
    }

    if (len > 1) {
      complex<double>* dnow = D0+k*ld+k+1;
      complex<double> ctau;
      complex<double> alpha2 = dnow[0];
      dnow[0] = 1.0;
      zlarfg_(len, alpha2, dnow+1, 1, ctau);

      if (k == 0) {
        W.write_lastcolumn<1>(dnow, len);
        const complex<double> zz = -ctau*zdotc_(len, W.block(0,0), 1, dnow, 1);
        T.data<0,1>(0,0) = zz*T.data<0,0>(0,0) -ctau*T.data<0,2>(0,0);
        T.data<2,1>(0,0) = -ctau*T.data<2,2>(0,0);
        T.data<1,1>(0,0) = -ctau;
        S.data<0,1>(0,0) = -ctau*S.data<0,2>(0,0);

        zgemv_("N", n-1, n-1, -ctau, D0+ld+1, ld, dnow, 1, 0.0, YD.block(0,1), 1);
        zgemv_("N", n-1, n-1, -ctau, D1+ld+1, ld, dnow, 1, 0.0, YE.block(0,1), 1);
        YD.add_lastcolumn<1>(YD.slice<0>(), zz);
        YE.add_lastcolumn<1>(YE.slice<0>(), zz);
        YD.add_lastcolumn<1>(YD.slice<2>(), -ctau);
        YE.add_lastcolumn<1>(YE.slice<2>(), -ctau);
        ZD.add_lastcolumn<1>(ZD.slice<2>(), -ctau);
        ZE.add_lastcolumn<1>(ZE.slice<2>(), -ctau);
      } else {
        R.append_row<1>();
        SuperMatrix<3,1> x(work2_3nb, k+1, 1, k+1, 1);
        x.nptr(1) = k;

        SuperMatrix<1,1> v(work1_3n, n-k-1, 1, n-k-1, 1);
        v.write_lastcolumn<0>(dnow, len);
        SuperMatrix<2,1> x2 = x.trunc<2>();
        contract<_C>(-ctau, W.shift(k), v, x2);
        x.data<2,0>(k,0) = -ctau;

        SuperMatrix<3,1> v2(work1_3n, k+1, 1);
        contract_tr<_N>(1.0, T, x, v2, work4_nb);
        T.append_column<1>(v2);
        T.append_row<1,1>(k, -ctau);

        SuperMatrix<1,1> v3(work3_nb, k+1, 1);
        contract_tr<_N>(1.0, S, x, v3, work4_nb);
        S.append_column<1>(v3);
        W.append_column<1>(dnow, len, k);

        x.data<2,0>(0,0) = -ctau;
        x.nptr(2) = 1;

        SuperMatrix<1,1> yx(work1_3n, n-1, 1);
        contract<_N>(1.0, YD, x, yx);
        YD.append_column<1>(yx);
        zgemv_("N", n-1, len, -ctau, D0+(k+1)*ld+1, ld, dnow, 1, 1.0, YD.ptr<0,1>(0,k), 1);
        yx.reset();
        contract<_N>(1.0, YE, x, yx);
        YE.append_column<1>(yx);
        zgemv_("N", n-1, len, -ctau, D1+(k+1)*ld+1, ld, dnow, 1, 1.0, YE.ptr<0,1>(0,k), 1);
        yx.reset();
        contract<_N>(1.0, ZD, x, yx);
        ZD.append_column<1>(yx);
        yx.reset();
        contract<_N>(1.0, ZE, x, yx);
        ZE.append_column<1>(yx);
      }
      dnow[0] = alpha2;
    }
  }
//print("tridiagonalization", T, W, R, S, n);

  // diagonalize this tri-diagonal matrix (this step is much cheaper than
  // the Householder transformation above).
  unique_ptr<complex<double>[]> Cmat(new complex<double>[n*n]);
  int info;
  zhbev_("V", "L", n, 1, D0, ld+1, eig, Cmat.get(), n, work1_3n, reinterpret_cast<double*>(work1_3n+n), info);
  if (info) throw runtime_error("zhbev failed in quaternion diagonalization");

  // form the coefficient matrix in D
  zgemm3m_("N", "N", n, n, n, 1.0, Q0, ld, Cmat.get(), n, 0.0, D, ld2);
  zgemm3m_("N", "N", n, n, n, 1.0, Q1, ld, Cmat.get(), n, 0.0, D+ld, ld2);

  // eigen vectors using symmetry
  for (int i = 0; i != n; ++i) {
    for (int j = 0; j != n; ++j) {
       D[j+ld2*(i+n)] = -conj(D[j+ld+ld2*i]);
       D[j+ld+ld2*(i+n)] = conj(D[j+ld2*i]);
    }
  }
}

}
