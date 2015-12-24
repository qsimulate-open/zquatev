//
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

#include <cstdlib> // atoi()
#include <chrono>
#include <algorithm>
#include <stdexcept>
#include <iostream>
#include <iomanip>
#include "zquatev.h"
#include "f77.h"

using namespace std;

int main(int argc, char * argv[]) {

  const int n = (argc>1) ? atoi(argv[1]) : 200;
  const int n2 = n*2;

  bool printout = (n<=200);

  unique_ptr<complex<double>[]> A(new complex<double>[n*n]);
  unique_ptr<complex<double>[]> B(new complex<double>[n*n]);
  unique_ptr<complex<double>[]> C(new complex<double>[n2*n2]);
  unique_ptr<complex<double>[]> D(new complex<double>[n2*n2]);

  // some random matrices A (Hermite) and B (anti-Hermite)
  srand(32);
  for (int i = 0; i != n; ++i) {
    for (int j = 0; j <= i; ++j) {
      const array<double,4> t = {{rand()%10000 * 0.0001, rand()%10000 * 0.0001,
                                  rand()%10000 * 0.0001, rand()%10000 * 0.0001}};
      A[j+n*i] = j == i ? t[0] : complex<double>(t[0], t[1]);
      A[i+n*j] = conj(A[j+n*i]);
      B[j+n*i] = j == i ? 0.0 : complex<double>(t[2], t[3]);
      B[i+n*j] = -B[j+n*i];
    }
  }

  for (int i = 0; i != n; ++i) {
    for (int j = 0; j != n; ++j) {
      C[j+n2*i] = A[j+n*i];
      C[n+j+n2*(n+i)] = conj(A[j+n*i]);
      C[j+n2*(n+i)] = B[j+n*i];
      C[n+j+n2*(i)] = -conj(B[j+n*i]);
    }
  }
  copy_n(C.get(), n2*n2, D.get());

  cout << endl;
  cout << " **** using zheev **** " << endl;
  auto time0 = chrono::high_resolution_clock::now();
  {
    unique_ptr<double[]> eig(new double[n2]);
    const int lwork = 5*n2;
    unique_ptr<double[]> rwork(new double[lwork]);
    unique_ptr<complex<double>[]> work(new complex<double>[lwork]);
    int info;
    zheev_("V", "U", &n2, C.get(), &n2, eig.get(), work.get(), &lwork, rwork.get(), &info);
    if (info) throw runtime_error("zheev failed");
    if (printout) {
      for (int i = 0; i != n; ++i) {
        cout << fixed << setw(30) << setprecision(15) << eig[i*2];
        if (i % 5 == 4) cout << endl;
      }
      cout << endl;
      for (int i = 0; i != n; ++i) {
        cout << fixed << setw(30) << setprecision(15) << eig[i*2+1];
        if (i % 5 == 4) cout << endl;
      }
    }
  }

  cout << " **** using zquartev **** " << endl;
  auto time1 = chrono::high_resolution_clock::now();
  {
    unique_ptr<double[]> eig(new double[n2]);
    ts::zquatev(n2, D.get(), eig.get());

    if (printout) {
      cout << endl;
      for (int i = 0; i != n; ++i) {
        cout << fixed << setw(30) << setprecision(15) << eig[i];
        if (i % 5 == 4) cout << endl;
      }
    }
  }
  auto time2 = chrono::high_resolution_clock::now();

  cout << endl;
  cout << " zheev   : " << setw(10) << fixed << setprecision(2) << chrono::duration_cast<chrono::milliseconds>(time1-time0).count()*0.001 << endl;
  cout << " zquartev: " << setw(10) << fixed << setprecision(2) << chrono::duration_cast<chrono::milliseconds>(time2-time1).count()*0.001 << endl;

  return 0;
}
