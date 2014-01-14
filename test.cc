//
// Filename: test.cc 
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

#include <cstdlib> // atoi()
#include <chrono>
#include <algorithm>
#include <stdexcept>
#include <iostream>
#include <iomanip>
#include "zquartev.h" 
#include "f77.h"

using namespace std;

int main(int argc, char * argv[]) {

  const int n = (argc>1) ? atoi(argv[1]) : 200;
  const int n2 = n*2;

  unique_ptr<complex<double>[]> A(new complex<double>[n*n]);
  unique_ptr<complex<double>[]> B(new complex<double>[n*n]);
  unique_ptr<complex<double>[]> C(new complex<double>[n2*n2]);
  unique_ptr<complex<double>[]> D(new complex<double>[n2*n2]);

  // some random matrices A (Hermite) and B (anti-Hermite)
  srand(32);
  for (int i = 0; i != n; ++i) {
    for (int j = 0; j <= i; ++j) {
      const array<double,4> t = {{(double)rand()/(double)RAND_MAX, (double)rand()/(double)RAND_MAX,
                                  (double)rand()/(double)RAND_MAX, (double)rand()/(double)RAND_MAX}};
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

  cout << " **** using zquartev **** " << endl;
  auto time1 = chrono::high_resolution_clock::now();
  {
    unique_ptr<double[]> eig(new double[n2]);
    ts::zquatev(n2, D.get(), eig.get()); 

    cout << endl;
    for (int i = 0; i != n; ++i) {
      cout << fixed << setw(30) << setprecision(15) << eig[i];
      if (i % 5 == 4) cout << endl;
    }
  }
  auto time2 = chrono::high_resolution_clock::now();

  cout << " zheev   : " << setw(10) << setprecision(2) << chrono::duration_cast<chrono::milliseconds>(time1-time0).count()*0.001 << endl;
  cout << " zquartev: " << setw(10) << setprecision(2) << chrono::duration_cast<chrono::milliseconds>(time2-time1).count()*0.001 << endl;

  return 0;
}
