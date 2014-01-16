//
// Filename: zquartev.h
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

#ifndef __TS_ZQUARTEV_H
#define __TS_ZQUARTEV_H

#include <memory>
#include <complex>

namespace ts {

  // Diagonalize a quaternion matrix:
  //
  // (  A   B  )
  // ( -B*  A* )
  //
  // where A is Hermite and B is anti-Hermite. Only the left half of the matrix
  // will be referenced as an input.
  //
  // This matrix has doubly-degenerate eigenvalues, and there is a set of
  // eivenvectors that has the same symmetry property:
  //
  // (  U   V  )
  // ( -V*  U* )
  //
  // The function zquatev is an implementation that works sort of fine with
  // one CPU core.

  // TODO: efficiency, threading, and parallelization...
  // Goal: make it competitive with MKL's zheev with threading, and with pzheev(r) in parallel.
  // 
  // The current implementation is based on matlab code in arXiv:1203.6151v4
  // with a tiny bit of improvement on efficiency
  //
  extern void zquatev(const int n2, std::complex<double>* const D, double* const eig);
}

#endif
