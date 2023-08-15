/****************************************************************************
 * Copyright (c) 2023 by the ArborX authors                                 *
 * All rights reserved.                                                     *
 *                                                                          *
 * This file is part of the ArborX library. ArborX is                       *
 * distributed under a BSD 3-clause license. For the licensing terms see    *
 * the LICENSE file in the top-level directory.                             *
 *                                                                          *
 * SPDX-License-Identifier: BSD-3-Clause                                    *
 ****************************************************************************/

#pragma once

#include <ArborX_DetailsKokkosExtAccessibilityTraits.hpp>

#include <Kokkos_Core.hpp>

#include <cassert>
#include <cmath>
#include <limits>

// Pseudo-inverse moment matrix using SVD
// We must find U, E (diagonal and positive) and V such that A = U.E.V^T
// We also know that A is symmetric (by construction), so U = SV where S is
// a sign matrix (only 1 or -1 in the diagonal, 0 elsewhere).
// Thus A = U.E.S.U^T
template <class ValueType, typename MemorySpace>
class SymmPseudoInverseSVD
{
public:
  template <typename ExecutionSpace>
  static Kokkos::View<ValueType ***, MemorySpace>
  computePseudoInverses(ExecutionSpace const &space,
                        Kokkos::View<ValueType ***, MemorySpace> const &mats)
  {
    static_assert(
        KokkosExt::is_accessible_from<MemorySpace, ExecutionSpace>::value);

    SymmPseudoInverseSVD spis(space, mats);

    // Iterative approach, we will "deconstruct" E.S until only the diagonal
    // is relevent inside the matrix
    // It is possible to prove that, at each step, the "norm" of the matrix
    // is strictly less that of the previous
    Kokkos::parallel_for(
        "Example::SVD::compute_U_ES",
        Kokkos::RangePolicy<ExecutionSpace>(space, 0, spis._num_matrices),
        KOKKOS_LAMBDA(int const i) {
          int p, q;
          ValueType norm = spis.argmaxOffDiagonal(i, p, q);
          while (norm > spis._epsilon)
          {
            spis.computeUESSingle(i, p, q);
            norm = spis.argmaxOffDiagonal(i, p, q);
          }
        });

    // From the SVD results, the pseudo inverse would be
    // U . [ E^-1.S ] . U^T
    Kokkos::parallel_for(
        "Example::SVD::fill_inv",
        Kokkos::MDRangePolicy<Kokkos::Rank<3>>(
            space, {0, 0, 0}, {spis._num_matrices, spis._size, spis._size}),
        KOKKOS_LAMBDA(int const i, int const j, int const k) {
          spis.fillInv(i, j, k);
        });

    return spis._inv;
  }

private:
  // U and E.S are computed, we can now build the inverse
  // U . [ E^-1.S ] . U^T
  KOKKOS_FUNCTION void fillInv(int const i, int const j, int const k) const
  {
    ValueType value = _zero;
    for (int l = 0; l < _size; l++)
    {
      ValueType v = _es(i, l, l);
      if (Kokkos::abs(v) > _epsilon)
      {
        value += _u(i, j, l) * _u(i, k, l) / v;
      }
    }

    _inv(i, j, k) = value;
  }

  // We found the biggest value in our off-diagonal. We will remove it by
  // computing a "local" svd and update U and E.S
  KOKKOS_FUNCTION void computeUESSingle(int const i, int const p,
                                        int const q) const
  {
    ValueType a = _es(i, p, p);
    ValueType b = _es(i, p, q);
    ValueType c = _es(i, q, q);

    // Our submatrix is now
    // +----------+----------+   +---+---+
    // | es(p, p) | es(p, q) |   | a | b |
    // +----------+----------+ = +---+---+
    // | es(q, p) | es(q, q) |   | b | c |
    // +----------+----------+   +---+---+

    // Lets compute u, v and theta such that
    // +---+---+              +---+---+
    // | a | b |              | u | 0 |
    // +---+---+ = R(theta) * +---+---+ * R(theta)^T
    // | b | c |              | 0 | v |
    // +---+---+              +---+---+

    ValueType theta, u, v;
    if (a == c) // <-- better to check if |a - c| < epsilon?
    {
      theta = _pi_4;
      u = a + b;
      v = a - b;
    }
    else
    {
      theta = _half * Kokkos::atan((_two * b) / (a - c));
      ValueType a_c_cos2 = (a - c) / Kokkos::cos(_two * theta);
      u = _half * (a + c + a_c_cos2);
      v = _half * (a + c - a_c_cos2);
    }
    ValueType cos = Kokkos::cos(theta);
    ValueType sin = Kokkos::sin(theta);

    // Now lets compute the following new values for U amd E.S
    // E.S <- R'(theta)^T . E.S . R'(theta)
    // U  <- U . R'(theta)

    // R'(theta)^T . E.S
    for (int j = 0; j < _size; j++)
    {
      ValueType es_ipj = _es(i, p, j);
      ValueType es_iqj = _es(i, q, j);
      _es(i, p, j) = cos * es_ipj + sin * es_iqj;
      _es(i, q, j) = -sin * es_ipj + cos * es_iqj;
    }

    // [R'(theta)^T . E.S] . R'(theta)
    for (int j = 0; j < _size; j++)
    {
      ValueType es_ijp = _es(i, j, p);
      ValueType es_ijq = _es(i, j, q);
      _es(i, j, p) = cos * es_ijp + sin * es_ijq;
      _es(i, j, q) = -sin * es_ijp + cos * es_ijq;
    }

    // U . R'(theta)
    for (int j = 0; j < _size; j++)
    {
      ValueType u_ijp = _u(i, j, p);
      ValueType u_ijq = _u(i, j, q);
      _u(i, j, p) = cos * u_ijp + sin * u_ijq;
      _u(i, j, q) = -sin * u_ijp + cos * u_ijq;
    }

    // These should theorically hold but is it ok to force them to their
    // real value?
    _es(i, p, p) = u;
    _es(i, q, q) = v;
    _es(i, p, q) = _zero;
    _es(i, q, p) = _zero;
  }

  // This finds the biggest off-diagonal value of E.S as well as its
  // coordinates. Being symmetric, we can always check on the upper
  // triangle (and always have q > p)
  KOKKOS_FUNCTION ValueType argmaxOffDiagonal(int const i, int &p, int &q) const
  {
    ValueType max = _zero;
    p = q = 0;
    for (int j = 0; j < _size; j++)
    {
      for (int k = j + 1; k < _size; k++)
      {
        ValueType val = Kokkos::abs(_es(i, j, k));
        if (max < val)
        {
          max = val;
          p = j;
          q = k;
        }
      }
    }

    return max;
  }

  template <typename ExecutionSpace>
  SymmPseudoInverseSVD(ExecutionSpace const &space,
                       Kokkos::View<ValueType ***, MemorySpace> const &mats)
      : _num_matrices(mats.extent(0))
      , _size(mats.extent(1))
  {
    // mats must be an array of (symmetric) square matrices
    assert(mats.extent(1) == mats.extent(2));

    _es = Kokkos::View<ValueType ***, MemorySpace>(
        Kokkos::view_alloc(Kokkos::WithoutInitializing, "Example::SVD::ES"),
        mats.layout());
    Kokkos::deep_copy(space, _es, mats);

    _u = Kokkos::View<ValueType ***, MemorySpace>(
        Kokkos::view_alloc(Kokkos::WithoutInitializing, "Example::SVD::U"),
        mats.layout());
    Kokkos::parallel_for(
        "Example::SVD::U_init",
        Kokkos::MDRangePolicy<Kokkos::Rank<3>>(space, {0, 0, 0},
                                               {_num_matrices, _size, _size}),
        KOKKOS_LAMBDA(int const i, int const j, int const k) {
          _u(i, j, k) = ValueType((j == k));
        });

    _inv = Kokkos::View<ValueType ***, MemorySpace>(
        Kokkos::view_alloc(Kokkos::WithoutInitializing, "Example::SVD::inv"),
        mats.layout());
  }

  Kokkos::View<ValueType ***, MemorySpace> _es;
  Kokkos::View<ValueType ***, MemorySpace> _u;
  Kokkos::View<ValueType ***, MemorySpace> _inv;
  std::size_t _num_matrices;
  std::size_t _size;

  static constexpr ValueType _pi_4 = ValueType(M_PI_4);
  static constexpr ValueType _epsilon =
      std::numeric_limits<ValueType>::epsilon();
  static constexpr ValueType _half = ValueType(0.5);
  static constexpr ValueType _two = ValueType(2);
  static constexpr ValueType _zero = ValueType(0);
};