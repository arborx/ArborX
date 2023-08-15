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

#include <ArborX.hpp>

#include <Kokkos_Core.hpp>

#include <cassert>

#include "symmetric_pseudoinverse_svd.hpp"

template <typename ValueType, typename PolynomialBasis, typename RBF,
          typename MemorySpace>
class MLSComputation
{
public:
  MLSComputation() = default;

  template <typename ExecutionSpace>
  MLSComputation(
      ExecutionSpace const &space,
      Kokkos::View<ArborX::Point *, MemorySpace> const &source_points,
      Kokkos::View<ArborX::Point *, MemorySpace> const &target_points)
      : _num_neighbors(source_points.extent(0) / target_points.extent(0))
      , _num_targets(target_points.extent(0))
  {
    // There must be a list of num_neighbors source points for each
    // target point
    assert(source_points.extent(0) == _num_targets * _num_neighbors);

    auto source_ref_target =
        translateToTarget(space, source_points, target_points);

    auto radii = computeRadii(space, source_ref_target);
    auto phi = computeWeight(space, source_ref_target, radii);
    auto p = computeVandermonde(space, source_ref_target);

    auto a = computeMoment(space, phi, p);
    auto a_inv =
        SymmPseudoInverseSVD<ValueType, MemorySpace>::computePseudoInverses(
            space, a);

    computeCoefficients(space, phi, p, a_inv);
  }

  template <typename ExecutionSpace>
  Kokkos::View<ValueType *>
  apply(ExecutionSpace const &space,
        Kokkos::View<ValueType *, MemorySpace> const &source_values)
  {
    assert(source_values.extent(0) == _num_targets * _num_neighbors);

    Kokkos::View<ValueType *, MemorySpace> target_values(
        "Example::MLSC::target_values", _num_targets);
    Kokkos::parallel_for(
        "Example::MLSC::target_interpolation",
        Kokkos::RangePolicy<ExecutionSpace>(space, 0, _num_targets),
        KOKKOS_LAMBDA(int const i) {
          ValueType tmp = _zero;
          for (int j = 0; j < _num_neighbors; j++)
          {
            tmp += _coeffs(i, j) * source_values(i * _num_neighbors + j);
          }
          target_values(i) = tmp;
        });

    return target_values;
  }

private:
  template <typename ExecutionSpace>
  Kokkos::View<ArborX::Point **, MemorySpace> translateToTarget(
      ExecutionSpace const &space,
      Kokkos::View<ArborX::Point *, MemorySpace> const &source_points,
      Kokkos::View<ArborX::Point *, MemorySpace> const &target_points)
  {
    // We center each group around the target as it ables you to
    // optimize the final computation
    Kokkos::View<ArborX::Point **, MemorySpace> source_ref_target(
        Kokkos::view_alloc(Kokkos::WithoutInitializing,
                           "Example::MLSC::source_ref_target"),
        _num_targets, _num_neighbors);
    Kokkos::parallel_for(
        "Example::MLSC::source_ref_target_fill",
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>(space, {0, 0},
                                               {_num_targets, _num_neighbors}),
        KOKKOS_LAMBDA(int const i, int const j) {
          ArborX::Point src = source_points(i * _num_neighbors + j);
          ArborX::Point tgt = target_points(i);
          source_ref_target(i, j) = ArborX::Point{
              src[0] - tgt[0],
              src[1] - tgt[1],
              src[2] - tgt[2],
          };
        });

    return source_ref_target;
  }

  template <typename ExecutionSpace>
  Kokkos::View<ValueType *, MemorySpace> computeRadii(
      ExecutionSpace const &space,
      Kokkos::View<ArborX::Point **, MemorySpace> const &source_ref_target)
  {
    Kokkos::View<ValueType *, MemorySpace> radii(
        Kokkos::view_alloc(Kokkos::WithoutInitializing, "Example::MLSC::radii"),
        _num_targets);
    Kokkos::parallel_for(
        "Example::MLSC::radii_computation",
        Kokkos::RangePolicy<ExecutionSpace>(space, 0, _num_targets),
        KOKKOS_LAMBDA(int const i) {
          ValueType radius = _ten * _epsilon;
          for (int j = 0; j < _num_neighbors; j++)
          {
            ValueType norm =
                ArborX::Details::distance(source_ref_target(i, j), _origin);
            radius = (radius < norm) ? norm : radius;
          }
          radii(i) = _one_extra * radius;
        });

    return radii;
  }

  template <typename ExecutionSpace>
  Kokkos::View<ValueType **, MemorySpace> computeWeight(
      ExecutionSpace const &space,
      Kokkos::View<ArborX::Point **, MemorySpace> const &source_ref_target,
      Kokkos::View<ValueType *, MemorySpace> const &radii)
  {
    Kokkos::View<ValueType **, MemorySpace> phi(
        Kokkos::view_alloc(Kokkos::WithoutInitializing, "Example::MLSC::phi"),
        _num_targets, _num_neighbors);
    Kokkos::parallel_for(
        "Example::MLSC::phi_computation",
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>(space, {0, 0},
                                               {_num_targets, _num_neighbors}),
        KOKKOS_LAMBDA(int const i, int const j) {
          ValueType norm =
              ArborX::Details::distance(source_ref_target(i, j), _origin);
          phi(i, j) = RBF::apply(norm / radii(i));
        });

    return phi;
  }

  template <typename ExecutionSpace>
  Kokkos::View<ValueType ***, MemorySpace> computeVandermonde(
      ExecutionSpace const &space,
      Kokkos::View<ArborX::Point **, MemorySpace> const &source_ref_target)
  {
    // Instead of relying on an external type, could it be produced
    // automatically?
    Kokkos::View<ValueType ***, MemorySpace> p(
        Kokkos::view_alloc(Kokkos::WithoutInitializing,
                           "Example::MLSC::vandermonde"),
        _num_targets, _num_neighbors, PolynomialBasis::size);
    Kokkos::parallel_for(
        "Example::MLSC::vandermonde_computation",
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>(space, {0, 0},
                                               {_num_targets, _num_neighbors}),
        KOKKOS_LAMBDA(int const i, int const j) {
          auto basis = PolynomialBasis::basis(source_ref_target(i, j));
          for (int k = 0; k < PolynomialBasis::size; k++)
          {
            p(i, j, k) = basis[k];
          }
        });

    return p;
  }

  template <typename ExecutionSpace>
  Kokkos::View<ValueType ***, MemorySpace>
  computeMoment(ExecutionSpace const &space,
                Kokkos::View<ValueType **, MemorySpace> const &phi,
                Kokkos::View<ValueType ***, MemorySpace> const &p)
  {
    Kokkos::View<ValueType ***, MemorySpace> a(
        Kokkos::view_alloc(Kokkos::WithoutInitializing,
                           "Example::MLSC::moment"),
        _num_targets, PolynomialBasis::size, PolynomialBasis::size);
    Kokkos::parallel_for(
        "Example::MLSC::moment_computation",
        Kokkos::MDRangePolicy<Kokkos::Rank<3>>(
            space, {0, 0, 0},
            {_num_targets, PolynomialBasis::size, PolynomialBasis::size}),
        KOKKOS_LAMBDA(int const i, int const j, int const k) {
          ValueType tmp = _zero;
          for (int l = 0; l < _num_neighbors; l++)
          {
            tmp += p(i, l, j) * p(i, l, k) * phi(i, l);
          }
          a(i, j, k) = tmp;
        });

    return a;
  }

  template <typename ExecutionSpace>
  void
  computeCoefficients(ExecutionSpace const &space,
                      Kokkos::View<ValueType **, MemorySpace> const &phi,
                      Kokkos::View<ValueType ***, MemorySpace> const &p,
                      Kokkos::View<ValueType ***, MemorySpace> const &a_inv)
  {
    _coeffs = Kokkos::View<ValueType **, MemorySpace>(
        Kokkos::view_alloc(Kokkos::WithoutInitializing,
                           "Example::MLSC::coefficients"),
        _num_targets, _num_neighbors);
    Kokkos::parallel_for(
        "Example::MLSC::coefficients",
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>(space, {0, 0},
                                               {_num_targets, _num_neighbors}),
        KOKKOS_LAMBDA(int const i, int const j) {
          ValueType tmp = _zero;
          for (int k = 0; k < PolynomialBasis::size; k++)
          {
            tmp += a_inv(i, 0, k) * p(i, j, k) * phi(i, j);
          }
          _coeffs(i, j) = tmp;
        });
  }

  Kokkos::View<ValueType **, MemorySpace> _coeffs;
  std::size_t _num_targets;
  std::size_t _num_neighbors;

  static constexpr ValueType _zero = ValueType(0);
  static constexpr ValueType _ten = ValueType(10);
  static constexpr ValueType _epsilon =
      std::numeric_limits<ValueType>::epsilon();
  static constexpr ValueType _one_extra = ValueType(1.1);
  static constexpr ArborX::Point _origin = ArborX::Point{0, 0, 0};
};