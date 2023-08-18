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

#include "DetailsSymmetricPseudoInverseSVD.hpp"

namespace Details
{

template <typename CoefficientType, typename MemorySpace>
class MovingLeastSquaresComputation
{
public:
  MovingLeastSquaresComputation() = default;

  template <typename ExecutionSpace, typename PolynomialBasis,
            typename RadialBasisFunction, typename SourcePoints,
            typename TargetPoints>
  MovingLeastSquaresComputation(ExecutionSpace const &space,
                                SourcePoints const &source_points,
                                TargetPoints const &target_points,
                                PolynomialBasis const &,
                                RadialBasisFunction const &)
  {
    using src_acc = ArborX::AccessTraits<SourcePoints, ArborX::PrimitivesTag>;
    using tgt_acc = ArborX::AccessTraits<TargetPoints, ArborX::PrimitivesTag>;

    _num_targets = tgt_acc::size(target_points);
    _num_neighbors = src_acc::size(source_points) / _num_targets;
    constexpr CoefficientType epsilon =
        std::numeric_limits<CoefficientType>::epsilon();
    constexpr ArborX::Point origin = ArborX::Point{0, 0, 0};

    // We center each group of points around the target as it ables us to
    // optimize the final computation and transfer point types into ours
    // TODO: Use multidimensional points!
    Kokkos::View<ArborX::Point **, MemorySpace> source_ref_target(
        Kokkos::view_alloc(Kokkos::WithoutInitializing,
                           "Example::MLSC::source_ref_target"),
        _num_targets, _num_neighbors);
    Kokkos::parallel_for(
        "Example::MLSC::source_ref_target_fill",
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>(space, {0, 0},
                                               {_num_targets, _num_neighbors}),
        KOKKOS_LAMBDA(int const i, int const j) {
          auto src = src_acc::get(source_points, i * _num_neighbors + j);
          auto tgt = tgt_acc::get(target_points, i);
          source_ref_target(i, j) = ArborX::Point{
              src[0] - tgt[0],
              src[1] - tgt[1],
              src[2] - tgt[2],
          };
        });

    // To properly use the RBF, we need to decide for a radius around each
    // target point that encapsulates all of the points
    Kokkos::View<CoefficientType *, MemorySpace> radii(
        Kokkos::view_alloc(Kokkos::WithoutInitializing, "Example::MLSC::radii"),
        _num_targets);
    Kokkos::parallel_for(
        "Example::MLSC::radii_computation",
        Kokkos::RangePolicy<ExecutionSpace>(space, 0, _num_targets),
        KOKKOS_LAMBDA(int const i) {
          CoefficientType radius = 10 * epsilon;

          for (int j = 0; j < _num_neighbors; j++)
          {
            CoefficientType norm =
                ArborX::Details::distance(source_ref_target(i, j), origin);
            radius = (radius < norm) ? norm : radius;
          }

          // The one at the limit would be valued at 0 due to how RBF works
          radii(i) = 1.1 * radius;
        });

    // Once the radius is computed, the wieght follows by evaluating the RBF at
    // each source point with their proper radii
    Kokkos::View<CoefficientType **, MemorySpace> phi(
        Kokkos::view_alloc(Kokkos::WithoutInitializing, "Example::MLSC::phi"),
        _num_targets, _num_neighbors);
    Kokkos::parallel_for(
        "Example::MLSC::phi_computation",
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>(space, {0, 0},
                                               {_num_targets, _num_neighbors}),
        KOKKOS_LAMBDA(int const i, int const j) {
          CoefficientType norm =
              ArborX::Details::distance(source_ref_target(i, j), origin);
          phi(i, j) = RadialBasisFunction::apply(norm / radii(i));
        });

    // We then need to create the Vandermonde matrix for each source point
    // Instead of relying on an external type, could it be produced
    // automatically?
    Kokkos::View<CoefficientType ***, MemorySpace> p(
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

    // From the weight and Vandermonde matrices, we can compute the moment
    // matrix as A = P^T.PHI.P
    Kokkos::View<CoefficientType ***, MemorySpace> a(
        Kokkos::view_alloc(Kokkos::WithoutInitializing,
                           "Example::MLSC::moment"),
        _num_targets, PolynomialBasis::size, PolynomialBasis::size);
    Kokkos::parallel_for(
        "Example::MLSC::moment_computation",
        Kokkos::MDRangePolicy<Kokkos::Rank<3>>(
            space, {0, 0, 0},
            {_num_targets, PolynomialBasis::size, PolynomialBasis::size}),
        KOKKOS_LAMBDA(int const i, int const j, int const k) {
          CoefficientType tmp = 0;

          for (int l = 0; l < _num_neighbors; l++)
          {
            tmp += p(i, l, j) * p(i, l, k) * phi(i, l);
          }

          a(i, j, k) = tmp;
        });

    // We then take the pseudo-inverse of that moment matrix.
    auto a_inv = symmetricPseudoInverseSVD(space, a);

    // We finally build the coefficients as C = [1 0 0 ...].A^-1.P^T.PHI
    _coeffs = Kokkos::View<CoefficientType **, MemorySpace>(
        Kokkos::view_alloc(Kokkos::WithoutInitializing,
                           "Example::MLSC::coefficients"),
        _num_targets, _num_neighbors);
    Kokkos::parallel_for(
        "Example::MLSC::coefficients",
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>(space, {0, 0},
                                               {_num_targets, _num_neighbors}),
        KOKKOS_LAMBDA(int const i, int const j) {
          CoefficientType tmp = 0;

          for (int k = 0; k < PolynomialBasis::size; k++)
          {
            tmp += a_inv(i, 0, k) * p(i, j, k) * phi(i, j);
          }

          _coeffs(i, j) = tmp;
        });
  }

  template <typename ExecutionSpace, typename SourceValues>
  Kokkos::View<typename SourceValues::non_const_value_type *,
               typename SourceValues::memory_space>
  apply(ExecutionSpace const &space, SourceValues const &source_values)
  {
    using value_t = typename SourceValues::non_const_value_type;
    using memory_space = typename SourceValues::memory_space;

    Kokkos::View<value_t *, memory_space> target_values(
        "Example::MLSC::target_values", _num_targets);
    Kokkos::parallel_for(
        "Example::MLSC::target_interpolation",
        Kokkos::RangePolicy<ExecutionSpace>(space, 0, _num_targets),
        KOKKOS_LAMBDA(int const i) {
          value_t tmp = 0;

          for (int j = 0; j < _num_neighbors; j++)
          {
            tmp += _coeffs(i, j) * source_values(i * _num_neighbors + j);
          }

          target_values(i) = tmp;
        });

    return target_values;
  }

private:
  Kokkos::View<CoefficientType **, MemorySpace> _coeffs;
  std::size_t _num_targets;
  std::size_t _num_neighbors;
};

} // namespace Details
