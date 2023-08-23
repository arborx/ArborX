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

#include "DetailsPolynomialBasis.hpp"
#include "DetailsSymmetricPseudoInverseSVD.hpp"

namespace Details
{

template <typename Point>
using PointEquivalence = ArborX::ExperimentalHyperGeometry::Point<
    ArborX::GeometryTraits::dimension_v<Point>,
    typename ArborX::GeometryTraits::coordinate_type<Point>::type>;

template <typename Points>
using PointEquivalenceFromAT =
    PointEquivalence<typename ArborX::Details::AccessTraitsHelper<
        ArborX::AccessTraits<Points, ArborX::PrimitivesTag>>::type>;

template <typename MemorySpace, typename CoefficientType>
class MovingLeastSquaresComputation
{
public:
  MovingLeastSquaresComputation() = default;

  template <typename ExecutionSpace, typename PolynomialDegree,
            typename RadialBasisFunction, typename SourcePoints,
            typename TargetPoints>
  MovingLeastSquaresComputation(ExecutionSpace const &space,
                                SourcePoints const &source_points,
                                TargetPoints const &target_points,
                                PolynomialDegree const &pd,
                                RadialBasisFunction const &rbf)
  {
    using src_acc = ArborX::AccessTraits<SourcePoints, ArborX::PrimitivesTag>;
    using tgt_acc = ArborX::AccessTraits<TargetPoints, ArborX::PrimitivesTag>;
    using point_t = PointEquivalenceFromAT<SourcePoints>;

    _num_targets = tgt_acc::size(target_points);
    _num_neighbors = src_acc::size(source_points) / _num_targets;

    static constexpr std::size_t polynomialBasisSize =
        polynomialBasisSizeFromAT<SourcePoints, PolynomialDegree::value>;

    // We center each group of points around the target as it ables us to
    // optimize the final computation and transfer point types into ours
    // TODO: Use multidimensional points!
    Kokkos::View<point_t **, MemorySpace> source_ref_target =
        sourceRefTargetFill(space, source_points, target_points, _num_targets,
                            _num_neighbors);

    // To properly use the RBF, we need to decide for a radius around each
    // target point that encapsulates all of the points
    Kokkos::View<CoefficientType *, MemorySpace> radii = radiiComputation(
        space, source_ref_target, _num_targets, _num_neighbors);

    // Once the radius is computed, the wieght follows by evaluating the RBF at
    // each source point with their proper radii
    Kokkos::View<CoefficientType **, MemorySpace> phi = weightComputation(
        space, source_ref_target, radii, _num_targets, _num_neighbors, rbf);

    // We then need to create the Vandermonde matrix for each source point
    // Instead of relying on an external type, could it be produced
    // automatically?
    Kokkos::View<CoefficientType ***, MemorySpace> p = vandermondeComputation(
        space, source_ref_target, _num_targets, _num_neighbors, pd);

    // From the weight and Vandermonde matrices, we can compute the moment
    // matrix as A = P^T.PHI.P
    Kokkos::View<CoefficientType ***, MemorySpace> a = momentComputation(
        space, phi, p, _num_targets, _num_neighbors, polynomialBasisSize);

    // We then take the pseudo-inverse of that moment matrix.
    Kokkos::View<CoefficientType ***, MemorySpace> a_inv =
        symmetricPseudoInverseSVD(space, a);

    // We finally build the coefficients as C = [1 0 0 ...].A^-1.P^T.PHI
    _coeffs = coefficientsComputation(space, phi, p, a_inv, _num_targets,
                                      _num_neighbors, polynomialBasisSize);
  }

  template <typename ExecutionSpace, typename SourceValues>
  Kokkos::View<typename SourceValues::non_const_value_type *,
               typename SourceValues::memory_space>
  apply(ExecutionSpace const &space, SourceValues const &source_values)
  {
    using value_t = typename SourceValues::non_const_value_type;
    using memory_space = typename SourceValues::memory_space;

    std::size_t num_neighbors = _num_neighbors;
    Kokkos::View<CoefficientType **, MemorySpace> coeffs = _coeffs;

    Kokkos::View<value_t *, memory_space> target_values(
        Kokkos::view_alloc(Kokkos::WithoutInitializing,
                           "Example::MLSC::target_values"),
        _num_targets);

    Kokkos::parallel_for(
        "Example::MLSC::target_interpolation",
        Kokkos::RangePolicy<ExecutionSpace>(space, 0, _num_targets),
        KOKKOS_LAMBDA(int const i) {
          value_t tmp = 0;

          for (int j = 0; j < num_neighbors; j++)
          {
            tmp += coeffs(i, j) * source_values(i * num_neighbors + j);
          }

          target_values(i) = tmp;
        });

    return target_values;
  }

  template <typename ExecutionSpace, typename SourcePoints,
            typename TargetPoints>
  static Kokkos::View<PointEquivalenceFromAT<SourcePoints> **, MemorySpace>
  sourceRefTargetFill(ExecutionSpace const &space,
                      SourcePoints const &source_points,
                      TargetPoints const &target_points,
                      std::size_t num_targets, std::size_t num_neighbors)
  {
    using src_acc = ArborX::AccessTraits<SourcePoints, ArborX::PrimitivesTag>;
    using tgt_acc = ArborX::AccessTraits<TargetPoints, ArborX::PrimitivesTag>;
    using point_t = PointEquivalenceFromAT<SourcePoints>;

    Kokkos::View<point_t **, MemorySpace> source_ref_target(
        Kokkos::view_alloc(Kokkos::WithoutInitializing,
                           "Example::MLSC::source_ref_target"),
        num_targets, num_neighbors);

    Kokkos::parallel_for(
        "Example::MLSC::source_ref_target_fill",
        Kokkos::MDRangePolicy<ExecutionSpace, Kokkos::Rank<2>>(
            space, {0, 0}, {num_targets, num_neighbors}),
        KOKKOS_LAMBDA(int const i, int const j) {
          auto src = src_acc::get(source_points, i * num_neighbors + j);
          auto tgt = tgt_acc::get(target_points, i);
          point_t t{};

          for (int k = 0; k < ArborX::GeometryTraits::dimension_v<point_t>; k++)
            t[k] = src[k] - tgt[k];

          source_ref_target(i, j) = t;
        });

    return source_ref_target;
  }

  template <typename ExecutionSpace, typename Point>
  static Kokkos::View<CoefficientType *, MemorySpace>
  radiiComputation(ExecutionSpace const &space,
                   Kokkos::View<Point **, MemorySpace> const &source_ref_target,
                   std::size_t num_targets, std::size_t num_neighbors)
  {
    constexpr CoefficientType epsilon =
        std::numeric_limits<CoefficientType>::epsilon();
    constexpr Point origin{};

    Kokkos::View<CoefficientType *, MemorySpace> radii(
        Kokkos::view_alloc(Kokkos::WithoutInitializing, "Example::MLSC::radii"),
        num_targets);

    Kokkos::parallel_for(
        "Example::MLSC::radii_computation",
        Kokkos::RangePolicy<ExecutionSpace>(space, 0, num_targets),
        KOKKOS_LAMBDA(int const i) {
          CoefficientType radius = 10 * epsilon;

          for (int j = 0; j < num_neighbors; j++)
          {
            CoefficientType norm =
                ArborX::Details::distance(source_ref_target(i, j), origin);
            radius = (radius < norm) ? norm : radius;
          }

          // The one at the limit would be valued at 0 due to how RBF works
          radii(i) = 1.1 * radius;
        });

    return radii;
  }

  template <typename ExecutionSpace, typename RadialBasisFunction,
            typename Point>
  static Kokkos::View<CoefficientType **, MemorySpace> weightComputation(
      ExecutionSpace const &space,
      Kokkos::View<Point **, MemorySpace> const &source_ref_target,
      Kokkos::View<CoefficientType *, MemorySpace> const &radii,
      std::size_t num_targets, std::size_t num_neighbors,
      RadialBasisFunction const &)
  {
    constexpr Point origin{};

    Kokkos::View<CoefficientType **, MemorySpace> phi(
        Kokkos::view_alloc(Kokkos::WithoutInitializing, "Example::MLSC::phi"),
        num_targets, num_neighbors);

    Kokkos::parallel_for(
        "Example::MLSC::phi_computation",
        Kokkos::MDRangePolicy<ExecutionSpace, Kokkos::Rank<2>>(
            space, {0, 0}, {num_targets, num_neighbors}),
        KOKKOS_LAMBDA(int const i, int const j) {
          CoefficientType norm =
              ArborX::Details::distance(source_ref_target(i, j), origin);
          phi(i, j) = RadialBasisFunction::apply(norm / radii(i));
        });

    return phi;
  }

  template <typename ExecutionSpace, typename PolynomialDegree, typename Point>
  static Kokkos::View<CoefficientType ***, MemorySpace> vandermondeComputation(
      ExecutionSpace const &space,
      Kokkos::View<Point **, MemorySpace> const &source_ref_target,
      std::size_t num_targets, std::size_t num_neighbors,
      PolynomialDegree const &)
  {
    static constexpr std::size_t polynomialBasisSize =
        polynomialBasisSizeFromT<Point, PolynomialDegree::value>;

    Kokkos::View<CoefficientType ***, MemorySpace> p(
        Kokkos::view_alloc(Kokkos::WithoutInitializing,
                           "Example::MLSC::vandermonde"),
        num_targets, num_neighbors, polynomialBasisSize);

    Kokkos::parallel_for(
        "Example::MLSC::vandermonde_computation",
        Kokkos::MDRangePolicy<ExecutionSpace, Kokkos::Rank<2>>(
            space, {0, 0}, {num_targets, num_neighbors}),
        KOKKOS_LAMBDA(int const i, int const j) {
          auto basis = polynomialBasis<Point, PolynomialDegree::value>(
              source_ref_target(i, j));

          for (int k = 0; k < polynomialBasisSize; k++)
          {
            p(i, j, k) = basis[k];
          }
        });

    return p;
  }

  template <typename ExecutionSpace>
  static Kokkos::View<CoefficientType ***, MemorySpace>
  momentComputation(ExecutionSpace const &space,
                    Kokkos::View<CoefficientType **, MemorySpace> const &phi,
                    Kokkos::View<CoefficientType ***, MemorySpace> const &p,
                    std::size_t num_targets, std::size_t num_neighbors,
                    std::size_t polynomialBasisSize)
  {
    Kokkos::View<CoefficientType ***, MemorySpace> a(
        Kokkos::view_alloc(Kokkos::WithoutInitializing,
                           "Example::MLSC::moment"),
        num_targets, polynomialBasisSize, polynomialBasisSize);

    Kokkos::parallel_for(
        "Example::MLSC::moment_computation",
        Kokkos::MDRangePolicy<ExecutionSpace, Kokkos::Rank<3>>(
            space, {0, 0, 0},
            {num_targets, polynomialBasisSize, polynomialBasisSize}),
        KOKKOS_LAMBDA(int const i, int const j, int const k) {
          CoefficientType tmp = 0;

          for (int l = 0; l < num_neighbors; l++)
          {
            tmp += p(i, l, j) * p(i, l, k) * phi(i, l);
          }

          a(i, j, k) = tmp;
        });

    return a;
  }

  template <typename ExecutionSpace>
  static Kokkos::View<CoefficientType **, MemorySpace> coefficientsComputation(
      ExecutionSpace const &space,
      Kokkos::View<CoefficientType **, MemorySpace> const &phi,
      Kokkos::View<CoefficientType ***, MemorySpace> const &p,
      Kokkos::View<CoefficientType ***, MemorySpace> const &a_inv,
      std::size_t num_targets, std::size_t num_neighbors,
      std::size_t polynomialBasisSize)
  {
    Kokkos::View<CoefficientType **, MemorySpace> coeffs(
        Kokkos::view_alloc(Kokkos::WithoutInitializing,
                           "Example::MLSC::coefficients"),
        num_targets, num_neighbors);

    Kokkos::parallel_for(
        "Example::MLSC::coefficients_computation",
        Kokkos::MDRangePolicy<ExecutionSpace, Kokkos::Rank<2>>(
            space, {0, 0}, {num_targets, num_neighbors}),
        KOKKOS_LAMBDA(int const i, int const j) {
          CoefficientType tmp = 0;

          for (int k = 0; k < polynomialBasisSize; k++)
          {
            tmp += a_inv(i, 0, k) * p(i, j, k) * phi(i, j);
          }

          coeffs(i, j) = tmp;
        });

    return coeffs;
  }

private:
  Kokkos::View<CoefficientType **, MemorySpace> _coeffs;
  std::size_t _num_targets;
  std::size_t _num_neighbors;
};

} // namespace Details
