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

#ifndef ARBORX_INTERP_DETAILS_MOVING_LEAST_SQUARES_COEFFICIENTS_HPP
#define ARBORX_INTERP_DETAILS_MOVING_LEAST_SQUARES_COEFFICIENTS_HPP

#include <ArborX_AccessTraits.hpp>
#include <ArborX_DetailsKokkosExtAccessibilityTraits.hpp>
#include <ArborX_GeometryTraits.hpp>
#include <ArborX_HyperPoint.hpp>
#include <ArborX_InterpDetailsPolynomialBasis.hpp>
#include <ArborX_InterpDetailsSymmetricPseudoInverseSVD.hpp>

#include <Kokkos_Core.hpp>

namespace ArborX::Interpolation::Details
{

template <typename CRBF, typename PolynomialDegree, typename ExecutionSpace,
          typename SourcePoints, typename TargetPoints, typename Coefficients>
void movingLeastSquaresCoefficients(ExecutionSpace const &space,
                                    SourcePoints const &source_points,
                                    TargetPoints const &target_points,
                                    Coefficients &coeffs)
{
  // SourcePoints is a 2D view of points
  static_assert(Kokkos::is_view_v<SourcePoints> && SourcePoints::rank == 2,
                "source points must be a 2D view of points");
  static_assert(
      KokkosExt::is_accessible_from<typename SourcePoints::memory_space,
                                    ExecutionSpace>::value,
      "source points must be accessible from the execution space");
  using src_point = typename SourcePoints::non_const_value_type;
  GeometryTraits::check_valid_geometry_traits(src_point{});
  static_assert(GeometryTraits::is_point<src_point>::value,
                "source points elements must be points");
  static constexpr int dimension = GeometryTraits::dimension_v<src_point>;

  // TargetPoints is an access trait of points
  ArborX::Details::check_valid_access_traits(PrimitivesTag{}, target_points);
  using tgt_acc = AccessTraits<TargetPoints, PrimitivesTag>;
  static_assert(KokkosExt::is_accessible_from<typename tgt_acc::memory_space,
                                              ExecutionSpace>::value,
                "target points must be accessible from the execution space");
  using tgt_point = typename ArborX::Details::AccessTraitsHelper<tgt_acc>::type;
  GeometryTraits::check_valid_geometry_traits(tgt_point{});
  static_assert(GeometryTraits::is_point<tgt_point>::value,
                "target points elements must be points");
  static_assert(dimension == GeometryTraits::dimension_v<tgt_point>,
                "target and source points must have the same dimension");

  // Coefficients is a 2D view of values
  static_assert(Kokkos::is_view_v<Coefficients> && Coefficients::rank == 2,
                "coeffs must be a 2D view");
  static_assert(
      KokkosExt::is_accessible_from<typename Coefficients::memory_space,
                                    ExecutionSpace>::value,
      "coeffs must be accessible from the execution space");
  static_assert(!std::is_const_v<typename Coefficients::value_type>,
                "coeffs must be writable");

  int const num_targets = tgt_acc::size(target_points);
  int const num_neighbors = source_points.extent(1);

  // There must be a set of neighbors for each target
  ARBORX_ASSERT(num_targets == source_points.extent_int(0));

  using value_t = typename Coefficients::non_const_value_type;
  using point_t = ExperimentalHyperGeometry::Point<dimension, value_t>;
  using memory_space = typename Coefficients::memory_space;
  static constexpr auto epsilon = Kokkos::Experimental::epsilon_v<value_t>;
  static constexpr int degree = PolynomialDegree::value;
  static constexpr int poly_size = polynomialBasisSize<dimension, degree>();

  // The goal is to compute the following line vector for each target point:
  // p(0).[P^T.PHI.P]^-1.P^T.PHI
  // Where:
  // - p(x) is the polynomial basis of point x (line vector).
  // - P is the multidimensional Vandermonde matrix built from the source
  //   points, that is each line is the polynomial basis of a source point.
  // - PHI is the diagonal weight matrix / CRBF evaluated at each source point.

  // We first change the origin of the evaluation to be at the target point.
  // This lets us use p(0) which is [1 0 ... 0].
  Kokkos::View<point_t **, memory_space> source_ref_target(
      Kokkos::view_alloc(
          space, Kokkos::WithoutInitializing,
          "ArborX::MovingLeastSquaresCoefficients::source_ref_target"),
      num_targets, num_neighbors);
  Kokkos::parallel_for(
      "ArborX::MovingLeastSquaresCoefficients::source_ref_target_fill",
      Kokkos::MDRangePolicy<ExecutionSpace, Kokkos::Rank<2>>(
          space, {0, 0}, {num_targets, num_neighbors}),
      KOKKOS_LAMBDA(int const i, int const j) {
        auto src = source_points(i, j);
        auto tgt = tgt_acc::get(target_points, i);
        point_t t{};

        for (int k = 0; k < dimension; k++)
          t[k] = src[k] - tgt[k];

        source_ref_target(i, j) = t;
      });

  // We then compute the radius for each target that will be used in evaluating
  // the weight for each source point.
  Kokkos::View<value_t *, memory_space> radii(
      Kokkos::view_alloc(space, Kokkos::WithoutInitializing,
                         "ArborX::MovingLeastSquaresCoefficients::radii"),
      num_targets);
  Kokkos::parallel_for(
      "ArborX::MovingLeastSquaresCoefficients::radii_computation",
      Kokkos::RangePolicy<ExecutionSpace>(space, 0, num_targets),
      KOKKOS_LAMBDA(int const i) {
        value_t radius = epsilon;

        for (int j = 0; j < num_neighbors; j++)
        {
          value_t norm =
              ArborX::Details::distance(source_ref_target(i, j), point_t{});
          radius = Kokkos::max(radius, norm);
        }

        // The one at the limit would be valued at 0 due to how CRBFs work
        radii(i) = 1.1 * radius;
      });

  // This computes PHI given the source points as well as the radius
  Kokkos::View<value_t **, memory_space> phi(
      Kokkos::view_alloc(space, Kokkos::WithoutInitializing,
                         "ArborX::MovingLeastSquaresCoefficients::phi"),
      num_targets, num_neighbors);
  Kokkos::parallel_for(
      "ArborX::MovingLeastSquaresCoefficients::phi_computation",
      Kokkos::MDRangePolicy<ExecutionSpace, Kokkos::Rank<2>>(
          space, {0, 0}, {num_targets, num_neighbors}),
      KOKKOS_LAMBDA(int const i, int const j) {
        value_t norm =
            ArborX::Details::distance(source_ref_target(i, j), point_t{});
        phi(i, j) = CRBF::evaluate(norm / radii(i));
      });

  // This builds the Vandermonde (P) matrix
  Kokkos::View<value_t ***, memory_space> p(
      Kokkos::view_alloc(space, Kokkos::WithoutInitializing,
                         "ArborX::MovingLeastSquaresCoefficients::vandermonde"),
      num_targets, num_neighbors, poly_size);
  Kokkos::parallel_for(
      "ArborX::MovingLeastSquaresCoefficients::vandermonde_computation",
      Kokkos::MDRangePolicy<ExecutionSpace, Kokkos::Rank<2>>(
          space, {0, 0}, {num_targets, num_neighbors}),
      KOKKOS_LAMBDA(int const i, int const j) {
        auto basis = evaluatePolynomialBasis<degree>(source_ref_target(i, j));
        for (int k = 0; k < poly_size; k++)
          p(i, j, k) = basis[k];
      });

  // We then create what is called the moment matrix, which is A = P^T.PHI.P. By
  // construction, A is symmetric.
  Kokkos::View<value_t ***, memory_space> a(
      Kokkos::view_alloc(space, Kokkos::WithoutInitializing,
                         "ArborX::MovingLeastSquaresCoefficients::moment"),
      num_targets, poly_size, poly_size);
  Kokkos::parallel_for(
      "ArborX::MovingLeastSquaresCoefficients::moment_computation",
      Kokkos::MDRangePolicy<ExecutionSpace, Kokkos::Rank<3>>(
          space, {0, 0, 0}, {num_targets, poly_size, poly_size}),
      KOKKOS_LAMBDA(int const i, int const j, int const k) {
        value_t tmp = 0;
        for (int l = 0; l < num_neighbors; l++)
          tmp += p(i, l, j) * p(i, l, k) * phi(i, l);
        a(i, j, k) = tmp;
      });

  // We need the inverse of A = P^T.PHI.P, and because A is symmetric, we can
  // use the symmetric SVD algorithm to get it.
  symmetricPseudoInverseSVD(space, a);
  // Now, A = [P^T.PHI.P]^-1

  // Finally, the result is produced by computing p(0).A.P^T.PHI
  Kokkos::resize(space, coeffs, num_targets, num_neighbors);
  Kokkos::parallel_for(
      "ArborX::MovingLeastSquaresCoefficients::coefficients_computation",
      Kokkos::MDRangePolicy<ExecutionSpace, Kokkos::Rank<2>>(
          space, {0, 0}, {num_targets, num_neighbors}),
      KOKKOS_LAMBDA(int const i, int const j) {
        value_t tmp = 0;
        for (int k = 0; k < poly_size; k++)
          tmp += a(i, 0, k) * p(i, j, k) * phi(i, j);
        coeffs(i, j) = tmp;
      });
}

} // namespace ArborX::Interpolation::Details

#endif