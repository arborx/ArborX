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
#include <ArborX_InterpDetailsCompactRadialBasisFunction.hpp>
#include <ArborX_InterpDetailsPolynomialBasis.hpp>
#include <ArborX_DetailsSymmetricSVD.hpp>

#include <Kokkos_Core.hpp>
#include <Kokkos_Profiling_ScopedRegion.hpp>

namespace ArborX::Interpolation::Details
{

template <typename SourcePoints, typename TargetAccess, typename Coefficients,
          typename ExecutionSpace, typename CRBFunc = CRBF::Wendland<0>,
          typename PolynomialDegree = PolynomialDegree<2>>
class MovingLeastSquaresCoefficientsKernel
{
private:
  using ScratchMemorySpace = typename ExecutionSpace::scratch_memory_space;

  using SourcePoint = typename SourcePoints::non_const_value_type;
  using TargetPoint = typename TargetAccess::value_type;

  using CoefficientsType = typename Coefficients::non_const_value_type;

  static constexpr int dimension = GeometryTraits::dimension_v<SourcePoint>;
  static constexpr int degree = PolynomialDegree::value;
  static constexpr int poly_size = polynomialBasisSize<dimension, degree>();

  template <typename T>
  using ScratchView =
      Kokkos::View<T, ScratchMemorySpace, Kokkos::MemoryUnmanaged>;

  using LocalSourcePoints = Kokkos::Subview<SourcePoints, int, Kokkos::ALL_t>;
  using LocalPhi = ScratchView<CoefficientsType *>;
  using LocalVandermonde = ScratchView<CoefficientsType *[poly_size]>;
  using LocalMoment = ScratchView<CoefficientsType[poly_size][poly_size]>;
  using LocalSVDDiag = ScratchView<CoefficientsType[poly_size]>;
  using LocalSVDUnit = ScratchView<CoefficientsType[poly_size][poly_size]>;
  using LocalCoefficients = Kokkos::Subview<Coefficients, int, Kokkos::ALL_t>;

public:
  MovingLeastSquaresCoefficientsKernel(ExecutionSpace const &,
                                       TargetAccess const &target_access,
                                       SourcePoints const &source_points,
                                       Coefficients const &coefficients)
      : _target_access(target_access)
      , _source_points(source_points)
      , _coefficients(coefficients)
      , _num_targets(target_access.size())
      , _num_neighbors(source_points.extent_int(1))
  {}

  template <typename TeamMember>
  KOKKOS_FUNCTION void operator()(TeamMember const &member) const
  {
    auto const &scratch = member.thread_scratch(0);

    int target = member.league_rank() * member.team_size() + member.team_rank();
    if (target >= _num_targets)
      return;

    auto target_point = _target_access(target);
    auto source_points = Kokkos::subview(_source_points, target, Kokkos::ALL);
    LocalPhi phi(scratch, _num_neighbors);
    LocalVandermonde vandermonde(scratch, _num_neighbors);
    LocalMoment moment(scratch);
    LocalSVDDiag svd_diag(scratch);
    LocalSVDUnit svd_unit(scratch);
    auto coefficients = Kokkos::subview(_coefficients, target, Kokkos::ALL);

    // The goal is to compute the following line vector for each target point:
    // p(x).[P^T.PHI.P]^-1.P^T.PHI
    // Where:
    // - p(x) is the polynomial basis of point x (line vector).
    // - P is the multidimensional Vandermonde matrix built from the source
    //   points, i.e., each line is the polynomial basis of a source point.
    // - PHI is the diagonal weight matrix / CRBF evaluated at each source
    // point.

    // We first change the origin of the evaluation to be at the target point.
    // This lets us use p(0) which is [1 0 ... 0].
    sourceRecentering(target_point, source_points);

    // This computes PHI given the source points (radius is computed inside)
    phiComputation(source_points, phi);

    // This builds the Vandermonde (P) matrix
    vandermondeComputation(source_points, vandermonde);

    // We then create what is called the moment matrix, which is P^T.PHI.P. By
    // construction, it is symmetric.
    momentComputation(phi, vandermonde, moment);

    // We need the inverse of P^T.PHI.P, and because it is symmetric, we can use
    // the symmetric SVD algorithm to get it.
    symmetricPseudoInverseSVDKernel(moment, svd_diag, svd_unit);
    // Now, the moment has [P^T.PHI.P]^-1

    // Finally, the result is produced by computing p(0).[P^T.PHI.P]^-1.P^T.PHI
    coefficientsComputation(phi, vandermonde, moment, coefficients);
  }

  Kokkos::TeamPolicy<ExecutionSpace>
  makePolicy(ExecutionSpace const &space) const
  {
    Kokkos::TeamPolicy<ExecutionSpace> dummy_policy(space, 1, Kokkos::AUTO);
    dummy_policy.set_scratch_size(0, Kokkos::PerThread(perTargetMem()));
    int team_size =
        dummy_policy.team_size_recommended(*this, Kokkos::ParallelForTag{});
    if (team_size != 0)
    {
      int league_size = (_num_targets + team_size - 1) / team_size;
      return Kokkos::TeamPolicy<ExecutionSpace>(space, league_size, team_size)
          .set_scratch_size(0, Kokkos::PerThread(perTargetMem()));
    }
    return Kokkos::TeamPolicy<ExecutionSpace>(space, _num_targets, 1, 1)
        .set_scratch_size(0, Kokkos::PerTeam(perTargetMem()));
  }

private:
  std::size_t perTargetMem() const
  {
    std::size_t val = 0;
    val += LocalPhi::shmem_size(_num_neighbors);
    val += LocalVandermonde::shmem_size(_num_neighbors);
    val += LocalMoment::shmem_size();
    val += LocalSVDDiag::shmem_size();
    val += LocalSVDUnit::shmem_size();
    return val;
  }

  // Recenters the source points so that the target is at the origin
  KOKKOS_FUNCTION void sourceRecentering(TargetPoint const &target_point,
                                         LocalSourcePoints &source_points) const
  {
    for (int neighbor = 0; neighbor < _num_neighbors; neighbor++)
      for (int k = 0; k < dimension; k++)
        source_points(neighbor)[k] -= target_point[k];
  }

  // Computes the weight matrix
  KOKKOS_FUNCTION void phiComputation(LocalSourcePoints const &source_points,
                                      LocalPhi &phi) const
  {
    CoefficientsType radius = Kokkos::Experimental::epsilon_v<CoefficientsType>;
    for (int neighbor = 0; neighbor < _num_neighbors; neighbor++)
    {
      CoefficientsType const norm =
          ArborX::Details::distance(source_points(neighbor), SourcePoint{});
      radius = Kokkos::max(radius, norm);
    }

    // The one at the limit would be 0 due to how CRBFs work
    radius *= CoefficientsType(1.1);

    for (int neighbor = 0; neighbor < _num_neighbors; neighbor++)
      phi(neighbor) = CRBF::evaluate<CRBFunc>(source_points(neighbor), radius);
  }

  // Computes the vandermonde matrix
  KOKKOS_FUNCTION void
  vandermondeComputation(LocalSourcePoints const &source_points,
                         LocalVandermonde &vandermonde) const
  {
    for (int neighbor = 0; neighbor < _num_neighbors; neighbor++)
    {
      auto basis = evaluatePolynomialBasis<degree>(source_points(neighbor));
      for (int k = 0; k < poly_size; k++)
        vandermonde(neighbor, k) = basis[k];
    }
  }

  // Computes the moment matrix
  KOKKOS_FUNCTION void momentComputation(LocalPhi const &phi,
                                         LocalVandermonde const &vandermonde,
                                         LocalMoment &moment) const
  {
    for (int i = 0; i < poly_size; i++)
      for (int j = 0; j < poly_size; j++)
      {
        moment(i, j) = 0;
        for (int neighbor = 0; neighbor < _num_neighbors; neighbor++)
          moment(i, j) += vandermonde(neighbor, i) * vandermonde(neighbor, j) *
                          phi(neighbor);
      }
  }

  // Computes the coefficients
  KOKKOS_FUNCTION void coefficientsComputation(
      LocalPhi const &phi, LocalVandermonde const &vandermonde,
      LocalMoment const &moment, LocalCoefficients &coefficients) const
  {
    for (int neighbor = 0; neighbor < _num_neighbors; neighbor++)
    {
      coefficients(neighbor) = 0;
      for (int i = 0; i < poly_size; i++)
        coefficients(neighbor) +=
            moment(0, i) * vandermonde(neighbor, i) * phi(neighbor);
    }
  }

  TargetAccess _target_access;
  SourcePoints _source_points;
  Coefficients _coefficients;
  int _num_targets;
  int _num_neighbors;
};

template <typename CRBFunc, typename PolynomialDegree,
          typename CoefficientsType, typename ExecutionSpace,
          typename SourcePoints, typename TargetAccess>
auto movingLeastSquaresCoefficients(ExecutionSpace const &space,
                                    SourcePoints const &source_points,
                                    TargetAccess const &target_access)
{
  auto guard =
      Kokkos::Profiling::ScopedRegion("ArborX::MovingLeastSquaresCoefficients");

  namespace KokkosExt = ::ArborX::Details::KokkosExt;

  using MemorySpace = typename SourcePoints::memory_space;
  static_assert(
      KokkosExt::is_accessible_from<MemorySpace, ExecutionSpace>::value,
      "Memory space must be accessible from the execution space");

  // SourcePoints is a 2D view of points
  static_assert(Kokkos::is_view_v<SourcePoints> && SourcePoints::rank == 2,
                "source points must be a 2D view of points");
  static_assert(
      KokkosExt::is_accessible_from<typename SourcePoints::memory_space,
                                    ExecutionSpace>::value,
      "source points must be accessible from the execution space");
  using SourcePoint = typename SourcePoints::non_const_value_type;
  GeometryTraits::check_valid_geometry_traits(SourcePoint{});
  static_assert(GeometryTraits::is_point<SourcePoint>::value,
                "source points elements must be points");
  static_assert(!std::is_const_v<typename SourcePoints::value_type>,
                "source points must be writable");
  constexpr int dimension = GeometryTraits::dimension_v<SourcePoint>;

  // TargetAccess is an access values of points
  static_assert(
      KokkosExt::is_accessible_from<typename TargetAccess::memory_space,
                                    ExecutionSpace>::value,
      "target access must be accessible from the execution space");
  using TargetPoint = typename TargetAccess::value_type;
  GeometryTraits::check_valid_geometry_traits(TargetPoint{});
  static_assert(GeometryTraits::is_point<TargetPoint>::value,
                "target access elements must be points");
  static_assert(dimension == GeometryTraits::dimension_v<TargetPoint>,
                "target and source points must have the same dimension");

  // The number of source groups must be correct
  KOKKOS_ASSERT(std::size_t{target_access.size()} == source_points.extent(0));

  Kokkos::View<CoefficientsType **, MemorySpace> coefficients(
      Kokkos::view_alloc(
          space, Kokkos::WithoutInitializing,
          "ArborX::MovingLeastSquaresCoefficients::coefficients"),
      source_points.extent_int(0), source_points.extent_int(1));

  MovingLeastSquaresCoefficientsKernel<SourcePoints, TargetAccess,
                                       decltype(coefficients), ExecutionSpace,
                                       CRBFunc, PolynomialDegree>
      kernel(space, target_access, source_points, coefficients);

  Kokkos::parallel_for("ArborX::MovingLeastSquaresCoefficients::kernel",
                       kernel.makePolicy(space), kernel);

  return coefficients;
}

} // namespace ArborX::Interpolation::Details

#endif
