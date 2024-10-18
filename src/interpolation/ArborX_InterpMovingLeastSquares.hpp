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

#ifndef ARBORX_INTERP_MOVING_LEAST_SQUARES_HPP
#define ARBORX_INTERP_MOVING_LEAST_SQUARES_HPP

#include <ArborX_Box.hpp>
#include <ArborX_GeometryTraits.hpp>
#include <ArborX_LinearBVH.hpp>
#include <detail/ArborX_AccessTraits.hpp>
#include <detail/ArborX_Indexable.hpp>
#include <detail/ArborX_InterpDetailsCompactRadialBasisFunction.hpp>
#include <detail/ArborX_InterpDetailsMovingLeastSquaresCoefficients.hpp>
#include <detail/ArborX_InterpDetailsPolynomialBasis.hpp>
#include <detail/ArborX_PredicateHelpers.hpp>

#include <Kokkos_Core.hpp>
#include <Kokkos_Profiling_ScopedRegion.hpp>

#include <optional>

namespace ArborX::Interpolation::Details
{

// Functor used in the tree query to create the 2D source view and indices
template <typename SourceView, typename IndicesView, typename CounterView>
struct MLSSearchNeighborsCallback
{
  SourceView source_view;
  IndicesView indices;
  CounterView counter;

  using SourcePoint = typename SourceView::non_const_value_type;

  template <typename Predicate>
  KOKKOS_FUNCTION void
  operator()(Predicate const &predicate,
             ArborX::PairValueIndex<SourcePoint> const &primitive) const
  {
    int const target = getData(predicate);
    int const source = primitive.index;
    auto count = Kokkos::atomic_fetch_add(&counter(target), 1);
    indices(target, count) = source;
    source_view(target, count) = primitive.value;
  }
};

} // namespace ArborX::Interpolation::Details

namespace ArborX::Interpolation
{

template <typename MemorySpace, typename FloatingCalculationType = double>
class MovingLeastSquares
{
public:
  template <typename ExecutionSpace, typename SourcePoints,
            typename TargetPoints, typename CRBFunc = CRBF::Wendland<0>,
            typename PolynomialDegree = PolynomialDegree<2>>
  MovingLeastSquares(ExecutionSpace const &space,
                     SourcePoints const &source_points,
                     TargetPoints const &target_points, CRBFunc = {},
                     PolynomialDegree = {},
                     std::optional<int> num_neighbors = std::nullopt)
  {
    namespace KokkosExt = ArborX::Details::KokkosExt;

    auto guard = Kokkos::Profiling::ScopedRegion("ArborX::MovingLeastSquares");

    static_assert(
        KokkosExt::is_accessible_from<MemorySpace, ExecutionSpace>::value,
        "Memory space must be accessible from the execution space");

    // SourcePoints is an access trait of points
    ArborX::Details::check_valid_access_traits(PrimitivesTag{}, source_points);
    using SourceAccess =
        ArborX::Details::AccessValues<SourcePoints, PrimitivesTag>;
    static_assert(
        KokkosExt::is_accessible_from<typename SourceAccess::memory_space,
                                      ExecutionSpace>::value,
        "Source points must be accessible from the execution space");
    using SourcePoint = typename SourceAccess::value_type;
    GeometryTraits::check_valid_geometry_traits(SourcePoint{});
    static_assert(GeometryTraits::is_point_v<SourcePoint>,
                  "Source points elements must be points");
    static constexpr int dimension = GeometryTraits::dimension_v<SourcePoint>;

    // TargetPoints is an access trait of points
    ArborX::Details::check_valid_access_traits(PrimitivesTag{}, target_points);
    using TargetAccess =
        ArborX::Details::AccessValues<TargetPoints, PrimitivesTag>;
    static_assert(
        KokkosExt::is_accessible_from<typename TargetAccess::memory_space,
                                      ExecutionSpace>::value,
        "Target points must be accessible from the execution space");
    using TargetPoint = typename TargetAccess::value_type;
    GeometryTraits::check_valid_geometry_traits(TargetPoint{});
    static_assert(GeometryTraits::is_point_v<TargetPoint>,
                  "Target points elements must be points");
    static_assert(dimension == GeometryTraits::dimension_v<TargetPoint>,
                  "Target and source points must have the same dimension");

    _num_neighbors =
        num_neighbors ? *num_neighbors
                      : Details::polynomialBasisSize<dimension,
                                                     PolynomialDegree::value>();

    TargetAccess target_access{target_points}; // NOLINT
    SourceAccess source_access{source_points}; // NOLINT

    _num_targets = target_access.size();
    _source_size = source_access.size();
    // There must be enough source points
    KOKKOS_ASSERT(0 < _num_neighbors && _num_neighbors <= _source_size);

    // Search for neighbors and get the arranged source points
    auto source_view = searchNeighbors(space, source_access, target_access);

    // Compute the moving least squares coefficients
    _coeffs = Details::movingLeastSquaresCoefficients<CRBFunc, PolynomialDegree,
                                                      FloatingCalculationType>(
        space, source_view, target_access);
  }

  template <typename ExecutionSpace, typename SourceValues,
            typename ApproxValues>
  void interpolate(ExecutionSpace const &space,
                   SourceValues const &source_values,
                   ApproxValues &approx_values) const
  {
    auto guard = Kokkos::Profiling::ScopedRegion(
        "ArborX::MovingLeastSquares::interpolate");

    namespace KokkosExt = ArborX::Details::KokkosExt;

    static_assert(
        KokkosExt::is_accessible_from<MemorySpace, ExecutionSpace>::value,
        "Memory space must be accessible from the execution space");

    // SourceValues is a 1D view of all source values
    static_assert(Kokkos::is_view_v<SourceValues> && SourceValues::rank() == 1,
                  "Source values must be a 1D view of values");
    static_assert(
        KokkosExt::is_accessible_from<typename SourceValues::memory_space,
                                      ExecutionSpace>::value,
        "Source values must be accessible from the execution space");

    // ApproxValues is a 1D view for approximated values
    static_assert(Kokkos::is_view_v<ApproxValues> && ApproxValues::rank() == 1,
                  "Approx values must be a 1D view");
    static_assert(
        KokkosExt::is_accessible_from<typename ApproxValues::memory_space,
                                      ExecutionSpace>::value,
        "Approx values must be accessible from the execution space");
    static_assert(!std::is_const_v<typename ApproxValues::value_type>,
                  "Approx values must be writable");

    // Source values must be a valuation on the points so must be as big as the
    // original input
    KOKKOS_ASSERT(_source_size == source_values.extent_int(0));

    // Approx values must have the correct size
    KOKKOS_ASSERT(approx_values.extent_int(0) == _num_targets);

    using Value = typename ApproxValues::non_const_value_type;

    Kokkos::parallel_for(
        "ArborX::MovingLeastSquares::target_interpolation",
        Kokkos::RangePolicy(space, 0, _num_targets),
        KOKKOS_CLASS_LAMBDA(int const i) {
          Value tmp = 0;
          for (int j = 0; j < _num_neighbors; j++)
            tmp += _coeffs(i, j) * source_values(_indices(i, j));
          approx_values(i) = tmp;
        });
  }

private:
  template <typename ExecutionSpace, typename SourceAccess,
            typename TargetAccess>
  auto searchNeighbors(ExecutionSpace const &space,
                       SourceAccess const &source_access,
                       TargetAccess const &target_access)
  {
    auto guard = Kokkos::Profiling::ScopedRegion(
        "ArborX::MovingLeastSquares::searchNeighbors");

    // Organize the source points as a tree
    using SourcePoint = typename SourceAccess::value_type;
    BoundingVolumeHierarchy source_tree(
        space, ArborX::Experimental::attach_indices(source_access));

    // Create the predicates
    auto predicates = Experimental::attach_indices(
        Experimental::make_nearest(target_access, _num_neighbors));

    // Create the callback
    Kokkos::View<SourcePoint **, MemorySpace> source_view(
        Kokkos::view_alloc(space, Kokkos::WithoutInitializing,
                           "ArborX::MovingLeastSquares::source_view"),
        _num_targets, _num_neighbors);
    _indices = Kokkos::View<int **, MemorySpace>(
        Kokkos::view_alloc(space, Kokkos::WithoutInitializing,
                           "ArborX::MovingLeastSquares::indices"),
        _num_targets, _num_neighbors);
    Kokkos::View<int *, MemorySpace> counter(
        "ArborX::MovingLeastSquares::counter", _num_targets);
    Details::MLSSearchNeighborsCallback<decltype(source_view),
                                        decltype(_indices), decltype(counter)>
        callback{source_view, _indices, counter};

    // Query the source tree
    source_tree.query(space, predicates, callback);

    return source_view;
  }

  Kokkos::View<FloatingCalculationType **, MemorySpace> _coeffs;
  Kokkos::View<int **, MemorySpace> _indices;
  int _num_targets;
  int _num_neighbors;
  int _source_size;
};

} // namespace ArborX::Interpolation

#endif
