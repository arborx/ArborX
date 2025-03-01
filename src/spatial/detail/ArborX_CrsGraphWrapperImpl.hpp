/****************************************************************************
 * Copyright (c) 2025, ArborX authors                                       *
 * All rights reserved.                                                     *
 *                                                                          *
 * This file is part of the ArborX library. ArborX is                       *
 * distributed under a BSD 3-clause license. For the licensing terms see    *
 * the LICENSE file in the top-level directory.                             *
 *                                                                          *
 * SPDX-License-Identifier: BSD-3-Clause                                    *
 ****************************************************************************/

#ifndef ARBORX_DETAIL_CRS_GRAPH_WRAPPER_IMPL_HPP
#define ARBORX_DETAIL_CRS_GRAPH_WRAPPER_IMPL_HPP

#include <ArborX_Box.hpp>
#include <detail/ArborX_Callbacks.hpp>
#include <detail/ArborX_PermutedData.hpp>
#include <detail/ArborX_Predicates.hpp>
#include <detail/ArborX_SpaceFillingCurves.hpp>
#include <detail/ArborX_TraversalPolicy.hpp>
#include <kokkos_ext/ArborX_KokkosExtStdAlgorithms.hpp>
#include <kokkos_ext/ArborX_KokkosExtViewHelpers.hpp>

namespace ArborX
{
namespace Details
{

enum BufferStatus
{
  PreallocationNone = 0,
  PreallocationHard = -1,
  PreallocationSoft = 1
};

inline BufferStatus toBufferStatus(int buffer_size)
{
  if (buffer_size == 0)
    return BufferStatus::PreallocationNone;
  if (buffer_size > 0)
    return BufferStatus::PreallocationSoft;
  return BufferStatus::PreallocationHard;
}

struct FirstPassTag
{};
struct FirstPassNoBufferOptimizationTag
{};
struct SecondPassTag
{};

template <typename PassTag, typename Callback, typename OutputView,
          typename CountView, typename PermutedOffset>
struct InsertGenerator
{
  Callback _callback;
  OutputView _out;
  CountView _counts;
  PermutedOffset _permuted_offset;

  using ValueType = typename OutputView::value_type;

  template <typename Predicate, typename Value>
  KOKKOS_FUNCTION auto operator()(Predicate const &predicate,
                                  Value const &value) const
  {
    auto const predicate_index = getData(predicate);
    auto const &raw_predicate = getPredicate(predicate);
    auto &count = _counts(predicate_index);

    if constexpr (std::is_same_v<PassTag, FirstPassTag>)
    {
      // With permutation, we access offset in random manner, and
      // _offset(permutated_predicate_index+1) may be in a completely different
      // place. Instead, use pointers to get the correct value for the buffer
      // size. For this reason, also take a reference for offset.
      auto const &offset = _permuted_offset(predicate_index);
      auto const buffer_size = *(&offset + 1) - offset;

      return _callback(raw_predicate, value, [&](ValueType const &v) {
        int count_old = Kokkos::atomic_fetch_add(&count, 1);
        if (count_old < buffer_size)
          _out(offset + count_old) = v;
      });
    }
    else if constexpr (std::is_same_v<PassTag,
                                      FirstPassNoBufferOptimizationTag>)
    {
      return _callback(raw_predicate, value,
                       [&](ValueType const &) { Kokkos::atomic_inc(&count); });
    }
    else
    {
      static_assert(std::is_same_v<PassTag, SecondPassTag>);
      // we store offsets in counts, and offset(permute(i)) = counts(i)
      auto &offset = count;

      // TODO: there is a tradeoff here between skipping computation offset +
      // count, and atomic increment of count. I think atomically incrementing
      // offset is problematic for OpenMP as you potentially constantly steal
      // cache lines.
      return _callback(raw_predicate, value, [&](ValueType const &v) {
        _out(Kokkos::atomic_fetch_add(&offset, 1)) = v;
      });
    }
  }
};

namespace CrsGraphWrapperImpl
{

template <typename ExecutionSpace, typename Tree, typename Predicates,
          typename Callback, typename OutputView, typename OffsetView,
          typename PermuteType>
void queryImpl(ExecutionSpace const &space, Tree const &tree,
               Predicates const &predicates, Callback const &callback,
               OutputView &out, OffsetView &offset, PermuteType permute,
               BufferStatus buffer_status)
{
  // pre-condition: offset and out are preallocated. If buffer_size > 0, offset
  // is pre-initialized

  static_assert(Kokkos::is_execution_space<ExecutionSpace>{});

  auto const n_queries = predicates.size();

  Kokkos::Profiling::pushRegion("ArborX::CrsGraphWrapper::two_pass");

  using CountView = OffsetView;
  CountView counts(Kokkos::view_alloc(space, "ArborX::CrsGraphWrapper::counts"),
                   n_queries);

  using PermutedPredicates =
      PermutedData<Predicates, PermuteType, true /*AttachIndices*/>;
  PermutedPredicates permuted_predicates = {predicates, permute};

  using PermutedOffset = PermutedData<OffsetView, PermuteType>;
  PermutedOffset permuted_offset = {offset, permute};

  Kokkos::Profiling::pushRegion(
      "ArborX::CrsGraphWrapper::two_pass::first_pass");
  bool underflow = false;
  bool overflow = false;
  if (buffer_status != BufferStatus::PreallocationNone)
  {
    tree.query(
        space, permuted_predicates,
        InsertGenerator<FirstPassTag, Callback, OutputView, CountView,
                        PermutedOffset>{callback, out, counts, permuted_offset},
        ArborX::Experimental::TraversalPolicy().setPredicateSorting(false));

    // Detecting overflow is a local operation that needs to be done for every
    // index. We allow individual buffer sizes to differ, so it's not as easy
    // as computing max counts.
    int overflow_int;
    Kokkos::parallel_reduce(
        "ArborX::CrsGraphWrapper::compute_overflow",
        Kokkos::RangePolicy(space, 0, n_queries),
        KOKKOS_LAMBDA(int i, int &update) {
          auto const *const offset_ptr = &permuted_offset(i);
          if (counts(i) > *(offset_ptr + 1) - *offset_ptr)
            update = 1;
        },
        overflow_int);
    overflow = (overflow_int > 0);

    if (!overflow)
    {
      int n_results;
      Kokkos::parallel_reduce(
          "ArborX::CrsGraphWrapper::compute_underflow",
          Kokkos::RangePolicy(space, 0, n_queries),
          KOKKOS_LAMBDA(int i, int &update) { update += counts(i); },
          n_results);
      underflow = (n_results < out.extent_int(0));
    }
  }
  else
  {
    tree.query(
        space, permuted_predicates,
        InsertGenerator<FirstPassNoBufferOptimizationTag, Callback, OutputView,
                        CountView, PermutedOffset>{callback, out, counts,
                                                   permuted_offset},
        ArborX::Experimental::TraversalPolicy().setPredicateSorting(false));
    // This may not be true, but it does not matter. As long as we have
    // (n_results == 0) check before second pass, this value is not used.
    // Otherwise, we know it's overflowed as there is no allocation.
    overflow = true;
  }

  Kokkos::Profiling::popRegion();
  Kokkos::Profiling::pushRegion(
      "ArborX::CrsGraphWrapper::first_pass_postprocess");

  OffsetView preallocated_offset("ArborX::CrsGraphWrapper::offset_copy", 0);
  if (underflow)
  {
    // Store a copy of the original offset. We'll need it for compression.
    preallocated_offset = KokkosExt::clone(space, offset);
  }

  Kokkos::parallel_for(
      "ArborX::CrsGraphWrapper::copy_counts_to_offsets",
      Kokkos::RangePolicy(space, 0, n_queries),
      KOKKOS_LAMBDA(int const i) { permuted_offset(i) = counts(i); });
  KokkosExt::exclusive_scan(space, offset, offset, 0);

  int const n_results = KokkosExt::lastElement(space, offset);

  Kokkos::Profiling::popRegion();

  if (n_results == 0)
  {
    // Exit early if either no results were found for any of the queries, or
    // nothing was inserted inside a callback for found results. This check
    // guarantees that the second pass will not be executed.
    Kokkos::resize(Kokkos::view_alloc(space, Kokkos::WithoutInitializing), out,
                   0);
    // FIXME: do we need to reset offset if it was preallocated here?
    Kokkos::Profiling::popRegion();
    return;
  }

  if (overflow || buffer_status == BufferStatus::PreallocationNone)
  {
    // Not enough (individual) storage for results

    // If it was hard preallocation, we simply throw
    ARBORX_ASSERT(buffer_status != BufferStatus::PreallocationHard);

    // Otherwise, do the second pass
    Kokkos::Profiling::pushRegion(
        "ArborX::CrsGraphWrapper::two_pass:second_pass");

    Kokkos::parallel_for(
        "ArborX::CrsGraphWrapper::copy_offsets_to_counts",
        Kokkos::RangePolicy(space, 0, n_queries),
        KOKKOS_LAMBDA(int const i) { counts(i) = permuted_offset(i); });

    KokkosExt::reallocWithoutInitializing(space, out, n_results);

    tree.query(
        space, permuted_predicates,
        InsertGenerator<SecondPassTag, Callback, OutputView, CountView,
                        PermutedOffset>{callback, out, counts, permuted_offset},
        ArborX::Experimental::TraversalPolicy().setPredicateSorting(false));

    Kokkos::Profiling::popRegion();
  }
  else if (underflow)
  {
    // More than enough storage for results, need compression
    Kokkos::Profiling::pushRegion(
        "ArborX::CrsGraphWrapper::two_pass:copy_values");

    OutputView tmp_out(
        Kokkos::view_alloc(space, Kokkos::WithoutInitializing, out.label()),
        n_results);

    Kokkos::parallel_for(
        "ArborX::CrsGraphWrapper::copy_valid_values",
        Kokkos::RangePolicy(space, 0, n_queries), KOKKOS_LAMBDA(int i) {
          int count = offset(i + 1) - offset(i);
          for (int j = 0; j < count; ++j)
          {
            tmp_out(offset(i) + j) = out(preallocated_offset(i) + j);
          }
        });
    out = tmp_out;

    Kokkos::Profiling::popRegion();
  }
  else
  {
    // The allocated storage was exactly enough for results, do nothing
  }
  Kokkos::Profiling::popRegion();
}

struct Iota
{
  KOKKOS_FUNCTION unsigned int operator()(int const i) const { return i; }
};

template <typename Tag, typename ExecutionSpace, typename Predicates,
          typename OffsetView, typename OutView>
std::enable_if_t<std::is_same_v<Tag, SpatialPredicateTag> ||
                 std::is_same_v<Tag, OrderedSpatialPredicateTag>>
allocateAndInitializeStorage(Tag, ExecutionSpace const &space,
                             Predicates const &predicates, OffsetView &offset,
                             OutView &out, int buffer_size)
{
  auto const n_queries = predicates.size();
  KokkosExt::reallocWithoutInitializing(space, offset, n_queries + 1);

  buffer_size = std::abs(buffer_size);

  Kokkos::deep_copy(space, offset, buffer_size);

  if (buffer_size != 0)
  {
    KokkosExt::exclusive_scan(space, offset, offset, 0);

    // Use calculation for the size to avoid calling lastElement(space, offset)
    // as it will launch an extra kernel to copy to host.
    KokkosExt::reallocWithoutInitializing(space, out, n_queries * buffer_size);
  }
}

template <typename Tag, typename ExecutionSpace, typename Predicates,
          typename OffsetView, typename OutView>
std::enable_if_t<std::is_same_v<Tag, NearestPredicateTag>>
allocateAndInitializeStorage(Tag, ExecutionSpace const &space,
                             Predicates const &predicates, OffsetView &offset,
                             OutView &out, int /*buffer_size*/)
{
  auto const n_queries = predicates.size();
  KokkosExt::reallocWithoutInitializing(space, offset, n_queries + 1);

  Kokkos::parallel_for(
      "ArborX::CrsGraphWrapper::query::nearest::"
      "scan_queries_for_numbers_of_nearest_neighbors",
      Kokkos::RangePolicy(space, 0, n_queries),
      KOKKOS_LAMBDA(int i) { offset(i) = getK(predicates(i)); });
  KokkosExt::exclusive_scan(space, offset, offset, 0);

  KokkosExt::reallocWithoutInitializing(space, out,
                                        KokkosExt::lastElement(space, offset));
}

// Views are passed by reference here because internally Kokkos::realloc()
// is called.
template <typename Tag, typename Tree, typename ExecutionSpace,
          typename Predicates, typename OutputView, typename OffsetView,
          typename Callback>
std::enable_if_t<!is_tagged_post_callback<Callback>::value &&
                 Kokkos::is_view_v<OutputView> && Kokkos::is_view_v<OffsetView>>
queryDispatch(Tag, Tree const &tree, ExecutionSpace const &space,
              Predicates const &predicates, Callback const &callback,
              OutputView &out, OffsetView &offset,
              Experimental::TraversalPolicy const &policy =
                  Experimental::TraversalPolicy())
{
  check_valid_callback<typename Tree::value_type>(callback, predicates, out);

  std::string profiling_prefix = "ArborX::CrsGraphWrapper::query::";
  if constexpr (std::is_same_v<Tag, SpatialPredicateTag>)
  {
    profiling_prefix += "spatial";
  }
  else if constexpr (std::is_same_v<Tag, OrderedSpatialPredicateTag>)
  {
    profiling_prefix += "ordered_spatial";
  }
  else if constexpr (std::is_same_v<Tag, NearestPredicateTag>)
  {
    profiling_prefix += "nearest";
  }
  else
  {
    static_assert(std::is_void_v<Tag>, "ArborX implementation bug");
  }

  Kokkos::Profiling::pushRegion(profiling_prefix);

  Kokkos::Profiling::pushRegion(profiling_prefix + "::init_and_alloc");

  allocateAndInitializeStorage(Tag{}, space, predicates, offset, out,
                               policy._buffer_size);

  Kokkos::Profiling::popRegion();

  auto buffer_status = (std::is_same<Tag, SpatialPredicateTag>{}
                            ? toBufferStatus(policy._buffer_size)
                            : BufferStatus::PreallocationSoft);

  if (policy._sort_predicates)
  {
    Kokkos::Profiling::pushRegion(profiling_prefix + "::compute_permutation");
    using bounding_volume_type = std::decay_t<decltype(tree.bounds())>;
    Box<GeometryTraits::dimension_v<bounding_volume_type>,
        typename GeometryTraits::coordinate_type_t<bounding_volume_type>>
        scene_bounding_box{};
    using namespace Details;
    expand(scene_bounding_box, tree.bounds());
    auto permute = computeSpaceFillingCurvePermutation(
        space, PredicateIndexables<Predicates>{predicates},
        Experimental::Morton32{}, scene_bounding_box);
    Kokkos::Profiling::popRegion();

    queryImpl(space, tree, predicates, callback, out, offset, permute,
              buffer_status);
  }
  else
  {
    Iota permute;
    queryImpl(space, tree, predicates, callback, out, offset, permute,
              buffer_status);
  }

  Kokkos::Profiling::popRegion();
}

template <typename Tag, typename Tree, typename ExecutionSpace,
          typename Predicates, typename Indices, typename Offset>
inline std::enable_if_t<Kokkos::is_view_v<Indices> && Kokkos::is_view_v<Offset>>
queryDispatch(Tag, Tree const &tree, ExecutionSpace const &space,
              Predicates const &predicates, Indices &indices, Offset &offset,
              Experimental::TraversalPolicy const &policy =
                  Experimental::TraversalPolicy())
{
  queryDispatch(Tag{}, tree, space, predicates, DefaultCallback{}, indices,
                offset, policy);
}

template <typename Tag, typename Tree, typename ExecutionSpace,
          typename Predicates, typename OutputView, typename OffsetView,
          typename Callback>
inline std::enable_if_t<is_tagged_post_callback<Callback>::value>
queryDispatch(Tag, Tree const &tree, ExecutionSpace const &space,
              Predicates const &predicates, Callback const &callback,
              OutputView &out, OffsetView &offset,
              Experimental::TraversalPolicy const &policy =
                  Experimental::TraversalPolicy())
{
  using MemorySpace = typename Tree::memory_space;

  Kokkos::View<typename Tree::value_type *, MemorySpace> indices(
      "ArborX::CrsGraphWrapper::query::indices", 0);
  queryDispatch(Tag{}, tree, space, predicates, DefaultCallback{}, indices,
                offset, policy);
  callback(predicates, offset, indices, out);
}

template <typename Value, typename Callback, typename Predicates,
          typename OutputView>
std::enable_if_t<!Kokkos::is_view_v<Callback> &&
                 !is_tagged_post_callback<Callback>::value>
check_valid_callback_if_first_argument_is_not_a_view(
    Callback const &callback, Predicates const &predicates,
    OutputView const &out)
{
  check_valid_callback<Value>(callback, predicates, out);
}

template <typename Value, typename Callback, typename Predicates,
          typename OutputView>
std::enable_if_t<!Kokkos::is_view_v<Callback> &&
                 is_tagged_post_callback<Callback>::value>
check_valid_callback_if_first_argument_is_not_a_view(Callback const &,
                                                     Predicates const &,
                                                     OutputView const &)
{
  // TODO
}

template <typename Value, typename View, typename Predicates,
          typename OutputView>
std::enable_if_t<Kokkos::is_view_v<View>>
check_valid_callback_if_first_argument_is_not_a_view(View const &,
                                                     Predicates const &,
                                                     OutputView const &)
{
  // do nothing
}

} // namespace CrsGraphWrapperImpl

} // namespace Details
} // namespace ArborX

#endif
