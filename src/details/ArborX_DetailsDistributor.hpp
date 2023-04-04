/****************************************************************************
 * Copyright (c) 2017-2022 by the ArborX authors                            *
 * All rights reserved.                                                     *
 *                                                                          *
 * This file is part of the ArborX library. ArborX is                       *
 * distributed under a BSD 3-clause license. For the licensing terms see    *
 * the LICENSE file in the top-level directory.                             *
 *                                                                          *
 * SPDX-License-Identifier: BSD-3-Clause                                    *
 ****************************************************************************/
#ifndef ARBORX_DETAILS_DISTRIBUTOR_HPP
#define ARBORX_DETAILS_DISTRIBUTOR_HPP

#include <ArborX_Config.hpp>

#include <ArborX_DetailsKokkosExtViewHelpers.hpp>
#include <ArborX_DetailsSortUtils.hpp>
#include <ArborX_DetailsUtils.hpp> // max
#include <ArborX_Exception.hpp>

#include <Kokkos_Core.hpp>

#include <algorithm> // max_element
#include <numeric>   // iota
#include <sstream>
#include <vector>

#include <mpi.h>

namespace ArborX
{
namespace Details
{

// Assuming that batched_ranks might contain elements multiply, but duplicates
// are not separated by other elements, return the unique elements in that array
// with the corresponding element counts and displacement (offsets).
template <typename ExecutionSpace, typename InputView, typename OutputView>
static void
determineBufferLayout(ExecutionSpace const &space, InputView batched_ranks,
                      InputView batched_offsets, OutputView permutation_indices,
                      std::vector<int> &unique_ranks, std::vector<int> &counts,
                      std::vector<int> &offsets)
{
  ARBORX_ASSERT(unique_ranks.empty());
  ARBORX_ASSERT(offsets.empty());
  ARBORX_ASSERT(counts.empty());
  ARBORX_ASSERT(permutation_indices.extent_int(0) == 0);
  ARBORX_ASSERT(batched_ranks.size() + 1 == batched_offsets.size());
  static_assert(
      std::is_same<typename InputView::non_const_value_type, int>::value);
  static_assert(std::is_same<typename OutputView::value_type, int>::value);

  // In case all the batches are empty, return an empty list of unique_ranks and
  // counts, but still have one element in offsets. This is conforming with
  // creating the total offsets from batched_ranks and batched_offsets ignoring
  // empty batches and calling sortAndDetermeineBufferLayout.
  offsets.push_back(0);

  auto const n_batched_ranks = batched_ranks.size();
  if (n_batched_ranks == 0 ||
      KokkosExt::lastElement(space, batched_offsets) == 0)
    return;

  using DeviceType = typename InputView::traits::device_type;

  // Find the indices in batched_ranks for which the rank changes and store
  // these ranks and the corresponding offsets in a new container that we can be
  // sure to be large enough.
  Kokkos::View<int *, DeviceType> compact_offsets(
      Kokkos::view_alloc(space, Kokkos::WithoutInitializing,
                         batched_offsets.label()),
      batched_offsets.size());
  Kokkos::View<int *, DeviceType> compact_ranks(
      Kokkos::view_alloc(space, Kokkos::WithoutInitializing,
                         batched_ranks.label()),
      batched_ranks.size());

  // Note that we never touch the first element of compact_offsets below.
  // Consequently, it is uninitialized.
  int n_unique_ranks;
  Kokkos::parallel_scan(
      "ArborX::Distributor::compact_offsets_and_ranks",
      Kokkos::RangePolicy<ExecutionSpace>(space, 0, n_batched_ranks),
      KOKKOS_LAMBDA(unsigned int i, int &update, bool last_pass) {
        if (i == batched_ranks.size() - 1 ||
            batched_ranks(i + 1) != batched_ranks(i))
        {
          if (last_pass)
          {
            compact_ranks(update) = batched_ranks(i);
            compact_offsets(update + 1) = batched_offsets(i + 1);
          }
          ++update;
        }
      },
      n_unique_ranks);

  // Now create subviews containing the elements we actually need, copy
  // everything to the CPU and fill the output variables.
  auto restricted_offsets = InputView(
      compact_offsets, std::make_pair(1, static_cast<int>(n_unique_ranks + 1)));
  auto restricted_unique_ranks = InputView(
      compact_ranks, std::make_pair(0, static_cast<int>(n_unique_ranks)));

  auto const unique_ranks_host = Kokkos::create_mirror_view(
      Kokkos::view_alloc(Kokkos::WithoutInitializing, Kokkos::HostSpace()),
      restricted_unique_ranks);
  Kokkos::deep_copy(space, unique_ranks_host, restricted_unique_ranks);
  auto const offsets_host = Kokkos::create_mirror_view(
      Kokkos::view_alloc(Kokkos::WithoutInitializing, Kokkos::HostSpace()),
      restricted_offsets);
  Kokkos::deep_copy(space, offsets_host, restricted_offsets);
  space.fence();
  unique_ranks.reserve(n_unique_ranks);
  offsets.reserve(n_unique_ranks + 1);
  counts.reserve(n_unique_ranks);

  for (int i = 0; i < n_unique_ranks; ++i)
  {
    int const count =
        i == 0 ? offsets_host(0) : offsets_host(i) - offsets_host(i - 1);
    if (count > 0)
    {
      counts.push_back(count);
      offsets.push_back(offsets_host(i));
      unique_ranks.push_back(unique_ranks_host(i));
    }
  }
}

// Computes the array of indices that sort the input array (in reverse order)
// but also returns the sorted unique elements in that array with the
// corresponding element counts and displacement (offsets)
template <typename ExecutionSpace, typename InputView, typename OutputView>
static void sortAndDetermineBufferLayout(ExecutionSpace const &space,
                                         InputView ranks,
                                         OutputView permutation_indices,
                                         std::vector<int> &unique_ranks,
                                         std::vector<int> &counts,
                                         std::vector<int> &offsets)
{
  ARBORX_ASSERT(unique_ranks.empty());
  ARBORX_ASSERT(offsets.empty());
  ARBORX_ASSERT(counts.empty());
  ARBORX_ASSERT(permutation_indices.extent_int(0) == ranks.extent_int(0));
  static_assert(
      std::is_same<typename InputView::non_const_value_type, int>::value);
  static_assert(std::is_same<typename OutputView::value_type, int>::value);

  offsets.push_back(0);

  auto const n = ranks.extent_int(0);
  if (n == 0)
    return;

  // this implements a "sort" which is O(N * R) where (R) is the total number of
  // unique destination ranks. it performs better than other algorithms in the
  // case when (R) is small, but results may vary
  using DeviceType = typename InputView::traits::device_type;

  Kokkos::View<int *, DeviceType> device_ranks_duplicate(
      Kokkos::view_alloc(space, Kokkos::WithoutInitializing, ranks.label()),
      ranks.size());
  Kokkos::deep_copy(space, device_ranks_duplicate, ranks);
  auto device_permutation_indices = Kokkos::create_mirror_view(
      Kokkos::view_alloc(space, Kokkos::WithoutInitializing,
                         typename DeviceType::memory_space{}),
      permutation_indices);
  int offset = 0;
  while (true)
  {
    int const largest_rank = ArborX::max(space, device_ranks_duplicate);
    if (largest_rank == -1)
      break;
    unique_ranks.push_back(largest_rank);
    int result = 0;
    Kokkos::parallel_scan(
        "ArborX::Distributor::process_biggest_rank_items",
        Kokkos::RangePolicy<ExecutionSpace>(space, 0, n),
        KOKKOS_LAMBDA(int i, int &update, bool last_pass) {
          bool const is_largest_rank =
              (device_ranks_duplicate(i) == largest_rank);
          if (is_largest_rank)
          {
            if (last_pass)
            {
              device_permutation_indices(i) = update + offset;
              device_ranks_duplicate(i) = -1;
            }
            ++update;
          }
        },
        result);
    offset += result;
    offsets.push_back(offset);
  }
  counts.reserve(offsets.size() - 1);
  for (unsigned int i = 1; i < offsets.size(); ++i)
    counts.push_back(offsets[i] - offsets[i - 1]);
  Kokkos::deep_copy(space, permutation_indices, device_permutation_indices);
  ARBORX_ASSERT(offsets.back() == static_cast<int>(ranks.size()));
}

template <typename DeviceType>
class Distributor
{
public:
  Distributor(MPI_Comm comm)
      : _comm(comm)
      , _permute{Kokkos::view_alloc(Kokkos::WithoutInitializing,
                                    "ArborX::Distributor::permute"),
                 0}
  {}

  template <typename ExecutionSpace, typename View>
  size_t createFromSends(ExecutionSpace const &space,
                         View const &batched_destination_ranks,
                         View const &batch_offsets)
  {
    static_assert(View::rank == 1);
    static_assert(
        std::is_same<typename View::non_const_value_type, int>::value);

    // The next two function calls are the only difference to the other
    // overload.
    // Note that we don't resize _permute here since we are assuming that no
    // reordering is necessary.
    determineBufferLayout(space, batched_destination_ranks, batch_offsets,
                          _permute, _destinations, _dest_counts, _dest_offsets);

    return preparePointToPointCommunication();
  }

  template <typename ExecutionSpace, typename View>
  size_t createFromSends(ExecutionSpace const &space,
                         View const &destination_ranks)
  {
    static_assert(View::rank == 1);
    static_assert(
        std::is_same<typename View::non_const_value_type, int>::value);

    // The next two function calls are the only difference to the other
    // overload.
    KokkosExt::reallocWithoutInitializing(space, _permute,
                                          destination_ranks.size());
    sortAndDetermineBufferLayout(space, destination_ranks, _permute,
                                 _destinations, _dest_counts, _dest_offsets);

    return preparePointToPointCommunication();
  }

  template <typename ExecutionSpace, typename ExportView, typename ImportView>
  void doPostsAndWaits(ExecutionSpace const &space, ExportView const &exports,
                       size_t num_packets, ImportView const &imports) const
  {
    ARBORX_ASSERT(num_packets * _src_offsets.back() == imports.size());
    ARBORX_ASSERT(num_packets * _dest_offsets.back() == exports.size());

    using ValueType = typename ImportView::value_type;
    static_assert(
        std::is_same<ValueType,
                     std::remove_cv_t<typename ExportView::value_type>>::value);
    static_assert(ImportView::rank == 1);

    static_assert(
        std::is_same<typename ExportView::memory_space,
                     typename decltype(_permute)::memory_space>::value);

    // This allows function to work even when ExportView is unmanaged.
    using ExportViewWithoutMemoryTraits =
        Kokkos::View<typename ExportView::data_type,
                     typename ExportView::array_layout,
                     typename ExportView::device_type>;

    using DestBufferMirrorViewType =
        decltype(ArborX::Details::create_layout_right_mirror_view_and_copy(
            space, std::declval<typename ImportView::memory_space>(),
            std::declval<ExportViewWithoutMemoryTraits>()));

    constexpr int pointer_depth = internal::PointerDepth<
        typename DestBufferMirrorViewType::traits::data_type>::value;
    DestBufferMirrorViewType dest_buffer_mirror(
        "ArborX::Distributor::doPostsAndWaits::destination_buffer_mirror", 0,
        pointer_depth > 1 ? 0 : KOKKOS_INVALID_INDEX,
        pointer_depth > 2 ? 0 : KOKKOS_INVALID_INDEX,
        pointer_depth > 3 ? 0 : KOKKOS_INVALID_INDEX,
        pointer_depth > 4 ? 0 : KOKKOS_INVALID_INDEX,
        pointer_depth > 5 ? 0 : KOKKOS_INVALID_INDEX,
        pointer_depth > 6 ? 0 : KOKKOS_INVALID_INDEX,
        pointer_depth > 7 ? 0 : KOKKOS_INVALID_INDEX);

    // If _permute is empty, we are assuming that we don't need to permute
    // exports.
    bool const permutation_necessary = _permute.size() != 0;
    if (permutation_necessary)
    {
      ExportViewWithoutMemoryTraits dest_buffer(
          Kokkos::view_alloc(
              space, Kokkos::WithoutInitializing,
              "ArborX::Distributor::doPostsAndWaits::destination_buffer"),
          exports.layout());

      ArborX::Details::applyInversePermutation(space, _permute, exports,
                                               dest_buffer);

      dest_buffer_mirror =
          ArborX::Details::create_layout_right_mirror_view_and_copy(
              space, typename ImportView::memory_space(), dest_buffer);
    }
    else
    {
      dest_buffer_mirror =
          ArborX::Details::create_layout_right_mirror_view_and_copy(
              space, typename ImportView::memory_space(), exports);
    }

    static_assert(
        decltype(dest_buffer_mirror)::rank == 1 ||
        std::is_same<typename decltype(dest_buffer_mirror)::array_layout,
                     Kokkos::LayoutRight>::value);
    static_assert(ImportView::rank == 1 ||
                  std::is_same<typename ImportView::array_layout,
                               Kokkos::LayoutRight>::value);

    int comm_rank;
    MPI_Comm_rank(_comm, &comm_rank);
    int comm_size;
    MPI_Comm_size(_comm, &comm_size);
    int const indegrees = _sources.size();
    int const outdegrees = _destinations.size();
    std::vector<MPI_Request> requests;
    requests.reserve(outdegrees + indegrees);
    for (int i = 0; i < indegrees; ++i)
    {
      if (_sources[i] != comm_rank)
      {
        auto const message_size =
            _src_counts[i] * num_packets * sizeof(ValueType);
        auto const receive_buffer_ptr =
            imports.data() + _src_offsets[i] * num_packets;
        requests.emplace_back();
        MPI_Irecv(receive_buffer_ptr, message_size, MPI_BYTE, _sources[i], 123,
                  _comm, &requests.back());
      }
    }

    // make sure the data in dest_buffer has been copied before sending it.
    if (permutation_necessary)
      space.fence("ArborX::Distributor::doPostsAndWaits"
                  " (permute done before packing data into send buffer)");

    for (int i = 0; i < outdegrees; ++i)
    {
      auto const message_size =
          _dest_counts[i] * num_packets * sizeof(ValueType);
      auto const send_buffer_ptr =
          dest_buffer_mirror.data() + _dest_offsets[i] * num_packets;
      if (_destinations[i] == comm_rank)
      {
        auto const it = std::find(_sources.begin(), _sources.end(), comm_rank);
        ARBORX_ASSERT(it != _sources.end());
        auto const position = it - _sources.begin();
        auto const receive_buffer_ptr =
            imports.data() + _src_offsets[position] * num_packets;

        Kokkos::View<ValueType *, typename ImportView::traits::device_type,
                     Kokkos::MemoryTraits<Kokkos::Unmanaged>>
            receive_view(receive_buffer_ptr, message_size / sizeof(ValueType));
        Kokkos::View<ValueType const *,
                     typename ExportView::traits::device_type,
                     Kokkos::MemoryTraits<Kokkos::Unmanaged>>
            send_view(send_buffer_ptr, message_size / sizeof(ValueType));
        Kokkos::deep_copy(space, receive_view, send_view);
      }
      else
      {
        requests.emplace_back();
        MPI_Isend(send_buffer_ptr, message_size, MPI_BYTE, _destinations[i],
                  123, _comm, &requests.back());
      }
    }
    if (!requests.empty())
      MPI_Waitall(requests.size(), requests.data(), MPI_STATUSES_IGNORE);
  }
  size_t getTotalReceiveLength() const { return _src_offsets.back(); }
  size_t getTotalSendLength() const { return _dest_offsets.back(); }

private:
  size_t preparePointToPointCommunication()
  {
    int comm_size;
    MPI_Comm_size(_comm, &comm_size);

    std::vector<int> src_counts_dense(comm_size);
    int const dest_size = _destinations.size();
    for (int i = 0; i < dest_size; ++i)
    {
      src_counts_dense[_destinations[i]] = _dest_counts[i];
    }
    MPI_Alltoall(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, src_counts_dense.data(), 1,
                 MPI_INT, _comm);

    _src_offsets.push_back(0);
    for (int i = 0; i < comm_size; ++i)
      if (src_counts_dense[i] > 0)
      {
        _sources.push_back(i);
        _src_counts.push_back(src_counts_dense[i]);
        _src_offsets.push_back(_src_offsets.back() + _src_counts.back());
      }

    return _src_offsets.back();
  }

  MPI_Comm _comm;
  Kokkos::View<int *, DeviceType> _permute;
  std::vector<int> _dest_offsets;
  std::vector<int> _dest_counts;
  std::vector<int> _src_offsets;
  std::vector<int> _src_counts;
  std::vector<int> _sources;
  std::vector<int> _destinations;
};

} // namespace Details
} // namespace ArborX

#endif
