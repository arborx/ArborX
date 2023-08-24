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

#include <ArborX_DetailsKokkosExtMinMaxReduce.hpp>
#include <ArborX_DetailsKokkosExtViewHelpers.hpp>
#include <ArborX_DetailsSortUtils.hpp>
#include <ArborX_Exception.hpp>

#include <Kokkos_Core.hpp>
#include <Kokkos_Profiling_ScopedRegion.hpp>

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
                      std::vector<int> &unique_ranks, std::vector<int> &offsets)
{
  Kokkos::Profiling::ScopedRegion guard(
      "ArborX::Distributor::determineBufferLayout");

  ARBORX_ASSERT(unique_ranks.empty());
  ARBORX_ASSERT(offsets.empty());
  ARBORX_ASSERT(permutation_indices.extent_int(0) == 0);
  ARBORX_ASSERT(batched_ranks.size() + 1 == batched_offsets.size());
  static_assert(std::is_same_v<typename InputView::non_const_value_type, int>);
  static_assert(std::is_same_v<typename OutputView::value_type, int>);

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
      Kokkos::RangePolicy(space, 0, n_batched_ranks),
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

  for (int i = 0; i < n_unique_ranks; ++i)
  {
    int const count =
        i == 0 ? offsets_host(0) : offsets_host(i) - offsets_host(i - 1);
    if (count > 0)
    {
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
                                         std::vector<int> &offsets)
{
  Kokkos::Profiling::ScopedRegion guard(
      "ArborX::Distributor::sortAndDetermineBufferLayout");

  ARBORX_ASSERT(unique_ranks.empty());
  ARBORX_ASSERT(offsets.empty());
  ARBORX_ASSERT(permutation_indices.extent_int(0) == ranks.extent_int(0));
  static_assert(std::is_same_v<typename InputView::non_const_value_type, int>);
  static_assert(std::is_same_v<typename OutputView::value_type, int>);

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
    int const largest_rank =
        KokkosExt::max_reduce(space, device_ranks_duplicate);
    if (largest_rank == -1)
      break;
    unique_ranks.push_back(largest_rank);
    int result = 0;
    Kokkos::parallel_scan(
        "ArborX::Distributor::process_biggest_rank_items",
        Kokkos::RangePolicy(space, 0, n),
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
    Kokkos::Profiling::ScopedRegion guard(
        "ArborX::Distributor::createFromSends(batched)");

    static_assert(View::rank == 1);
    static_assert(
        std::is_same<typename View::non_const_value_type, int>::value);

    // The next two function calls are the only difference to the other
    // overload.
    // Note that we don't resize _permute here since we are assuming that no
    // reordering is necessary.
    determineBufferLayout(space, batched_destination_ranks, batch_offsets,
                          _permute, _destinations, _dest_offsets);

    return preparePointToPointCommunication();
  }

  template <typename ExecutionSpace, typename View>
  size_t createFromSends(ExecutionSpace const &space,
                         View const &destination_ranks)
  {
    Kokkos::Profiling::ScopedRegion guard(
        "ArborX::Distributor::createFromSends");

    static_assert(View::rank == 1);
    static_assert(
        std::is_same<typename View::non_const_value_type, int>::value);

    auto const n = destination_ranks.extent_int(0);

    if (n == 0)
    {
      _destinations = {};
      _dest_offsets = {0};
    }
    else
    {
      auto [smallest_rank, largest_rank] =
          KokkosExt::minmax_reduce(space, destination_ranks);

      if (smallest_rank == largest_rank)
      {
        // The data is only sent to one rank, no need to permute
        _destinations = {smallest_rank};
        _dest_offsets = {0, n};
      }
      else
      {
        KokkosExt::reallocWithoutInitializing(space, _permute,
                                              destination_ranks.size());

        sortAndDetermineBufferLayout(space, destination_ranks, _permute,
                                     _destinations, _dest_offsets);
      }
    }

    return preparePointToPointCommunication();
  }

  template <typename ExecutionSpace, typename ExportView, typename ImportView>
  void doPostsAndWaits(ExecutionSpace const &space, ExportView const &exports,
                       ImportView const &imports) const
  {
    Kokkos::Profiling::ScopedRegion guard(
        "ArborX::Distributor::doPostsAndWaits");

    static_assert(ExportView::rank == 1 &&
                  (std::is_same_v<typename ExportView::array_layout,
                                  Kokkos::LayoutLeft> ||
                   std::is_same_v<typename ExportView::array_layout,
                                  Kokkos::LayoutRight>));
    static_assert(ImportView::rank == 1 &&
                  (std::is_same_v<typename ImportView::array_layout,
                                  Kokkos::LayoutLeft> ||
                   std::is_same_v<typename ImportView::array_layout,
                                  Kokkos::LayoutRight>));

    using MemorySpace = typename ExportView::memory_space;
    static_assert(
        std::is_same_v<MemorySpace, typename ImportView::memory_space>);
    static_assert(
        std::is_same_v<MemorySpace, typename decltype(_permute)::memory_space>);

    using ValueType = typename ImportView::value_type;
    static_assert(
        std::is_same<ValueType,
                     std::remove_cv_t<typename ExportView::value_type>>::value);

    bool const permutation_necessary = _permute.size() != 0;

    ARBORX_ASSERT(!permutation_necessary || exports.size() == _permute.size());
    ARBORX_ASSERT(exports.size() == getTotalSendLength());
    ARBORX_ASSERT(imports.size() == getTotalReceiveLength());

    // Make sure things work even if ExportView is unmanaged
    using ExportViewWithoutMemoryTraits =
        Kokkos::View<typename ExportView::data_type,
                     typename ExportView::array_layout,
                     typename ExportView::device_type>;
    ExportViewWithoutMemoryTraits permuted_exports_storage(
        "ArborX::Distributor::doPostsAndWaits::permuted_exports", 0);

    auto permuted_exports = exports;
    if (permutation_necessary)
    {
      KokkosExt::reallocWithoutInitializing(space, permuted_exports_storage,
                                            exports.size());

      permuted_exports = permuted_exports_storage;
      ArborX::Details::applyInversePermutation(space, _permute, exports,
                                               permuted_exports);
    }

    int comm_rank;
    MPI_Comm_rank(_comm, &comm_rank);

    int same_rank_destination = -1;
    int same_rank_source = -1;
    {
      auto it =
          std::find(_destinations.begin(), _destinations.end(), comm_rank);
      if (it != _destinations.end())
      {
        same_rank_destination = it - _destinations.begin();

        it = std::find(_sources.begin(), _sources.end(), comm_rank);
        ARBORX_ASSERT(it != _sources.end());
        same_rank_source = it - _sources.begin();
      }
    }

#ifndef ARBORX_ENABLE_GPU_AWARE_MPI
    using MirrorSpace = typename ExportView::host_mirror_space;

    auto exports_comm = Kokkos::create_mirror_view(
        Kokkos::WithoutInitializing, MirrorSpace{}, permuted_exports);
    if (same_rank_destination != -1)
    {
      // Only copy the parts of the exports that we need to send remotely
      for (auto interval :
           {std::make_pair(0, _dest_offsets[same_rank_destination]),
            std::make_pair(_dest_offsets[same_rank_destination + 1],
                           _dest_offsets.back())})
        Kokkos::deep_copy(space, Kokkos::subview(exports_comm, interval),
                          Kokkos::subview(permuted_exports, interval));
    }
    else
    {
      Kokkos::deep_copy(space, exports_comm, permuted_exports);
    }
    auto imports_comm = Kokkos::create_mirror_view(Kokkos::WithoutInitializing,
                                                   MirrorSpace{}, imports);
#else
    auto exports_comm = permuted_exports;
    auto imports_comm = imports;
#endif

    int const indegrees = _sources.size();
    int const outdegrees = _destinations.size();
    std::vector<MPI_Request> requests;
    requests.reserve(outdegrees + indegrees);
    for (int i = 0; i < indegrees; ++i)
    {
      if (_sources[i] != comm_rank)
      {
        auto const receive_buffer_ptr = imports_comm.data() + _src_offsets[i];
        auto const message_size =
            (_src_offsets[i + 1] - _src_offsets[i]) * sizeof(ValueType);
        requests.emplace_back();
        MPI_Irecv(receive_buffer_ptr, message_size, MPI_BYTE, _sources[i], 123,
                  _comm, &requests.back());
      }
    }

    // Make sure the data is ready before sending it
    space.fence(
        "ArborX::Distributor::doPostsAndWaits (data ready before sending)");

    for (int i = 0; i < outdegrees; ++i)
    {
      if (_destinations[i] != comm_rank)
      {
        requests.emplace_back();
        MPI_Isend(exports_comm.data() + _dest_offsets[i],
                  (_dest_offsets[i + 1] - _dest_offsets[i]) * sizeof(ValueType),
                  MPI_BYTE, _destinations[i], 123, _comm, &requests.back());
      }
    }
    if (!requests.empty())
      MPI_Waitall(requests.size(), requests.data(), MPI_STATUSES_IGNORE);

    if (same_rank_destination != -1)
    {
      ARBORX_ASSERT((_src_offsets[same_rank_source + 1] -
                     _src_offsets[same_rank_source]) ==
                    (_dest_offsets[same_rank_destination + 1] -
                     _dest_offsets[same_rank_destination]));
      Kokkos::deep_copy(
          space,
          Kokkos::subview(imports,
                          std::pair(_src_offsets[same_rank_source],
                                    _src_offsets[same_rank_source + 1])),
          Kokkos::subview(permuted_exports,
                          std::pair(_dest_offsets[same_rank_destination],
                                    _dest_offsets[same_rank_destination + 1])));
    }

#ifndef ARBORX_ENABLE_GPU_AWARE_MPI
    if (same_rank_destination != -1)
    {
      for (auto interval : {std::make_pair(0, _src_offsets[same_rank_source]),
                            std::make_pair(_src_offsets[same_rank_source + 1],
                                           _src_offsets.back())})
        Kokkos::deep_copy(space, Kokkos::subview(imports, interval),
                          Kokkos::subview(imports_comm, interval));
    }
    else
    {
      Kokkos::deep_copy(space, imports, imports_comm);
    }
#endif
  }
  size_t getTotalReceiveLength() const { return _src_offsets.back(); }
  size_t getTotalSendLength() const { return _dest_offsets.back(); }

private:
  size_t preparePointToPointCommunication()
  {
    Kokkos::Profiling::ScopedRegion guard(
        "ArborX::Distributor::preparePointToPointCommunication");

    int comm_size;
    MPI_Comm_size(_comm, &comm_size);

    std::vector<int> src_counts_dense(comm_size);
    int const dest_size = _destinations.size();
    for (int i = 0; i < dest_size; ++i)
    {
      src_counts_dense[_destinations[i]] =
          _dest_offsets[i + 1] - _dest_offsets[i];
    }
    MPI_Alltoall(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, src_counts_dense.data(), 1,
                 MPI_INT, _comm);

    _src_offsets.push_back(0);
    for (int i = 0; i < comm_size; ++i)
      if (src_counts_dense[i] > 0)
      {
        _sources.push_back(i);
        _src_offsets.push_back(_src_offsets.back() + src_counts_dense[i]);
      }

    return _src_offsets.back();
  }

  MPI_Comm _comm;
  Kokkos::View<int *, DeviceType> _permute;
  std::vector<int> _dest_offsets;
  std::vector<int> _src_offsets;
  std::vector<int> _sources;
  std::vector<int> _destinations;
};

} // namespace Details
} // namespace ArborX

#endif
