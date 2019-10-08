/****************************************************************************
 * Copyright (c) 2012-2019 by the ArborX authors                            *
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

#include <ArborX_Exception.hpp>

#include <Kokkos_Core.hpp> // FIXME

#include <algorithm> // max_element
#include <numeric>   // iota
#include <sstream>
#include <vector>

#include <mpi.h>

namespace ArborX
{
namespace Details
{

// Computes the array of indices that sort the input array (in reverse order)
// but also returns the sorted unique elements in that array with the
// corresponding element counts and displacement (offsets)
template <typename InputView, typename OutputView>
static void sortAndDetermineBufferLayout(InputView ranks,
                                         OutputView permutation_indices,
                                         std::vector<int> &unique_ranks,
                                         std::vector<int> &counts,
                                         std::vector<int> &offsets)
{
  ARBORX_ASSERT(unique_ranks.empty());
  ARBORX_ASSERT(offsets.empty());
  ARBORX_ASSERT(counts.empty());
  ARBORX_ASSERT(permutation_indices.extent(0) == ranks.extent(0));
  static_assert(
      std::is_same<typename InputView::non_const_value_type, int>::value, "");
  static_assert(std::is_same<typename OutputView::value_type, int>::value, "");
  static_assert(
      Kokkos::Impl::MemorySpaceAccess<typename OutputView::memory_space,
                                      Kokkos::HostSpace>::accessible,
      "");

  offsets.push_back(0);

  auto const n = ranks.extent_int(0);
  if (n == 0)
    return;

  Kokkos::View<int *, Kokkos::HostSpace> ranks_duplicate(
      Kokkos::ViewAllocateWithoutInitializing(ranks.label()), ranks.size());
  Kokkos::deep_copy(ranks_duplicate, ranks);

  while (true)
  {
    // TODO consider replacing with parallel reduce
    int const largest_rank =
        *std::max_element(ranks_duplicate.data(), ranks_duplicate.data() + n);
    if (largest_rank == -1)
      break;
    unique_ranks.push_back(largest_rank);
    counts.push_back(0);
    // TODO consider replacing with parallel scan
    for (int i = 0; i < n; ++i)
    {
      if (ranks_duplicate(i) == largest_rank)
      {
        ranks_duplicate(i) = -1;
        permutation_indices(i) = offsets.back() + counts.back();
        ++counts.back();
      }
    }
    offsets.push_back(offsets.back() + counts.back());
  }
}

class Distributor
{
public:
  Distributor(MPI_Comm comm)
      : _comm(comm)
  {
  }

  template <typename View>
  size_t createFromSends(View const &destination_ranks)
  {
    static_assert(View::rank == 1, "");
    static_assert(std::is_same<typename View::non_const_value_type, int>::value,
                  "");
    int comm_rank;
    MPI_Comm_rank(_comm, &comm_rank);
    int comm_size;
    MPI_Comm_size(_comm, &comm_size);

    _permute = Kokkos::View<int *, Kokkos::HostSpace>(
        Kokkos::ViewAllocateWithoutInitializing("permute"),
        destination_ranks.size());
    sortAndDetermineBufferLayout(destination_ranks, _permute, _destinations,
                                 _dest_counts, _dest_offsets);

    {
      std::vector<int> mask(comm_size);
      for (auto i : _destinations)
        mask[i] = 1;
      MPI_Alltoall(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, mask.data(), 1, MPI_INT,
                   _comm);
      for (int i = 0; i < comm_size; ++i)
        if (mask[i] == 1)
          _sources.push_back(i);
    }

    int const indegrees = _sources.size();
    int const outdegrees = _destinations.size();
    _src_counts.resize(indegrees);
    std::vector<MPI_Request> requests;
    requests.reserve(outdegrees + indegrees);
    for (int i = 0; i < indegrees; ++i)
    {
      requests.emplace_back();
      MPI_Irecv(&_src_counts[i], 1, MPI_INT, _sources[i], MPI_ANY_TAG, _comm,
                &requests.back());
    }
    for (int i = 0; i < outdegrees; ++i)
    {
      requests.emplace_back();
      MPI_Isend(&_dest_counts[i], 1, MPI_INT, _destinations[i], 255, _comm,
                &requests.back());
    }
    if (!requests.empty())
      MPI_Waitall(requests.size(), requests.data(), MPI_STATUSES_IGNORE);

    _src_offsets.resize(indegrees + 1);
    // exclusive scan
    _src_offsets[0] = 0;
    for (int i = 0; i < indegrees; ++i)
      _src_offsets[i + 1] = _src_offsets[i] + _src_counts[i];

    return _src_offsets.back();
  }
  template <typename View>
  void doPostsAndWaits(typename View::const_type const &exports,
                       size_t num_packets, View const &imports) const
  {
    ARBORX_ASSERT(num_packets * _src_offsets.back() == imports.size());
    ARBORX_ASSERT(num_packets * _dest_offsets.back() == exports.size());

    using ValueType = typename View::value_type;
    static_assert(View::rank == 1, "");
    static_assert(
        Kokkos::Impl::MemorySpaceAccess<typename View::memory_space,
                                        Kokkos::HostSpace>::accessible,
        "");

    std::vector<ValueType> dest_buffer(exports.size());
    std::vector<ValueType> src_buffer(imports.size());

    // TODO
    // * apply permutation on the device in a parallel for
    // * switch to MPI with CUDA support (do not copy to host)
    for (int i = 0; i < _dest_offsets.back(); ++i)
      std::copy(&exports[num_packets * i],
                &exports[num_packets * i] + num_packets,
                &dest_buffer[num_packets * _permute[i]]);

    int comm_rank;
    MPI_Comm_rank(_comm, &comm_rank);
    int comm_size;
    MPI_Comm_size(_comm, &comm_size);
    int const indegrees = _sources.size();
    int const outdegrees = _destinations.size();
    std::vector<MPI_Request> requests;
    requests.reserve(outdegrees + indegrees);
    int const chunk_size = std::numeric_limits<int>::max();
    for (int i = 0; i < indegrees; ++i)
    {
      std::size_t const total_message_size =
          _src_counts[i] * num_packets * sizeof(ValueType);
      int const n_chunks = (total_message_size + chunk_size - 1) / chunk_size;
      if (_sources[i] != comm_rank)
        for (int chunk = 0; chunk < n_chunks; ++chunk)
        {
          requests.emplace_back();
          int const this_chunk_size = std::min<std::size_t>(
              chunk_size, total_message_size - chunk * chunk_size);
          MPI_Irecv(reinterpret_cast<char *>(src_buffer.data() +
                                             _src_offsets[i] * num_packets) +
                        chunk * chunk_size,
                    this_chunk_size, MPI_BYTE, _sources[i], 123 + chunk, _comm,
                    &requests.back());
        }
    }
    for (int i = 0; i < outdegrees; ++i)
    {
      std::size_t const total_message_size =
          _dest_counts[i] * num_packets * sizeof(ValueType);
      int const n_chunks = (total_message_size + chunk_size - 1) / chunk_size;
      if (_destinations[i] == comm_rank)
      {
        auto const it = std::find(_sources.begin(), _sources.end(), comm_rank);
        ARBORX_ASSERT(it != _sources.end());
        auto const position = it - _sources.begin();
        std::memcpy(src_buffer.data() + _src_offsets[position] * num_packets,
                    dest_buffer.data() + _dest_offsets[i] * num_packets,
                    total_message_size);
      }
      else
      {
        for (int chunk = 0; chunk < n_chunks; ++chunk)
        {
          requests.emplace_back();
          int const this_chunk_size = std::min<std::size_t>(
              chunk_size, total_message_size - chunk * chunk_size);
          MPI_Isend(reinterpret_cast<char *>(dest_buffer.data() +
                                             _dest_offsets[i] * num_packets) +
                        chunk * chunk_size,
                    this_chunk_size, MPI_BYTE, _destinations[i], 123 + chunk,
                    _comm, &requests.back());
        }
      }
    }
    if (!requests.empty())
      MPI_Waitall(requests.size(), requests.data(), MPI_STATUSES_IGNORE);

    std::copy(src_buffer.begin(), src_buffer.end(), imports.data());
  }
  size_t getTotalReceiveLength() const { return _src_offsets.back(); }
  size_t getTotalSendLength() const { return _dest_offsets.back(); }

private:
  MPI_Comm _comm;
  Kokkos::View<int *, Kokkos::HostSpace> _permute;
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
