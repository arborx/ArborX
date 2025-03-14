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

#ifndef ARBORX_INTERP_DETAILS_DISTRIBUTED_VALUES_DISTRIBUTOR_HPP
#define ARBORX_INTERP_DETAILS_DISTRIBUTED_VALUES_DISTRIBUTOR_HPP

#include <ArborX_DetailsDistributedTreeImpl.hpp>
#include <ArborX_DetailsDistributor.hpp>
#include <ArborX_PairIndexRank.hpp>

#include <Kokkos_Core.hpp>
#include <Kokkos_Profiling_ScopedRegion.hpp>

#include <array>
#include <memory>

#include <mpi.h>

namespace ArborX::Interpolation::Details
{

struct DVDKernelData
{
  int dst_index;
  int src_rank;
  int src_index;
};

template <typename DstRanks, typename SrcData, typename IndicesAndRanks>
struct DVDKernel1
{
  DstRanks dst_ranks;
  SrcData src_data;
  IndicesAndRanks indices_and_ranks;
  int rank;

  KOKKOS_FUNCTION void operator()(int const i) const
  {
    dst_ranks(i) = indices_and_ranks(i).rank;
    src_data(i).dst_index = indices_and_ranks(i).index;
    src_data(i).src_rank = rank;
    src_data(i).src_index = i;
  }
};

template <typename SendIndices, typename DstData, typename SrcRanks,
          typename SrcIndices>
struct DVDKernel2
{
  SendIndices send_indices;
  DstData dst_data;
  SrcRanks src_ranks;
  SrcIndices src_indices;

  KOKKOS_FUNCTION void operator()(int const i) const
  {
    send_indices(i) = dst_data(i).dst_index;
    src_ranks(i) = dst_data(i).src_rank;
    src_indices(i) = dst_data(i).src_index;
  }
};

template <typename MemorySpace>
class DistributedValuesDistributor
{
public:
  template <typename ExecutionSpace, typename IndicesAndRanks>
  DistributedValuesDistributor(MPI_Comm comm, ExecutionSpace const &space,
                               IndicesAndRanks const &indices_and_ranks)
      : _distributor(nullptr)
  {
    auto guard =
        Kokkos::Profiling::ScopedRegion("ArborX::DistributedValuesDistributor");

    namespace KokkosExt = ArborX::Details::KokkosExt;

    static_assert(
        KokkosExt::is_accessible_from<MemorySpace, ExecutionSpace>::value,
        "Memory space must be accessible from the execution space");

    // IndicesAndRanks must be a 1D view of ArborX::PairIndexRank
    static_assert(
        Kokkos::is_view_v<IndicesAndRanks> && IndicesAndRanks::rank == 1,
        "indices and ranks must be a 1D view of ArborX::PairIndexRank");
    static_assert(
        KokkosExt::is_accessible_from<typename IndicesAndRanks::memory_space,
                                      ExecutionSpace>::value,
        "indices and ranks must be accessible from the execution space");
    static_assert(std::is_same_v<typename IndicesAndRanks::non_const_value_type,
                                 PairIndexRank>,
                  "indices and ranks elements must be ArborX::PairIndexRank");

    _comm.reset(
        [comm]() {
          auto p = new MPI_Comm;
          MPI_Comm_dup(comm, p);
          return p;
        }(),
        [](MPI_Comm *p) {
          MPI_Comm_free(p);
          delete p;
        });

    // indices_and_ranks contains the indices and ranks of distant data. The
    // goal is to create a distributor that takes the distant data and sends it
    // where it is required

    int const data_len = indices_and_ranks.extent(0);
    int rank;
    MPI_Comm_rank(*_comm, &rank);

    Kokkos::View<int *, MemorySpace> dst_ranks(
        Kokkos::view_alloc(space, Kokkos::WithoutInitializing,
                           "ArborX::DistributedValuesDistributor::dst_ranks"),
        data_len);
    Kokkos::View<DVDKernelData *, MemorySpace> src_data(
        Kokkos::view_alloc(space, Kokkos::WithoutInitializing,
                           "ArborX::DistributedValuesDistributor::src_data"),
        data_len);
    DVDKernel1<decltype(dst_ranks), decltype(src_data),
               std::decay_t<decltype(indices_and_ranks)>>
        kernel1{dst_ranks, src_data, indices_and_ranks, rank};
    Kokkos::parallel_for(
        "ArborX::DistributedValuesDistributor::prepare_first_transfer",
        Kokkos::RangePolicy<ExecutionSpace>(space, 0, data_len), kernel1);

    _distributor = ArborX::Details::Distributor<MemorySpace>(*_comm);
    _num_requests = _distributor.createFromSends(space, dst_ranks);

    Kokkos::View<DVDKernelData *, MemorySpace> dst_data(
        Kokkos::view_alloc(space, Kokkos::WithoutInitializing,
                           "ArborX::DistributedValuesDistributor::dst_data"),
        _num_requests);
    ArborX::Details::DistributedTreeImpl<MemorySpace>::sendAcrossNetwork(
        space, _distributor, src_data, dst_data);

    _send_indices = Kokkos::View<int *, MemorySpace>(
        Kokkos::view_alloc(
            space, Kokkos::WithoutInitializing,
            "ArborX::DistributedValuesDistributor::send_indices"),
        _num_requests);
    Kokkos::View<int *, MemorySpace> src_ranks(
        Kokkos::view_alloc(space, Kokkos::WithoutInitializing,
                           "ArborX::DistributedValuesDistributor::src_ranks"),
        _num_requests);
    Kokkos::View<int *, MemorySpace> src_indices(
        Kokkos::view_alloc(space, Kokkos::WithoutInitializing,
                           "ArborX::DistributedValuesDistributor::src_indices"),
        _num_requests);
    DVDKernel2<decltype(_send_indices), decltype(dst_data), decltype(src_ranks),
               decltype(src_indices)>
        kernel2{_send_indices, dst_data, src_ranks, src_indices};
    Kokkos::parallel_for(
        "ArborX::DistributedValuesDistributor::prepare_second_transfer",
        Kokkos::RangePolicy<ExecutionSpace>(space, 0, _num_requests), kernel2);

    _distributor = ArborX::Details::Distributor<MemorySpace>(*_comm);
    _num_responses = _distributor.createFromSends(space, src_ranks);

    // The amount of data that will be received must be the same as the amount
    // of original data
    KOKKOS_ASSERT(_num_responses == data_len);

    _recv_indices = Kokkos::View<int *, MemorySpace>(
        Kokkos::view_alloc(
            space, Kokkos::WithoutInitializing,
            "ArborX::DistributedValuesDistributor::recv_indices"),
        _num_responses);
    ArborX::Details::DistributedTreeImpl<MemorySpace>::sendAcrossNetwork(
        space, _distributor, src_indices, _recv_indices);
  }

  template <typename ExecutionSpace, typename Values>
  void distribute(ExecutionSpace const &space, Values &values) const
  {
    auto guard = Kokkos::Profiling::ScopedRegion(
        "ArborX::DistributedValuesDistributor::distribute");

    namespace KokkosExt = ArborX::Details::KokkosExt;

    static_assert(
        KokkosExt::is_accessible_from<MemorySpace, ExecutionSpace>::value,
        "Memory space must be accessible from the execution space");

    // Values is a 1D view of values
    static_assert(Kokkos::is_view_v<Values> && Values::rank == 1,
                  "values must be a 1D view");
    static_assert(KokkosExt::is_accessible_from<typename Values::memory_space,
                                                ExecutionSpace>::value,
                  "values must be accessible from the execution space");
    static_assert(!std::is_const_v<typename Values::value_type>,
                  "values must be writable");

    using Value = typename Values::non_const_value_type;

    // We know what each process want so we prepare the data to be sent
    Kokkos::View<Value *, MemorySpace> data_to_send(
        Kokkos::view_alloc(
            space, Kokkos::WithoutInitializing,
            "ArborX::DistributedValuesDistributor::data_to_send"),
        _num_requests);
    Kokkos::parallel_for(
        "ArborX::DistributedValuesDistributor::data_to_send_fill",
        Kokkos::RangePolicy<ExecutionSpace>(space, 0, _num_requests),
        KOKKOS_CLASS_LAMBDA(int const i) {
          data_to_send(i) = values(_send_indices(i));
        });

    // We properly send the data, and each process has what it wants, but in the
    // wrong order
    Kokkos::View<Value *, MemorySpace> data_to_recv(
        Kokkos::view_alloc(
            space, Kokkos::WithoutInitializing,
            "ArborX::DistributedValuesDistributor::data_to_recv"),
        _num_responses);
    ArborX::Details::DistributedTreeImpl<MemorySpace>::sendAcrossNetwork(
        space, _distributor, data_to_send, data_to_recv);

    // So we fix this by moving everything
    Kokkos::resize(Kokkos::view_alloc(space, Kokkos::WithoutInitializing),
                   values, _num_responses);
    Kokkos::parallel_for(
        "ArborX::DistributedValuesDistributor::output_fill",
        Kokkos::RangePolicy<ExecutionSpace>(space, 0, _num_responses),
        KOKKOS_CLASS_LAMBDA(int const i) {
          values(_recv_indices(i)) = data_to_recv(i);
        });
  }

private:
  std::shared_ptr<MPI_Comm> _comm;
  Kokkos::View<int *, MemorySpace> _send_indices;
  Kokkos::View<int *, MemorySpace> _recv_indices;
  ArborX::Details::Distributor<MemorySpace> _distributor;
  int _num_requests;
  int _num_responses;
};

} // namespace ArborX::Interpolation::Details

#endif
