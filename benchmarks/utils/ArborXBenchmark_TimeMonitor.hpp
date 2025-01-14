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
#ifndef ARBORX_BENCHMARK_TIME_MONITOR
#define ARBORX_BENCHMARK_TIME_MONITOR

#include <algorithm> // min_element, max_element
#include <cassert>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <memory>  // unique_ptr
#include <numeric> // accumulate
#include <sstream>
#include <string>
#include <utility> // pair
#include <vector>

#include <mpi.h>

namespace ArborXBenchmark
{

// The TimeMonitor class can be used to measure for a series of events, i.e. it
// represents a set of timers of type Timer. It is a poor man's drop-in
// replacement for Teuchos::TimeMonitor
class TimeMonitor
{
  using container_type = std::vector<std::pair<std::string, double>>;
  using entry_reference_type = container_type::reference;
  container_type _data;

public:
  class Timer
  {
    entry_reference_type _entry;
    bool _started = false;
    std::chrono::high_resolution_clock::time_point _tick;

  public:
    Timer(entry_reference_type ref)
        : _entry{ref}
    {}
    void start()
    {
      assert(!_started);
      _tick = std::chrono::high_resolution_clock::now();
      _started = true;
    }
    void stop()
    {
      assert(_started);
      std::chrono::duration<double> duration =
          std::chrono::high_resolution_clock::now() - _tick;
      // NOTE I have put much thought into whether we should use the
      // operator+= and keep track of how many times the timer was
      // restarted.  To be honest I have not even looked was the original
      // TimeMonitor behavior is :)
      _entry.second = duration.count();
      _started = false;
    }
  };
  // NOTE Original code had the pointer semantics.  Can change in the future.
  // The smart pointer is a distraction.  The main problem here is that the
  // reference stored by the timer is invalidated if the time monitor gets
  // out of scope.
  std::unique_ptr<Timer> getNewTimer(std::string name)
  {
    // FIXME Consider searching whether there already is an entry with the
    // same name.
    _data.emplace_back(std::move(name), 0.);
    return std::make_unique<Timer>(_data.back());
  }

  void summarize(MPI_Comm comm, std::ostream &os = std::cout)
  {
    int comm_size;
    MPI_Comm_size(comm, &comm_size);
    int comm_rank;
    MPI_Comm_rank(comm, &comm_rank);
    int n_timers = _data.size();

    os << std::left << std::scientific;

    // Initialize with length of "Timer Name"
    std::string const timer_name = "Timer Name";
    std::size_t const max_section_length = std::accumulate(
        _data.begin(), _data.end(), timer_name.size(),
        [](std::size_t current_max, entry_reference_type section) {
          return std::max(current_max, section.first.size());
        });

    if (comm_size == 1)
    {
      std::string const header_without_timer_name = " | GlobalTime";
      std::stringstream dummy_string_stream;
      dummy_string_stream << std::setprecision(os.precision())
                          << std::scientific << " | " << 1.;
      int const header_width =
          max_section_length + std::max<int>(header_without_timer_name.size(),
                                             dummy_string_stream.str().size());

      os << std::string(header_width, '=') << "\n\n";
      os << "TimeMonitor results over 1 processor\n\n";
      os << std::setw(max_section_length) << timer_name
         << header_without_timer_name << '\n';
      os << std::string(header_width, '-') << '\n';
      for (int i = 0; i < n_timers; ++i)
      {
        os << std::setw(max_section_length) << _data[i].first << " | "
           << _data[i].second << '\n';
      }
      os << std::string(header_width, '=') << '\n';
      return;
    }
    std::vector<double> all_entries(comm_size * n_timers);
    std::transform(
        _data.begin(), _data.end(), all_entries.begin() + comm_rank * n_timers,
        [](std::pair<std::string, double> const &x) { return x.second; });
    // FIXME No guarantee that all processors have the same timers!
    MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, all_entries.data(),
                  n_timers, MPI_DOUBLE, comm);
    std::string const header_without_timer_name =
        " | MinOverProcs | MeanOverProcs | MaxOverProcs";
    if (comm_rank == 0)
    {
      os << std::string(max_section_length + header_without_timer_name.size(),
                        '=')
         << "\n\n";
      os << "TimeMonitor results over " << comm_size << " processors\n";
      os << std::setw(max_section_length) << timer_name
         << header_without_timer_name << '\n';
      os << std::string(max_section_length + header_without_timer_name.size(),
                        '-')
         << '\n';
    }
    std::vector<double> tmp(comm_size);
    for (int i = 0; i < n_timers; ++i)
    {
      for (int j = 0; j < comm_size; ++j)
      {
        tmp[j] = all_entries[j * n_timers + i];
      }
      auto min = *std::min_element(tmp.begin(), tmp.end());
      auto max = *std::max_element(tmp.begin(), tmp.end());
      auto mean = std::accumulate(tmp.begin(), tmp.end(), 0.) / comm_size;
      if (comm_rank == 0)
      {
        os << std::setw(max_section_length) << _data[i].first << " | " << min
           << " |  " << mean << " | " << max << '\n';
      }
    }
    if (comm_rank == 0)
    {
      os << std::string(max_section_length + header_without_timer_name.size(),
                        '=')
         << '\n';
    }
  }
};

} // namespace ArborXBenchmark

#endif
