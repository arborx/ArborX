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

#include <Kokkos_Core.hpp>

#include <chrono>
#include <vector>

namespace
{

struct Timer
{
  using clock_type = std::chrono::high_resolution_clock;
  std::string label;
  clock_type::time_point tick = {};
  clock_type::duration duration = {};

  Timer(std::string label)
      : label(std::move(label))
  {}

  double elapsed() const
  {
    return std::chrono::duration<double>(duration).count();
  }
};

std::vector<Timer> arborx_dbscan_example_timers;

} // namespace

void arborx_dbscan_example_set_create_profile_section(char const *label,
                                                      std::uint32_t *id)
{
  *id = arborx_dbscan_example_timers.size();
  arborx_dbscan_example_timers.emplace_back(label);
}

void arborx_dbscan_example_set_destroy_profile_section(std::uint32_t) {}

void arborx_dbscan_example_set_start_profile_section(std::uint32_t id)
{
  Kokkos::fence();
  auto now = Timer::clock_type::now();
  auto &timer = arborx_dbscan_example_timers[id];
  timer.tick = now;
}

void arborx_dbscan_example_set_stop_profile_section(std::uint32_t id)
{
  Kokkos::fence();
  auto now = Timer::clock_type::now();
  auto &timer = arborx_dbscan_example_timers[id];
  timer.duration = now - timer.tick;
}

double arborx_dbscan_example_get_time(std::string const &label)
{
  for (auto const &timer : arborx_dbscan_example_timers)
    if (timer.label == label)
      return timer.elapsed();
  Kokkos::abort(("ArborX: no timer with label \"" + label + "\"").c_str());
}
