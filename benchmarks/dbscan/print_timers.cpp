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

#include "print_timers.hpp"

#include <Kokkos_Core.hpp>

#include <chrono>
#include <stack>
#include <vector>

namespace
{

struct Timer
{
  using clock_type = std::chrono::high_resolution_clock;
  std::string label;
  clock_type::time_point tick;
  double duration;
};

std::stack<Timer> current_timers;
std::vector<Timer> done_timers;

} // namespace

void ArborX_Benchmark::push_region(char const *label)
{
  Kokkos::fence();
  auto now = Timer::clock_type::now();
  current_timers.push({label, now, {}});
}

void ArborX_Benchmark::pop_region()
{
  Kokkos::fence();
  auto now = Timer::clock_type::now();
  auto timer = current_timers.top();
  current_timers.pop();
  timer.duration = std::chrono::duration<double>(now - timer.tick).count();
  done_timers.push_back(timer);
}

double ArborX_Benchmark::get_time(std::string const &label)
{
  for (auto const &timer : done_timers)
    if (timer.label == label)
      return timer.duration;
  Kokkos::abort(("ArborX: no timer with label \"" + label + "\"").c_str());
}
