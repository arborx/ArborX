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

#include <string>

namespace ArborXBenchmark
{

struct Parameters
{
  std::string filename;
  std::string algorithm;
  bool binary;
  bool verify;
  bool print_dbscan_timers;
  float eps;
  int cluster_min_size;
  int core_min_size;
  int max_num_points;
  int num_samples;
  std::string filename_labels;
  std::string implementation;
};

template <int DIM>
bool run(Parameters const &params);

} // namespace ArborXBenchmark
