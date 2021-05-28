/****************************************************************************
 * Copyright (c) 2017-2021 by the ArborX authors                            *
 * All rights reserved.                                                     *
 *                                                                          *
 * This file is part of the ArborX library. ArborX is                       *
 * distributed under a BSD 3-clause license. For the licensing terms see    *
 * the LICENSE file in the top-level directory.                             *
 *                                                                          *
 * SPDX-License-Identifier: BSD-3-Clause                                    *
 ****************************************************************************/

#include <ArborX_Point.hpp>

#include <vector>

std::vector<ArborX::Point>
seedSpreader(int n, int dim = 3, bool variable_density = false,
             int num_clusters = 10, int c_reset = 100, float rho_noise = 1e-4);
