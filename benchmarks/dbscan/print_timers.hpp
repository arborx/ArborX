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

void arborx_dbscan_example_set_create_profile_section(char const *,
                                                      std::uint32_t *);
void arborx_dbscan_example_set_destroy_profile_section(std::uint32_t);
void arborx_dbscan_example_set_start_profile_section(std::uint32_t);
void arborx_dbscan_example_set_stop_profile_section(std::uint32_t);
double arborx_dbscan_example_get_time(std::string const &label);
