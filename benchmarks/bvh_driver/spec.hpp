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

#ifndef SPEC_HPP
#define SPEC_HPP

#include <string>

#include <benchmark/benchmark.h>
#include <point_clouds.hpp>

struct Spec
{
  std::string backends;
  int n_values;
  int n_queries;
  int n_neighbors;
  bool sort_predicates;
  int buffer_size;
  PointCloudType source_point_cloud_type;
  PointCloudType target_point_cloud_type;

  Spec() = default;
  Spec(std::string const &spec_string)
  {
    std::istringstream ss(spec_string);
    std::string token;

    // clang-format off
    getline(ss, token, '/');  backends = token;
    getline(ss, token, '/');  n_values = std::stoi(token);
    getline(ss, token, '/');  n_queries = std::stoi(token);
    getline(ss, token, '/');  n_neighbors = std::stoi(token);
    getline(ss, token, '/');  sort_predicates = static_cast<bool>(std::stoi(token));
    getline(ss, token, '/');  buffer_size = std::stoi(token);
    getline(ss, token, '/');  source_point_cloud_type = static_cast<PointCloudType>(std::stoi(token));
    getline(ss, token, '/');  target_point_cloud_type = static_cast<PointCloudType>(std::stoi(token));
    // clang-format on

    if (!(backends == "all" || backends == "serial" || backends == "openmp" ||
          backends == "threads" || backends == "cuda" || backends == "rtree" ||
          backends == "hip" || backends == "sycl" ||
          backends == "openmptarget"))
      throw std::runtime_error("Backend " + backends + " invalid!");
  }

  std::string create_label_construction(std::string const &tree_name) const
  {
    std::string s = std::string("BM_construction<") + tree_name + ">";
    for (auto const &var :
         {n_values, static_cast<int>(source_point_cloud_type)})
      s += "/" + std::to_string(var);
    return s;
  }

  std::string create_label_radius_search(std::string const &tree_name,
                                         std::string const &flavor = "") const
  {
    std::string s = std::string("BM_radius_") +
                    (flavor.empty() ? "" : flavor + "_") + "search<" +
                    tree_name + ">";
    for (auto const &var :
         {n_values, n_queries, n_neighbors, static_cast<int>(sort_predicates),
          buffer_size, static_cast<int>(source_point_cloud_type),
          static_cast<int>(target_point_cloud_type)})
      s += "/" + std::to_string(var);
    return s;
  };

  std::string create_label_knn_search(std::string const &tree_name,
                                      std::string const &flavor = "") const
  {
    std::string s = std::string("BM_knn_") +
                    (flavor.empty() ? "" : flavor + "_") + "search<" +
                    tree_name + ">";
    for (auto const &var :
         {n_values, n_queries, n_neighbors, static_cast<int>(sort_predicates),
          static_cast<int>(source_point_cloud_type),
          static_cast<int>(target_point_cloud_type)})
      s += "/" + std::to_string(var);
    return s;
  };
};

#endif
