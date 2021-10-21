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

#include <ArborX_DBSCAN.hpp>
#include <ArborX_DBSCANVerification.hpp>
#include <ArborX_DetailsHeap.hpp>
#include <ArborX_DetailsOperatorFunctionObjects.hpp> // Less
#include <ArborX_Version.hpp>

#include <Kokkos_Core.hpp>

#include <boost/program_options.hpp>

#include <fstream>

template <typename MemorySpace, int DIM>
struct MultiDimensionalData
{
  using memory_space = MemorySpace;

  Kokkos::View<float **, MemorySpace> _data;
};

template <typename MemorySpace, int DIM>
struct ArborX::AccessTraits<MultiDimensionalData<MemorySpace, DIM>,
                            ArborX::PrimitivesTag>
{
  using memory_space = MemorySpace;
  using Primitives = MultiDimensionalData<MemorySpace, DIM>;

  static KOKKOS_FUNCTION std::size_t size(Primitives const &data)
  {
    return data._data.extent(0);
  }
  template <int D = DIM, std::enable_if_t<D == 1> * = nullptr>
  static KOKKOS_FUNCTION ArborX::Point get(Primitives const &data,
                                           std::size_t i)
  {
    return ArborX::Point{data._data(i, 0), 0.f, 0.f};
  }
  template <int D = DIM, std::enable_if_t<D == 2> * = nullptr>
  static KOKKOS_FUNCTION ArborX::Point get(Primitives const &data,
                                           std::size_t i)
  {
    return ArborX::Point{data._data(i, 0), data._data(i, 1), 0.f};
  }
  template <int D = DIM, std::enable_if_t<D == 3> * = nullptr>
  static KOKKOS_FUNCTION ArborX::Point get(Primitives const &data,
                                           std::size_t i)
  {
    return ArborX::Point{data._data(i, 0), data._data(i, 1), data._data(i, 2)};
  }
};

Kokkos::View<float **, Kokkos::HostSpace> loadData(std::string const &filename,
                                                   bool binary = true,
                                                   int max_num_points = -1)
{
  std::cout << "Reading in \"" << filename << "\" in "
            << (binary ? "binary" : "text") << " mode...";
  std::cout.flush();

  std::ifstream input;
  if (!binary)
    input.open(filename);
  else
    input.open(filename, std::ifstream::binary);
  ARBORX_ASSERT(input.good());

  int num_points = 0;
  int dim = 0;
  if (!binary)
  {
    input >> num_points;
    input >> dim;
  }
  else
  {
    input.read(reinterpret_cast<char *>(&num_points), sizeof(int));
    input.read(reinterpret_cast<char *>(&dim), sizeof(int));
  }

  // For now, only allow reading in 2D or 3D data. Will relax in the future.
  ARBORX_ASSERT(dim == 2 || dim == 3);

  if (max_num_points > 0 && max_num_points < num_points)
    num_points = max_num_points;

  Kokkos::View<float **, Kokkos::HostSpace> v("Example::primitives", num_points,
                                              dim);
  if (!binary)
  {
    auto it = std::istream_iterator<float>(input);
    for (int i = 0; i < num_points; ++i)
      for (int j = 0; j < dim; ++j)
        v(i, j) = *it++;
  }
  else
  {
    input.read(reinterpret_cast<char *>(v.data()),
               num_points * dim * sizeof(float));
  }
  input.close();
  std::cout << "done\nRead in " << num_points << " " << dim << "D points"
            << std::endl;

  return v;
}

Kokkos::View<float **, Kokkos::HostSpace>
sampleData(Kokkos::View<float **, Kokkos::HostSpace> const &data,
           int num_samples)
{
  auto const n = data.extent_int(0);
  auto const m = num_samples;

  if (m <= 0 || m >= n)
    return data;

  auto const dim = data.extent_int(1);
  Kokkos::View<float **, Kokkos::HostSpace> sampled_data(data.label(),
                                                         num_samples, dim);

  // Knuth algorithm
  for (int in = 0, im = 0; in < n && im < m; ++in)
  {
    int rn = n - in;
    int rm = m - im;
    if (rand() % rn < rm)
    {
      for (int j = 0; j < dim; ++j)
        sampled_data(im, j) = data(in, j);
      ++im;
    }
  }
  return sampled_data;
}

template <typename MemorySpace>
void writeLabelsData(std::string const &filename,
                     Kokkos::View<int *, MemorySpace> labels)
{
  std::ofstream out(filename, std::ofstream::binary);
  ARBORX_ASSERT(out.good());

  auto labels_host =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, labels);

  int n = labels_host.size();
  out.write((char *)&n, sizeof(int));
  out.write((char *)labels_host.data(), sizeof(int) * n);
}

template <typename ExecutionSpace, typename LabelsView,
          typename ClusterIndicesView, typename ClusterOffsetView>
void sortAndFilterClusters(ExecutionSpace const &exec_space,
                           LabelsView const &labels,
                           ClusterIndicesView &cluster_indices,
                           ClusterOffsetView &cluster_offset,
                           int cluster_min_size = 1)
{
  Kokkos::Profiling::pushRegion("ArborX::DBSCAN::sortAndFilterClusters");

  static_assert(Kokkos::is_view<LabelsView>{}, "");
  static_assert(Kokkos::is_view<ClusterIndicesView>{}, "");
  static_assert(Kokkos::is_view<ClusterOffsetView>{}, "");

  using MemorySpace = typename LabelsView::memory_space;

  static_assert(std::is_same<typename LabelsView::value_type, int>{}, "");
  static_assert(std::is_same<typename ClusterIndicesView::value_type, int>{},
                "");
  static_assert(std::is_same<typename ClusterOffsetView::value_type, int>{},
                "");

  static_assert(std::is_same<typename LabelsView::memory_space, MemorySpace>{},
                "");
  static_assert(
      std::is_same<typename ClusterIndicesView::memory_space, MemorySpace>{},
      "");
  static_assert(
      std::is_same<typename ClusterOffsetView::memory_space, MemorySpace>{},
      "");

  ARBORX_ASSERT(cluster_min_size >= 1);

  int const n = labels.extent_int(0);

  Kokkos::View<int *, MemorySpace> cluster_sizes(
      "ArborX::DBSCAN::cluster_sizes", n);
  Kokkos::parallel_for("ArborX::DBSCAN::compute_cluster_sizes",
                       Kokkos::RangePolicy<ExecutionSpace>(exec_space, 0, n),
                       KOKKOS_LAMBDA(int const i) {
                         // Ignore noise points
                         if (labels(i) < 0)
                           return;

                         Kokkos::atomic_increment(&cluster_sizes(labels(i)));
                       });

  // This kernel serves dual purpose:
  // - it constructs an offset array through exclusive prefix sum, with a
  //   caveat that small clusters (of size < cluster_min_size) are filtered out
  // - it creates a mapping from a cluster index into the cluster's position in
  //   the offset array
  // We reuse the cluster_sizes array for the second, creating a new alias for
  // it for clarity.
  auto &map_cluster_to_offset_position = cluster_sizes;
  constexpr int IGNORED_CLUSTER = -1;
  int num_clusters;
  ArborX::reallocWithoutInitializing(cluster_offset, n + 1);
  Kokkos::parallel_scan(
      "ArborX::DBSCAN::compute_cluster_offset_with_filter",
      Kokkos::RangePolicy<ExecutionSpace>(exec_space, 0, n),
      KOKKOS_LAMBDA(int const i, int &update, bool final_pass) {
        bool is_cluster_too_small = (cluster_sizes(i) < cluster_min_size);
        if (!is_cluster_too_small)
        {
          if (final_pass)
          {
            cluster_offset(update) = cluster_sizes(i);
            map_cluster_to_offset_position(i) = update;
          }
          ++update;
        }
        else
        {
          if (final_pass)
            map_cluster_to_offset_position(i) = IGNORED_CLUSTER;
        }
      },
      num_clusters);
  Kokkos::resize(Kokkos::WithoutInitializing, cluster_offset, num_clusters + 1);
  ArborX::exclusivePrefixSum(exec_space, cluster_offset);

  auto cluster_starts = ArborX::clone(exec_space, cluster_offset);
  ArborX::reallocWithoutInitializing(cluster_indices,
                                     ArborX::lastElement(cluster_offset));
  Kokkos::parallel_for("ArborX::DBSCAN::compute_cluster_indices",
                       Kokkos::RangePolicy<ExecutionSpace>(exec_space, 0, n),
                       KOKKOS_LAMBDA(int const i) {
                         // Ignore noise points
                         if (labels(i) < 0)
                           return;

                         auto offset_pos =
                             map_cluster_to_offset_position(labels(i));
                         if (offset_pos != IGNORED_CLUSTER)
                         {
                           auto position = Kokkos::atomic_fetch_add(
                               &cluster_starts(offset_pos), 1);
                           cluster_indices(position) = i;
                         }
                       });

  Kokkos::Profiling::popRegion();
}

namespace ArborX
{
namespace DBSCAN
{
// This function is required for Boost program_options to be able to use the
// Implementation enum.
std::istream &operator>>(std::istream &in, Implementation &implementation)
{
  std::string impl_string;
  in >> impl_string;

  if (impl_string == "fdbscan")
    implementation = ArborX::DBSCAN::Implementation::FDBSCAN;
  else if (impl_string == "fdbscan-densebox")
    implementation = ArborX::DBSCAN::Implementation::FDBSCAN_DenseBox;
  else
    in.setstate(std::ios_base::failbit);

  return in;
}

// This function is required for Boost program_options to use Implementation
// enum as the default_value().
std::ostream &operator<<(std::ostream &out,
                         Implementation const &implementation)
{
  switch (implementation)
  {
  case ArborX::DBSCAN::Implementation::FDBSCAN:
    out << "fdbscan";
    break;
  case ArborX::DBSCAN::Implementation::FDBSCAN_DenseBox:
    out << "fdbscan-densebox";
    break;
  }
  return out;
}
} // namespace DBSCAN
} // namespace ArborX

template <typename ExecutionSpace, typename MemorySpace, int DIM>
auto main_(ExecutionSpace const &exec_space,
           MultiDimensionalData<MemorySpace, DIM> const &primitives, float eps,
           int core_min_size, ArborX::DBSCAN::Parameters const &dbscan_params,
           int cluster_min_size, bool verify)
{
  Kokkos::Timer timer_total;
  Kokkos::Timer timer;
  std::map<std::string, double> elapsed;

  bool const verbose = dbscan_params._print_timers;
  auto timer_start = [&exec_space, verbose](Kokkos::Timer &timer) {
    if (verbose)
      exec_space.fence();
    timer.reset();
  };
  auto timer_seconds = [&exec_space, verbose](Kokkos::Timer const &timer) {
    if (verbose)
      exec_space.fence();
    return timer.seconds();
  };

  timer_start(timer_total);

  int n = primitives._data.extent(0);

  auto labels =
      ArborX::dbscan(exec_space, primitives, eps, core_min_size, dbscan_params);

  timer_start(timer);
  Kokkos::View<int *, MemorySpace> cluster_indices("Testing::cluster_indices",
                                                   0);
  Kokkos::View<int *, MemorySpace> cluster_offset("Testing::cluster_offset", 0);
  sortAndFilterClusters(exec_space, labels, cluster_indices, cluster_offset,
                        cluster_min_size);
  elapsed["cluster"] = timer_seconds(timer);
  elapsed["total"] = timer_seconds(timer_total);

  printf("-- postprocess      : %10.3f\n", elapsed["cluster"]);
  printf("total time          : %10.3f\n", elapsed["total"]);

  int num_clusters = cluster_offset.size() - 1;
  int num_cluster_points = cluster_indices.size();
  printf("\n#clusters       : %d\n", num_clusters);
  printf("#cluster points : %d [%.2f%%]\n", num_cluster_points,
         (100.f * num_cluster_points / n));

  bool success = true;
  if (verify)
  {
    success = ArborX::Details::verifyDBSCAN(exec_space, primitives, eps,
                                            core_min_size, labels);
    printf("Verification %s\n", (success ? "passed" : "failed"));
  }

  return std::make_pair(success, labels);
}

int main(int argc, char *argv[])
{
  using ExecutionSpace = Kokkos::DefaultExecutionSpace;
  using MemorySpace = typename ExecutionSpace::memory_space;

  Kokkos::ScopeGuard guard(argc, argv);

  std::cout << "ArborX version    : " << ArborX::version() << std::endl;
  std::cout << "ArborX hash       : " << ArborX::gitCommitHash() << std::endl;
  std::cout << "Kokkos version    : " << KokkosExt::version() << std::endl;

  namespace bpo = boost::program_options;
  using ArborX::DBSCAN::Implementation;

  std::string filename;
  bool binary;
  bool verify;
  bool print_dbscan_timers;
  float eps;
  int cluster_min_size;
  int core_min_size;
  int max_num_points;
  int num_samples;
  std::string filename_labels;
  Implementation implementation;

  bpo::options_description desc("Allowed options");
  // clang-format off
  desc.add_options()
      ( "help", "help message" )
      ( "filename", bpo::value<std::string>(&filename), "filename containing data" )
      ( "binary", bpo::bool_switch(&binary)->default_value(false), "binary file indicator")
      ( "max-num-points", bpo::value<int>(&max_num_points)->default_value(-1), "max number of points to read in")
      ( "eps", bpo::value<float>(&eps), "DBSCAN eps" )
      ( "cluster-min-size", bpo::value<int>(&cluster_min_size)->default_value(1), "minimum cluster size")
      ( "core-min-size", bpo::value<int>(&core_min_size)->default_value(2), "DBSCAN min_pts")
      ( "verify", bpo::bool_switch(&verify)->default_value(false), "verify connected components")
      ( "samples", bpo::value<int>(&num_samples)->default_value(-1), "number of samples" )
      ( "labels", bpo::value<std::string>(&filename_labels)->default_value(""), "clutering results output" )
      ( "print-dbscan-timers", bpo::bool_switch(&print_dbscan_timers)->default_value(false), "print dbscan timers")
      ( "impl", bpo::value<Implementation>(&implementation)->default_value(Implementation::FDBSCAN), R"(implementation ("fdbscan" or "fdbscan-densebox"))")
      ;
  // clang-format on
  bpo::variables_map vm;
  bpo::store(bpo::command_line_parser(argc, argv).options(desc).run(), vm);
  bpo::notify(vm);

  if (vm.count("help") > 0)
  {
    std::cout << desc << '\n';
    return 1;
  }

  std::stringstream ss;
  ss << implementation;

  // Print out the runtime parameters
  printf("eps               : %f\n", eps);
  printf("minpts            : %d\n", core_min_size);
  printf("cluster min size  : %d\n", cluster_min_size);
  printf("filename          : %s [%s, max_pts = %d]\n", filename.c_str(),
         (binary ? "binary" : "text"), max_num_points);
  printf("filename [labels] : %s [binary]\n", filename_labels.c_str());
  printf("implementation    : %s\n", ss.str().c_str());
  printf("samples           : %d\n", num_samples);
  printf("verify            : %s\n", (verify ? "true" : "false"));
  printf("print timers      : %s\n", (print_dbscan_timers ? "true" : "false"));

  ExecutionSpace exec_space;

  // read in data
  auto data_host = loadData(filename, binary, max_num_points);

  // sample data
  data_host = sampleData(data_host, num_samples);

  // copy data to device
  Kokkos::View<float **, MemorySpace> data_device(
      Kokkos::view_alloc(Kokkos::WithoutInitializing, "Example::primitives"),
      data_host.extent(0), data_host.extent(1));
  {
    // Do the copy into the new layout on the host, as the GPU memory may be
    // limited
    auto data_host_device_layout = Kokkos::create_mirror_view(data_device);
    Kokkos::deep_copy(data_host_device_layout, data_host);
    Kokkos::deep_copy(exec_space, data_device, data_host_device_layout);
  }

  ArborX::DBSCAN::Parameters dbscan_params;
  dbscan_params.setPrintTimers(print_dbscan_timers)
      .setImplementation(implementation)
      .setDimension(data_device.extent(1));

  auto const dim = data_host.extent(1);
  bool success;
  Kokkos::View<int *, MemorySpace> labels("Example::labels", 0);
  switch (dim)
  {
  case 1:
  {
    auto mddata = MultiDimensionalData<MemorySpace, 1>{data_device};
    std::tie(success, labels) = main_(exec_space, mddata, eps, core_min_size,
                                      dbscan_params, cluster_min_size, verify);
    break;
  }
  case 2:
    std::tie(success, labels) =
        main_(exec_space, MultiDimensionalData<MemorySpace, 2>{data_device},
              eps, core_min_size, dbscan_params, cluster_min_size, verify);
    break;
  case 3:
    std::tie(success, labels) =
        main_(exec_space, MultiDimensionalData<MemorySpace, 3>{data_device},
              eps, core_min_size, dbscan_params, cluster_min_size, verify);
    break;
  default:
    success = false;
  }

  if (success && !filename_labels.empty())
    writeLabelsData(filename_labels, labels);

  return success ? EXIT_SUCCESS : EXIT_FAILURE;
}
