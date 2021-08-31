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

std::vector<ArborX::Point> loadData(std::string const &filename,
                                    bool binary = true, int max_num_points = -1)
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

  std::vector<ArborX::Point> v;

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

  if (!binary)
  {
    v.reserve(num_points);

    auto it = std::istream_iterator<float>(input);
    auto read_point = [&it, dim]() {
      float xyz[3] = {0.f, 0.f, 0.f};
      for (int i = 0; i < dim; ++i)
        xyz[i] = *it++;
      return ArborX::Point{xyz[0], xyz[1], xyz[2]};
    };
    std::generate_n(std::back_inserter(v), num_points, read_point);
  }
  else
  {
    v.resize(num_points);

    if (dim == 3)
    {
      // Can directly read into ArborX::Point
      input.read(reinterpret_cast<char *>(v.data()),
                 num_points * sizeof(ArborX::Point));
    }
    else
    {
      std::vector<float> aux(num_points * dim);
      input.read(reinterpret_cast<char *>(aux.data()),
                 aux.size() * sizeof(float));

      for (int i = 0; i < num_points; ++i)
      {
        ArborX::Point p{0.f, 0.f, 0.f};
        for (int d = 0; d < dim; ++d)
          p[d] = aux[i * dim + d];
        v[i] = p;
      }
    }
  }
  input.close();
  std::cout << "done\nRead in " << num_points << " " << dim << "D points"
            << std::endl;

  return v;
}

std::vector<ArborX::Point> sampleData(std::vector<ArborX::Point> const &data,
                                      int num_samples)
{
  std::vector<ArborX::Point> sampled_data(num_samples);

  // Knuth algorithm
  auto const N = (int)data.size();
  auto const M = num_samples;
  for (int in = 0, im = 0; in < N && im < M; ++in)
  {
    int rn = N - in;
    int rm = M - im;
    if (rand() % rn < rm)
      sampled_data[im++] = data[in];
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

template <typename... P, typename T>
auto vec2view(std::vector<T> const &in, std::string const &label = "")
{
  Kokkos::View<T *, P...> out(
      Kokkos::view_alloc(label, Kokkos::WithoutInitializing), in.size());
  Kokkos::deep_copy(out, Kokkos::View<T const *, Kokkos::HostSpace,
                                      Kokkos::MemoryTraits<Kokkos::Unmanaged>>{
                             in.data(), in.size()});
  return out;
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

                         Kokkos::atomic_fetch_add(&cluster_sizes(labels(i)), 1);
                       });

  // This kernel serves dual purpose:
  // - it constructs an offset array through exclusive prefix sum, with a
  //   caveat that small clusters (of size < cluster_min_size) are filtered out
  // - it creates a mapping from a cluster index into the cluster's position in
  //   the offset array
  // We reuse the cluster_sizes array for the second, creating a new alias for
  // it for clarity.
  auto &map_cluster_to_offset_position = cluster_sizes;
  int constexpr IGNORED_CLUSTER = -1;
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

template <typename ExecutionSpace, typename Primitives,
          typename ClusterIndicesView, typename ClusterOffsetView>
void printClusterSizesAndCenters(ExecutionSpace const &exec_space,
                                 Primitives const &primitives,
                                 ClusterIndicesView &cluster_indices,
                                 ClusterOffsetView &cluster_offset)
{
  auto const num_clusters = static_cast<int>(cluster_offset.size()) - 1;

  using MemorySpace = typename ClusterIndicesView::memory_space;

  Kokkos::View<ArborX::Point *, MemorySpace> cluster_centers(
      Kokkos::view_alloc(Kokkos::WithoutInitializing, "Testing::centers"),
      num_clusters);
  Kokkos::parallel_for(
      "Testing::compute_centers",
      Kokkos::RangePolicy<ExecutionSpace>(exec_space, 0, num_clusters),
      KOKKOS_LAMBDA(int const i) {
        // The only reason we sort indices here is for reproducibility.
        // Current DBSCAN algorithm does not guarantee that the indices
        // corresponding to the same cluster are going to appear in the same
        // order from run to run. Using sorted indices, we explicitly
        // guarantee the same summation order when computing cluster centers.

        auto *cluster_start = cluster_indices.data() + cluster_offset(i);
        auto cluster_size = cluster_offset(i + 1) - cluster_offset(i);

        // Sort cluster indices in ascending order. This uses heap for
        // sorting, only because there is no other convenient utility that
        // could sort within a kernel.
        ArborX::Details::makeHeap(cluster_start, cluster_start + cluster_size,
                                  ArborX::Details::Less<int>());
        ArborX::Details::sortHeap(cluster_start, cluster_start + cluster_size,
                                  ArborX::Details::Less<int>());

        // Compute cluster centers
        ArborX::Point cluster_center{0.f, 0.f, 0.f};
        for (int j = cluster_offset(i); j < cluster_offset(i + 1); j++)
        {
          auto const &cluster_point = primitives(cluster_indices(j));
          // NOTE The explicit casts below are intended to silence warnings
          // about narrowing conversion from 'int' to 'float'. A potential
          // accuracy issue here is that 'float' can represent all integer
          // values in the range [-2^23, 2^23] but 'int' can actually represent
          // values in the range [-2^31, 2^31-1]. However, we ignore it for
          // now.
          cluster_center[0] +=
              cluster_point[0] / static_cast<float>(cluster_size);
          cluster_center[1] +=
              cluster_point[1] / static_cast<float>(cluster_size);
          cluster_center[2] +=
              cluster_point[2] / static_cast<float>(cluster_size);
        }
        cluster_centers(i) = cluster_center;
      });

  auto cluster_offset_host =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, cluster_offset);
  auto cluster_centers_host =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, cluster_centers);
  for (int i = 0; i < num_clusters; i++)
  {
    int cluster_size = cluster_offset_host(i + 1) - cluster_offset_host(i);

    // This is HACC specific filtering. It is only interested in the clusters
    // with centers in [0,64]^3 domain.
    auto const &cluster_center = cluster_centers_host(i);
    if (cluster_center[0] >= 0 && cluster_center[1] >= 0 &&
        cluster_center[2] >= 0 && cluster_center[0] < 64 &&
        cluster_center[1] < 64 && cluster_center[2] < 64)
    {
      printf("%d %e %e %e\n", cluster_size, cluster_center[0],
             cluster_center[1], cluster_center[2]);
    }
  }
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

int main(int argc, char *argv[])
{
  using ExecutionSpace = Kokkos::DefaultExecutionSpace;
  using MemorySpace = typename ExecutionSpace::memory_space;

  Kokkos::ScopeGuard guard(argc, argv);

  std::cout << "ArborX version    : " << ArborX::version() << std::endl;
  std::cout << "ArborX hash       : " << ArborX::gitCommitHash() << std::endl;

  namespace bpo = boost::program_options;
  using ArborX::DBSCAN::Implementation;

  std::string filename;
  bool binary;
  bool verify;
  bool print_dbscan_timers;
  bool print_sizes_centers;
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
      ( "output-sizes-and-centers", bpo::bool_switch(&print_sizes_centers)->default_value(false), "print cluster sizes and centers")
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
  printf("output centers    : %s\n", (print_sizes_centers ? "true" : "false"));

  // read in data
  std::vector<ArborX::Point> data = loadData(filename, binary, max_num_points);
  if (num_samples > 0 && num_samples < (int)data.size())
    data = sampleData(data, num_samples);
  auto const primitives = vec2view<MemorySpace>(data, "primitives");

  ExecutionSpace exec_space;

  Kokkos::Timer timer_total;
  Kokkos::Timer timer;
  std::map<std::string, double> elapsed;

  bool const verbose = print_dbscan_timers;
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

  auto labels = ArborX::dbscan(exec_space, primitives, eps, core_min_size,
                               ArborX::DBSCAN::Parameters()
                                   .setPrintTimers(print_dbscan_timers)
                                   .setImplementation(implementation));

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
         (100.f * num_cluster_points / data.size()));

  if (verify)
  {
    auto passed = ArborX::Details::verifyDBSCAN(exec_space, primitives, eps,
                                                core_min_size, labels);
    printf("Verification %s\n", (passed ? "passed" : "failed"));
  }

  if (!filename_labels.empty())
    writeLabelsData(filename_labels, labels);

  if (print_sizes_centers)
    printClusterSizesAndCenters(exec_space, primitives, cluster_indices,
                                cluster_offset);

  return EXIT_SUCCESS;
}
