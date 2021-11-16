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

#include <ArborX_SOD.hpp>
#include <ArborX_Version.hpp>

#include <Kokkos_Core.hpp>

#include <boost/program_options.hpp>

#include <fstream>

struct InputData
{
  template <typename T>
  using View = Kokkos::View<T *, Kokkos::HostSpace>;

  View<ArborX::Point> particles;
  View<float> particle_masses;

  View<ArborX::Point> fof_halo_centers;
  View<float> fof_halo_masses;

  InputData()
      : particles("particles", 0)
      , particle_masses("particle_masses", 0)
      , fof_halo_centers("fof_halo_centers", 0)
      , fof_halo_masses("fof_halo_masses", 0)
  {
  }
};

template <typename MemorySpace>
struct OutputData
{
  template <typename T>
  using View = Kokkos::View<T *, MemorySpace>;
  template <typename T>
  using BinView = Kokkos::View<T **, MemorySpace>;

  View<float> sod_halo_masses;
  View<float> sod_halo_rdeltas;

  BinView<int> sod_halo_bin_ids;
  BinView<int> sod_halo_bin_counts;
  BinView<float> sod_halo_bin_masses;
  BinView<float> sod_halo_bin_outer_radii;
  BinView<float> sod_halo_bin_radial_velocities;

  View<int> sod_particles_offsets;
  View<int> sod_particles_indices;

  OutputData()
      : sod_halo_masses("sod_halo_masses", 0)
      , sod_halo_rdeltas("sod_halo_rdeltas", 0)
      , sod_halo_bin_ids("sod_halo_bin_ids", 0, 0)
      , sod_halo_bin_counts("sod_halo_bin_counts", 0, 0)
      , sod_halo_bin_masses("sod_halo_bin_masses", 0, 0)
      , sod_halo_bin_radial_velocities("sod_hlo_bin_radial_velocities", 0, 0)
  {
  }
};

template <typename ExecutionSpace, typename Permute, typename View>
std::enable_if_t<Kokkos::is_view<View>{} && View::rank == 1>
applyPermutation(ExecutionSpace const &exec_space, Permute const &permute,
                 View view)
{
  auto const n = view.extent_int(0);
  ARBORX_ASSERT(permute.extent_int(0) == n);

  auto view_clone = ArborX::clone(exec_space, view);
  for (int i = 0; i < n; ++i)
    view(i) = view_clone(permute(i));
}

template <typename ExecutionSpace, typename Permute, typename View>
std::enable_if_t<Kokkos::is_view<View>{} && View::rank == 2>
applyPermutation2(ExecutionSpace const &exec_space, Permute const &permute,
                  View view)
{
  auto const n = view.extent_int(0);
  ARBORX_ASSERT(permute.extent_int(0) == n);

  auto const m = view.extent_int(1);

  auto view_clone = ArborX::clone(exec_space, view);
  for (int i = 0; i < n; ++i)
    for (int j = 0; j < m; ++j)
      view(i, j) = view_clone(permute(i), j);
}

void loadParticlesData(std::string const &filename, InputData &in,
                       int max_num_points = -1)
{
  std::cout << "Reading in \"" << filename << "\" in binary mode...";
  std::cout.flush();

  std::ifstream input(filename, std::ifstream::binary);
  ARBORX_ASSERT(input.good());

  int N;
  input.read(reinterpret_cast<char *>(&N), sizeof(int));

  int n = N;
  if (max_num_points > 0 && max_num_points < N)
    n = max_num_points;

  Kokkos::resize(Kokkos::WithoutInitializing, in.particles, n);
  Kokkos::resize(Kokkos::WithoutInitializing, in.particle_masses, n);

  for (int d = 0; d < 3; ++d)
  {
    std::vector<float> tmp(n);
    input.read(reinterpret_cast<char *>(tmp.data()), n * sizeof(float));
    input.ignore((N - n) * sizeof(float));

    for (int i = 0; i < n; ++i)
      in.particles(i)[d] = tmp[i];
  }
  input.read(reinterpret_cast<char *>(in.particle_masses.data()),
             n * sizeof(float));
  input.ignore((N - n) * sizeof(float));

  std::cout << "done\nRead in " << n << " particles" << std::endl;

  input.close();
}

void loadHalosData(std::string const &filename, InputData &in,
                   Kokkos::View<int64_t *, Kokkos::HostSpace> &in_fof_halo_tags,
                   OutputData<Kokkos::HostSpace> &out)
{
  std::cout << "Reading in \"" << filename << "\" in binary mode...";
  std::cout.flush();

  std::ifstream input(filename, std::ifstream::binary);
  ARBORX_ASSERT(input.good());

  int num_halos;
  input.read(reinterpret_cast<char *>(&num_halos), 4);

  auto read_vector = [&input](auto &v, int n) {
    v.resize(n);
    input.read(reinterpret_cast<char *>(v.data()), n * sizeof(v[0]));
  };
  auto read_view = [&input](auto &view, int n) {
    Kokkos::resize(Kokkos::WithoutInitializing, view, n);
    input.read(reinterpret_cast<char *>(view.data()),
               n * sizeof(typename std::decay_t<decltype(view)>::value_type));
  };

  Kokkos::View<int *, Kokkos::HostSpace> in_fof_halo_sizes("fof_halo_sizes", 0);
  Kokkos::View<int64_t *, Kokkos::HostSpace> out_sod_halo_sizes(
      "sod_halo_sizes", 0);

  read_view(in_fof_halo_tags, num_halos);
  read_view(in_fof_halo_sizes, num_halos);
  read_view(in.fof_halo_masses, num_halos);
  {
    std::vector<float> x, y, z;
    read_vector(x, num_halos);
    read_vector(y, num_halos);
    read_vector(z, num_halos);

    Kokkos::resize(Kokkos::WithoutInitializing, in.fof_halo_centers, num_halos);
    for (int i = 0; i < num_halos; i++)
      in.fof_halo_centers(i) = {x[i], y[i], z[i]};
  }
  read_view(out.sod_halo_masses, num_halos);
  read_view(out_sod_halo_sizes, num_halos);
  read_view(out.sod_halo_rdeltas, num_halos);

  // Filter out
  // - small halos (FOF halo size < 500)
  // - invalid halos (SOD count size = -101)
  auto swap = [](auto &view, int i, int j) { std::swap(view(i), view(j)); };
  int i = 0;
  int num_filtered = 0;
  do
  {
    if (in_fof_halo_sizes(i) < 500 || out_sod_halo_sizes(i) < 0)
    {
      // Instead of using erase(), swap with the last element
      ++num_filtered;

      int j = num_halos - num_filtered;
      if (i < j)
      {
        swap(in_fof_halo_tags, i, j);
        swap(in_fof_halo_sizes, i, j);
        swap(in.fof_halo_masses, i, j);
        swap(in.fof_halo_centers, i, j);
        swap(out.sod_halo_masses, i, j);
        swap(out_sod_halo_sizes, i, j);
        swap(out.sod_halo_rdeltas, i, j);
      }
    }
    else
    {
      ++i;
    }
  } while (i < num_halos - num_filtered);

  if (num_filtered > 0)
  {
    num_halos -= num_filtered;
    Kokkos::resize(in_fof_halo_tags, num_halos);
    Kokkos::resize(in_fof_halo_sizes, num_halos);
    Kokkos::resize(in.fof_halo_masses, num_halos);
    Kokkos::resize(in.fof_halo_centers, num_halos);
    Kokkos::resize(out.sod_halo_masses, num_halos);
    Kokkos::resize(out_sod_halo_sizes, num_halos);
    Kokkos::resize(out.sod_halo_rdeltas, num_halos);
  }

  printf("done\nRead in %d halos [%d total, %d filtered]\n", num_halos,
         num_halos + num_filtered, num_filtered);

  // Sort halos by tags for consistency
  auto host_space = Kokkos::DefaultHostExecutionSpace{};
  auto permute = ArborX::Details::sortObjects(host_space, in_fof_halo_tags);
  applyPermutation(host_space, permute, in.fof_halo_masses);
  applyPermutation(host_space, permute, in.fof_halo_centers);
  applyPermutation(host_space, permute, out.sod_halo_masses);
  applyPermutation(host_space, permute, out.sod_halo_rdeltas);

  input.close();
}

void loadProfilesData(
    std::string const &filename, int num_sod_bins,
    OutputData<Kokkos::HostSpace> &out,
    Kokkos::View<int64_t *, Kokkos::HostSpace> &out_fof_halo_tags)
{
  std::cout << "Reading in \"" << filename << "\" in binary mode...";
  std::cout.flush();

  std::ifstream input(filename, std::ifstream::binary);
  ARBORX_ASSERT(input.good());

  // The profile file does not contain first bin with < R_min)
  ARBORX_ASSERT(num_sod_bins == 21);

  int num_records;
  input.read(reinterpret_cast<char *>(&num_records), 4);
  ARBORX_ASSERT(num_records % (num_sod_bins - 1) == 0);

  int num_halos = num_records / (num_sod_bins - 1);

  auto read_view = [&input](auto &view, int n) {
    Kokkos::resize(Kokkos::WithoutInitializing, view, n);
    input.read(reinterpret_cast<char *>(view.data()),
               n * sizeof(typename std::decay_t<decltype(view)>::value_type));
  };
  auto read_bin_view = [&input, &num_sod_bins](auto &view, int n) {
    using view_type = std::decay_t<decltype(view)>;
    using value_type = typename view_type::value_type;

    Kokkos::View<value_type *, typename view_type::device_type> v(
        Kokkos::ViewAllocateWithoutInitializing("tmp"), n * (num_sod_bins - 1));
    input.read(reinterpret_cast<char *>(v.data()),
               v.extent(0) * sizeof(value_type));

    // First bin is unused, shift data
    Kokkos::resize(Kokkos::WithoutInitializing, view, n, num_sod_bins);
    for (int i = 0; i < n; ++i)
    {
      view(i, 0) = -1; // just want a value that is clearly unidentifiable
      for (int j = 0; j < num_sod_bins - 1; ++j)
        view(i, j + 1) = v(i * (num_sod_bins - 1) + j);
    }
  };

  // FOF halo tags are repeated in groups of size num_sod_bins-1, make them
  // unique
  read_view(out_fof_halo_tags, num_records);
  for (int i = 1; i < num_halos; ++i)
    out_fof_halo_tags(i) = out_fof_halo_tags(i * (num_sod_bins - 1));
  Kokkos::resize(out_fof_halo_tags, num_halos);

  read_bin_view(out.sod_halo_bin_ids, num_halos);
  read_bin_view(out.sod_halo_bin_counts, num_halos);
  read_bin_view(out.sod_halo_bin_masses, num_halos);
  read_bin_view(out.sod_halo_bin_outer_radii, num_halos);
  read_bin_view(out.sod_halo_bin_radial_velocities, num_halos);

  // Sort halos by tags for consistency
  auto host_space = Kokkos::DefaultHostExecutionSpace{};
  auto permute = ArborX::Details::sortObjects(host_space, out_fof_halo_tags);
  applyPermutation2(host_space, permute, out.sod_halo_bin_ids);
  applyPermutation2(host_space, permute, out.sod_halo_bin_counts);
  applyPermutation2(host_space, permute, out.sod_halo_bin_masses);
  applyPermutation2(host_space, permute, out.sod_halo_bin_outer_radii);
  applyPermutation2(host_space, permute, out.sod_halo_bin_radial_velocities);

  printf("done\nRead in %d halos\n", num_halos);

  input.close();
}

template <typename MemorySpace>
struct MassAvgRadiiCountProfiles
{
  Kokkos::View<ArborX::Point *, MemorySpace> _particles;
  Kokkos::View<float *, MemorySpace> _particle_masses;
  Kokkos::View<ArborX::Point *, MemorySpace> _fof_halo_centers;
  Kokkos::View<double **, MemorySpace> _sod_halo_bin_masses;
  Kokkos::View<int **, MemorySpace> _sod_halo_bin_counts;

  KOKKOS_FUNCTION void operator()(int particle_index, int halo_index,
                                  int bin_id) const
  {
    Kokkos::atomic_fetch_add(&_sod_halo_bin_counts(halo_index, bin_id), 1);
    Kokkos::atomic_fetch_add(&_sod_halo_bin_masses(halo_index, bin_id),
                             _particle_masses(particle_index));
  }
};

// Compute R_min and R_max for each FOF halo
template <typename ExecutionSpace, typename FOFHaloMases>
std::pair<float, Kokkos::View<float *, typename FOFHaloMases::memory_space>>
computeSODRadii(ExecutionSpace const &exec_space,
                ArborX::SOD::Parameters const &params,
                FOFHaloMases const &fof_halo_masses)
{
  Kokkos::Profiling::pushRegion("ArborX::SOD::compute_sod_radii");

  using MemorySpace = typename FOFHaloMases::memory_space;

  float r_min = params._min_factor * params._r_smooth;

  auto const num_halos = fof_halo_masses.extent(0);
  Kokkos::View<float *, MemorySpace> r_max(
      Kokkos::view_alloc(Kokkos::WithoutInitializing, "ArborX::SOD::r_max"),
      num_halos);
  Kokkos::parallel_for(
      "ArborX::SOD::compute_r_max",
      Kokkos::RangePolicy<ExecutionSpace>(exec_space, 0, num_halos),
      KOKKOS_LAMBDA(int i) {
        float R_init = std::cbrt(fof_halo_masses(i) / params._sod_mass);
        r_max(i) = params._max_factor * R_init;
      });

  Kokkos::Profiling::popRegion();

  return std::make_pair(r_min, r_max);
}

template <typename ExecutionSpace, typename MemorySpace, typename Particles,
          typename ParticleMasses, typename FOFHaloCenters,
          typename FOFHaloMasses>
void sod(ExecutionSpace const &exec_space, Particles &particles,
         ParticleMasses &particle_masses, FOFHaloCenters &fof_halo_centers,
         FOFHaloMasses &fof_halo_masses, OutputData<MemorySpace> &out,
         ArborX::SOD::Parameters const &params)
{
  static_assert(
      KokkosExt::is_accessible_from<MemorySpace, ExecutionSpace>::value, "");
  static_assert(std::is_same<typename Particles::memory_space, MemorySpace>{},
                "");
  static_assert(
      std::is_same<typename ParticleMasses::memory_space, MemorySpace>{}, "");
  static_assert(
      std::is_same<typename FOFHaloCenters::memory_space, MemorySpace>{}, "");
  static_assert(
      std::is_same<typename FOFHaloMasses::memory_space, MemorySpace>{}, "");

  auto const num_halos = fof_halo_centers.extent(0);
  auto const num_sod_bins = params._num_sod_bins;

  // Compute R_min and R_max radii for every FOF halo
  float r_min;
  Kokkos::View<float *, MemorySpace> r_max;
  std::tie(r_min, r_max) =
      ArborX::Details::computeSODRadii(exec_space, params, fof_halo_masses);
  Kokkos::resize(fof_halo_masses, 0); // free as not used afterwards

  ArborX::SODHandle<MemorySpace> sod_handle(exec_space, fof_halo_centers, r_min,
                                            r_max);

  // Step 2: compute some profiles (mass, count, avg radius);
  // NOTE: we will accumulate float quantities into double in order to
  // avoid loss of precision, which will occur once we start adding small
  // quantities to large
  Kokkos::View<double **, MemorySpace> sod_halo_bin_masses(
      "ArborX::SOD::sod_halo_bin_masses", num_halos, num_sod_bins);
  Kokkos::resize(out.sod_halo_bin_counts, num_halos, num_sod_bins);
  sod_handle.computeBinProfiles(exec_space, particles, num_sod_bins,
                                MassAvgRadiiCountProfiles<MemorySpace>{
                                    particles, particle_masses,
                                    fof_halo_centers, sod_halo_bin_masses,
                                    out.sod_halo_bin_counts});

  Kokkos::resize(out.sod_halo_bin_masses, num_halos, num_sod_bins);
  Kokkos::parallel_for(
      "ArborX::SOD::copy_bin_masses",
      Kokkos::RangePolicy<ExecutionSpace>(exec_space, 0, num_halos),
      KOKKOS_LAMBDA(int halo_index) {
        // double -> float conversion
        for (int bin_id = 0; bin_id < num_sod_bins; ++bin_id)
          out.sod_halo_bin_masses(halo_index, bin_id) =
              sod_halo_bin_masses(halo_index, bin_id);
      });

  Kokkos::View<int *, MemorySpace> sod_halo_rdeltas_index(
      "Examples:sod_halo_rdeltas_index", 0);
  sod_handle.computeRdelta(exec_space, particles, particle_masses, params,
                           out.sod_halo_rdeltas, sod_halo_rdeltas_index);

  Kokkos::Profiling::popRegion();
}

int main(int argc, char *argv[])
{
  using ExecutionSpace = Kokkos::DefaultExecutionSpace;
  using MemorySpace = typename ExecutionSpace::memory_space;

  Kokkos::ScopeGuard guard(argc, argv);

  std::cout << "ArborX version    : " << ArborX::version() << std::endl;
  std::cout << "ArborX hash       : " << ArborX::gitCommitHash() << std::endl;

  namespace bpo = boost::program_options;

  int max_num_points;
  std::string filename_particles;
  std::string filename_halos;
  std::string filename_profiles;
  bool validate;

  bpo::options_description desc("Allowed options");
  // clang-format off
  desc.add_options()
      ( "help", "help message" )
      ( "filename-particles", bpo::value<std::string>(&filename_particles), "filename containing particles data" )
      ( "filename-halos", bpo::value<std::string>(&filename_halos), "filename containing halos data" )
      ( "filename-profiles", bpo::value<std::string>(&filename_profiles), "filename containing profiles data" )
      ( "max-num-points", bpo::value<int>(&max_num_points)->default_value(-1), "max number of points to read in" )
      ( "validate", bpo::bool_switch(&validate)->default_value(false), "output validation results" )
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

  // print out the runtime parameters
  printf("filename [particles] : %s [max_pts = %d]\n",
         filename_particles.c_str(), max_num_points);
  printf("filename [halos]     : %s\n", filename_halos.c_str());
  printf("filename [profiles]  : %s\n", filename_profiles.c_str());

  int const num_sod_bins = 20 + 1;

  // read in data
  InputData in_host;
  OutputData<Kokkos::HostSpace> validation_data;
  Kokkos::View<int64_t *, Kokkos::HostSpace> in_fof_halo_tags(
      "in_fof_halo_tags", 0);
  Kokkos::View<int64_t *, Kokkos::HostSpace> validation_fof_halo_tags(
      "in_fof_halo_tags", 0);
  loadParticlesData(filename_particles, in_host, max_num_points);
  loadHalosData(filename_halos, in_host, in_fof_halo_tags, validation_data);
  loadProfilesData(filename_profiles, num_sod_bins, validation_data,
                   validation_fof_halo_tags);

  int const num_halos = in_host.fof_halo_centers.extent_int(0);

  // Validate tags
  ARBORX_ASSERT(validation_fof_halo_tags.extent_int(0) ==
                in_fof_halo_tags.extent_int(0));
  for (int i = 0; i < num_halos; ++i)
    ARBORX_ASSERT(in_fof_halo_tags(i) == validation_fof_halo_tags(i));

  // run SOD
  ArborX::SOD::Parameters params;
  params.setNumSODBins(num_sod_bins)
      .setMinFactor(0.05)
      .setMaxFactor(2.0)
      .setRho(2.77536627e11)
      .setRhoRatio(200)
      .setRSmooth(250.f / 3072)
      .setSODMass(1e14);

  ExecutionSpace exec_space;

  // Transfer data from host to device
  Kokkos::Profiling::pushRegion("Example::copy_in_host_to_device");
  Kokkos::View<ArborX::Point *, MemorySpace> particles_device =
      Kokkos::create_mirror_view_and_copy(exec_space, in_host.particles);
  Kokkos::View<float *, MemorySpace> particle_masses_device =
      Kokkos::create_mirror_view_and_copy(exec_space, in_host.particle_masses);
  Kokkos::View<ArborX::Point *, MemorySpace> fof_halo_centers_device =
      Kokkos::create_mirror_view_and_copy(exec_space, in_host.fof_halo_centers);
  Kokkos::View<float *, MemorySpace> fof_halo_masses_device =
      Kokkos::create_mirror_view_and_copy(exec_space, in_host.fof_halo_masses);
  Kokkos::Profiling::popRegion();

  // Execute the kernels on the device
  OutputData<MemorySpace> out_device;
  sod(exec_space, particles_device, particle_masses_device,
      fof_halo_centers_device, fof_halo_masses_device, out_device, params);

  OutputData<Kokkos::HostSpace> out_host;

  // Transfer data from device to host
  Kokkos::Profiling::pushRegion("Example::copy_out_host_from_device");
  auto copy_bins_to_host = [](auto &view_host, auto &view_device) {
    auto view_mirror =
        Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, view_device);
    Kokkos::resize(Kokkos::WithoutInitializing, view_host,
                   view_device.extent(0), view_device.extent(1));
    Kokkos::deep_copy(view_host, view_mirror);
  };
  copy_bins_to_host(out_host.sod_halo_bin_masses,
                    out_device.sod_halo_bin_masses);
  copy_bins_to_host(out_host.sod_halo_bin_counts,
                    out_device.sod_halo_bin_counts);

  auto copy_to_host = [](auto &view_host, auto &view_device) {
    Kokkos::resize(Kokkos::WithoutInitializing, view_host,
                   view_device.extent(0));
    Kokkos::deep_copy(view_host, view_device);
  };
  copy_to_host(out_host.sod_halo_rdeltas, out_device.sod_halo_rdeltas);
  copy_to_host(out_host.sod_particles_offsets,
               out_device.sod_particles_offsets);
  copy_to_host(out_host.sod_particles_indices,
               out_device.sod_particles_indices);

  // validate
  if (validate)
  {
    auto const num_halos = in_host.fof_halo_centers.extent_int(0);

    auto relative_error = [](auto a, auto b) {
      if (a != 0)
        return std::abs(float(a - b) / a);
      if (a == 0 && b != 0)
        return std::numeric_limits<float>::infinity();
      return 0.f;
    };

    float max_error;

    // bin counts
    printf(">>> validating bin counts\n");
    for (int i = 0; i < num_halos; ++i)
    {
      bool matched = true;
      for (int bin_id = 1; bin_id < num_sod_bins; ++bin_id)
        matched &= (out_host.sod_halo_bin_counts(i, bin_id) ==
                    validation_data.sod_halo_bin_counts(i, bin_id));
      if (!matched)
      {
        printf("counts for halo tag %ld do not match: validation = [",
               in_fof_halo_tags(i));
        for (int bin_id = 1; bin_id < num_sod_bins; ++bin_id)
          printf(" %d", validation_data.sod_halo_bin_counts(i, bin_id));
        printf(" ], errors = [");
        for (int bin_id = 1; bin_id < num_sod_bins; ++bin_id)
          printf(" %d", out_host.sod_halo_bin_counts(i, bin_id) -
                            validation_data.sod_halo_bin_counts(i, bin_id));
        printf(" ]\n");
      }
    }

    // bin masses
    printf(">>> validating bin masses\n");
    max_error = 0.f;
    for (int i = 0; i < num_halos; ++i)
    {
      bool matched = true;
      for (int bin_id = 1; bin_id < num_sod_bins; ++bin_id)
        matched &= (out_host.sod_halo_bin_masses(i, bin_id) ==
                    validation_data.sod_halo_bin_masses(i, bin_id));
      if (!matched)
      {
        printf("masses for halo tag %ld do not match: relative errors [",
               in_fof_halo_tags(i));
        for (int bin_id = 1; bin_id < num_sod_bins; ++bin_id)
        {
          auto error =
              relative_error(out_host.sod_halo_bin_masses(i, bin_id),
                             validation_data.sod_halo_bin_masses(i, bin_id));
          max_error = std::max(error, max_error);
          printf(" %e", error);
        }
        printf(" ]\n");
      }
    }
    printf(">>> bin masses max error = %e\n", max_error);

    // r_delta
    printf(">>> validating rdelta\n");
    max_error = 0.f;
    for (int i = 0; i < num_halos; ++i)
    {
      float a = out_host.sod_halo_rdeltas(i);
      float b = validation_data.sod_halo_rdeltas(i);

      if (a != b)
      {
        auto error = relative_error(a, b);
        max_error = std::max(error, max_error);
        printf("%d rdelta for halo tag %ld do not match: "
               "output = %f, validation = %f, relative error = %e\n",
               i, in_fof_halo_tags(i), a, b, error);
      }
    }
    printf(">>> rdelta max error = %e\n", max_error);
  }

  return EXIT_SUCCESS;
}
