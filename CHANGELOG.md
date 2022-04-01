# Changelog

## [1.2](https://github.com/arborx/arborx/tree/v1.2) (2022-04-01)

[Full Changelog](https://github.com/arborx/arborx/compare/v1.1...v1.2)

**New features:**

- Implement Euclidean minimum spanning tree algorithm (EXPERIMENTAL) [\#624](https://github.com/arborx/ArborX/pull/624), [\#631](https://github.com/arborx/ArborX/pull/631), [\#661](https://github.com/arborx/ArborX/pull/631)

**Build changes:**

- Require Kokkos 3.4 [\#578](https://github.com/arborx/ArborX/pull/578)

**Enhancements:**

- Indicate the status of the worktree in the hash string ("-dirty") [\#558](https://github.com/arborx/ArborX/pull/558)
- Check valid callback and space accessibility [\#563](https://github.com/arborx/ArborX/pull/563)
- Add a molecular dynamics example [\#564](https://github.com/arborx/ArborX/pull/564)
- Improve performance of the HIP backend when using HIP-Clang [\#575](https://github.com/arborx/ArborX/pull/575)
- Add Point-Sphere intersection algorithm [\#584](https://github.com/arborx/ArborX/pull/584)
- Avoid host copy in distributed tree construction when using CUDA-aware MPI [\#597](https://github.com/arborx/ArborX/pull/597)
- Print Kokkos version during configuration [\#609](https://github.com/arborx/ArborX/pull/609)
- Improve performance of the brute force algorithm [\#616](https://github.com/arborx/ArborX/pull/616)
- Use 64-bit Morton indices by default in the construction [\#637](https://github.com/arborx/ArborX/pull/637)
- Allow alternate space filling curves (EXPERIMENTAL) [\#646](https://github.com/arborx/ArborX/pull/646)
- Add ray-triangle intersection algorithm [\#617](https://github.com/arborx/ArborX/pull/617)

**Deprecations:**

- Deprecate `lastElement` helper function in the `ArborX` namespace [\#648](https://github.com/arborx/ArborX/pull/648)
- Deprecate `ArborX::InlineCallbackTag` [\#656](https://github.com/arborx/ArborX/pull/656)

**Fixed bugs:**

- Fix the exit status of the DBSCAN example [\#566](https://github.com/arborx/ArborX/pull/566)
- Bring DBSCAN README documentation in line with the code [\#568](https://github.com/arborx/ArborX/pull/568), [\#668](https://github.com/arborx/ArborX/pull/668)
- Fix a race condition in the BVH construction [\#579](https://github.com/arborx/ArborX/pull/579)
- Fix a bug in the nearest search using the DistributedTree [\#653](https://github.com/arborx/ArborX/pull/653)

## [1.1](https://github.com/arborx/arborx/tree/v1.1) (2021-09-23)

[Full Changelog](https://github.com/arborx/arborx/compare/v1.0...v1.1)

**New features:**

- Allow k-DOP as a bounding volume in BVH [\#386](https://github.com/arborx/ArborX/pull/386)

**Enhancements:**

- Store a dimension value in a DBSCAN datafile [\#523](https://github.com/arborx/ArborX/pull/523)
- Enable knn queries with spheres and boxes [\#495](https://github.com/arborx/ArborX/pull/495) and [\#494](https://github.com/arborx/ArborX/pull/494)
- Use double-precision floating-points to normalize `Ray` direction [\#535](https://github.com/arborx/ArborX/pull/535)
- Improve performance of the brute force algorithm [\#534](https://github.com/arborx/ArborX/pull/534)
- Improve performance of the DBSCAN algorithm (new FDBSCAN-DenseBox algorithm) [\#508](https://github.com/arborx/ArborX/pull/508)
- Allow non-View input for DBSCAN [\#509](https://github.com/arborx/ArborX/pull/509)
- Install CMake package version file [\#521](https://github.com/arborx/ArborX/pull/521)
- Add a simple intersection example [\#513](https://github.com/arborx/ArborX/pull/513)
- Improve performance for the SYCL backend through the use of oneDPL for sorting [\#456](https://github.com/arborx/ArborX/pull/456)

**Fixed bugs:**

- Fix a bug in DBSCAN noise marking [\#525](https://github.com/arborx/ArborX/pull/525)

## [1.0](https://github.com/arborx/arborx/tree/v1.0) (2021-03-13)

[Full Changelog](https://github.com/arborx/arborx/compare/v1.0-rc0...v1.0)

**New features:**

- Allow early termination of a traversal by a thread [\#427](https://github.com/arborx/ArborX/pull/427)
- Implement DBSCAN clustering algorithm [\#331](https://github.com/arborx/ArborX/pull/331)
- Implement brute-force algorithm [\#468](https://github.com/arborx/ArborX/pull/468)
- Add initial ray-tracing support [\#414](https://github.com/arborx/ArborX/pull/414) and [\#461](https://github.com/arborx/ArborX/pull/461)

**Build changes:**

- Require CMake 3.16 [\#486](https://github.com/arborx/ArborX/pull/486)

**Enhancements:**

- Add `KOKKOS_FUNCTION` to `AccessTraits::size()` in View specialization [\#463](https://github.com/arborx/ArborX/pull/463)
- Add `query()` free function [\#425](https://github.com/arborx/ArborX/pull/425)
- Improve performance for the HIP backend through the use of rocThrust for sorting [\#424](https://github.com/arborx/ArborX/pull/424)
- Support for SYCL and OpenMPTarget [\#422](https://github.com/arborx/ArborX/pull/422)

**Backward incompatible changes:**

- Change signature of the nearest callback [\#366](https://github.com/arborx/ArborX/pull/366)

## [1.0-rc0](https://github.com/arborx/arborx/tree/v1.0-rc0) (2020-10-03)

[Full Changelog](https://github.com/arborx/arborx/compare/v0.9-beta...v1.0-rc0)

**New features:**

- New `BVH::query()` overload that only takes predicates and callback [\#329](https://github.com/arborx/ArborX/pull/329)

**Enhancements:**

- Implement stackless tree traversal using escape index (ropes) [\#364](https://github.com/arborx/ArborX/pull/364)
- Add support for Kokkos HIP backend [\#236](https://github.com/arborx/ArborX/pull/236)
- Ensure that all kernels and memory allocations are prefixed with `ArborX::` [\#362](https://github.com/arborx/ArborX/issues/362) and [\#380](https://github.com/arborx/ArborX/pull/380)
- Improve performance of knn traversal [\#357](https://github.com/arborx/ArborX/pull/357)
- Add new query overloads for the distributed tree [\#356](https://github.com/arborx/ArborX/pull/356)
- Increase performance of the BVH construction [\#350](https://github.com/arborx/ArborX/pull/350)
- Allow non device type template parameter for output views in query\(\) on distributed trees [\#349](https://github.com/arborx/ArborX/pull/349)

**Fixed bugs:**

- Fix double free when making copies of a distributed tree [\#369](https://github.com/arborx/ArborX/pull/369)
- Resolve duplicate `Details::toBufferStatus(int)` symbol error downstream [\#360](https://github.com/arborx/ArborX/pull/360)
- Fix narrowing conversion warnings [\#343](https://github.com/arborx/ArborX/pull/343)

**Deprecations:**

- Deprecate `DistributedSearchTree` in favor of `DistributedTree` [\#396](https://github.com/arborx/ArborX/pull/396)

## [0.9-beta](https://github.com/arborx/arborx/tree/v0.9-beta) (2020-06-10)

[Full Changelog](https://github.com/arborx/arborx/compare/v0.8-beta2...v0.9-beta)

**New features:**

- Enable user-defined callback in BVH search [\#166](https://github.com/arborx/ArborX/pull/166)
- Make predicates sorting optional [\#243](https://github.com/arborx/ArborX/pull/243)
- Use user-provided execution space instances [\#250](https://github.com/arborx/ArborX/pull/250)
- Implement algorithm for finding strongly connected components (halo finder) [\#237](https://github.com/arborx/ArborX/pull/237)
- Store bounding boxes in single-precision floating-point format [\#235](https://github.com/arborx/ArborX/pull/235)
- Allow usage of CUDA-aware MPI [\#162](https://github.com/arborx/ArborX/pull/162)

**Build changes:**

- Require Kokkos 3.1 [\#268](https://github.com/arborx/ArborX/pull/268)
- Require C++14 standard when building ArborX [\#226](https://github.com/arborx/ArborX/pull/226)
- Require Kokkos CMake build [\#93](https://github.com/arborx/ArborX/pull/93)

**Enhancements:**

- Template BVH on memory space [\#251](https://github.com/arborx/ArborX/pull/251)
- Add example for callbacks and lift requirement for tagging inline [\#325](https://github.com/arborx/ArborX/pull/325)
- Enable building against Trilinos' Kokkos installation [\#156](https://github.com/arborx/ArborX/pull/156)
- Add access traits CUDA example [\#107](https://github.com/arborx/ArborX/pull/107)
- Let `BVH::bounds()` be a `KOKKOS_FUNCTION` [\#326](https://github.com/arborx/ArborX/pull/326)
- Improve performance of the radius search [\#306](https://github.com/arborx/ArborX/pull/306)
- Improve performance of the kNN search [\#308](https://github.com/arborx/ArborX/pull/308)
- Retain the original path to Kokkos [\#287](https://github.com/arborx/ArborX/pull/287)
- Disable tests, examples, benchmarks by default [\#284](https://github.com/arborx/ArborX/pull/284)
- Template distributed search tree on the memory space [\#260](https://github.com/arborx/ArborX/pull/260)
- Enable predicates access traits in distributed search tree [\#196](https://github.com/arborx/ArborX/pull/196)
- Set default build type to `RelWithDebInfo` [\#188](https://github.com/arborx/ArborX/pull/188)
- Remove all fences [\#150](https://github.com/arborx/ArborX/pull/150)
- Improve performance for sorting with CUDA by using Thrust [\#147](https://github.com/arborx/ArborX/pull/147)
- Improve compilation error messages produced by `BVH::query()` [\#279](https://github.com/arborx/ArborX/pulls/279)

**Fixed bugs:**

- Fix ambiguity in `queryDispatch()` overload resolution [\#293](https://github.com/arborx/ArborX/pull/293)
- Properly update hash in version file when building from subdirs [\#266](https://github.com/arborx/ArborX/pull/266)
- Avoid second pass for radius search when the results are empty [\#240](https://github.com/arborx/ArborX/pull/240)
- Avoid more compiler warnings for `nvcc_wrapper` [\#185](https://github.com/arborx/ArborX/pull/185)
- Fix segfault in `Distributor` [\#296](https://github.com/arborx/ArborX/pulls/296)
- Allow non device type template parameter for output views in `BVH::query()` [\#271](https://github.com/arborx/ArborX/pull/271)

**Deprecations:**

- Deprecate `Traits::Access` in favor of `AccessTraits` [\#300](https://github.com/arborx/ArborX/pull/300)

## [0.8-beta2](https://github.com/arborx/arborx/tree/v0.8-beta2) (2019-10-10)

[Full Changelog](https://github.com/arborx/arborx/compare/97bbec21cc92dd2b4bd3a68c52a230b4c3c4643c...v0.8-beta2)

**New features:**

- Provide access traits for predicates [\#117](https://github.com/arborx/ArborX/pull/117)
- Add `version()` and `gitCommitHash()` [\#61](https://github.com/arborx/ArborX/pull/61)

**Build changes:**

- Replace TriBITS build system with "raw" CMake [\#14](https://github.com/arborx/ArborX/pull/14)
- Rename `ArborX_ENABLE_XYZ` options to `ARBORX_ENABLE_XYZ` [\#99](https://github.com/arborx/ArborX/pull/99)
- Provide an option to disable tests and examples [\#31](https://github.com/arborx/ArborX/pull/31)
- Make MPI dependency optional [\#42](https://github.com/arborx/ArborX/pull/42)

**Enhancements:**

- Use `MPI_Comm_dup` to separate ArborX comm context from user's [\#135](https://github.com/arborx/ArborX/pull/135)
- Add CMake option for enabling benchmarks [\#138](https://github.com/arborx/ArborX/pull/138)
- Add intersection of a `Point` with `Box` [\#122](https://github.com/arborx/ArborX/pull/122)
- Improve error messages in BVH constructor and BVH::query\(\) [\#113](https://github.com/arborx/ArborX/pull/113)
- Relax CudaUVM requirement [\#24](https://github.com/arborx/ArborX/pull/24)
- Find Boost in subdirectories that actually require it [\#22](https://github.com/arborx/ArborX/pull/22)

**Fixed bugs:**

- Optimize communication within the same rank [\#134](https://github.com/arborx/ArborX/pull/134)
- Fix for distributed searches with large count of results per query [\#129](https://github.com/arborx/ArborX/pull/129)
