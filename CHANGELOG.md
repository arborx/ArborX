# Changelog

## [2.0.1](https://github.com/arborx/arborx/tree/v2.0.1) (2025-07-17)

[Full Changelog](https://github.com/arborx/arborx/compare/v2.0...v2.0.1)

**Fixed bugs:**
- Fix a bug in experimental predicate helpers (e.g., `make_intersects`) requiring `memory_space` typedef [\#1255](https://github.com/arborx/ArborX/pull/1255)
- Add missing `distance(sphere, point)` and `distance(box, sphere)` [\#1266](https://github.com/arborx/ArborX/pull/1266)
- Fix usage of hardcoded floats in distance calculations [\#1267](https://github.com/arborx/ArborX/pull/1267)
- Fix Clang 17 type deduction failures with aggregate initialization [\#1279](https://github.com/arborx/ArborX/pull/1279)

## [2.0](https://github.com/arborx/arborx/tree/v2.0) (2025-04-16)

[Full Changelog](https://github.com/arborx/arborx/compare/v1.7...v2.0)

**New features:**
- New interfaces for search indexes (APIv2) [\#1155](https://github.com/arborx/ArborX/pull/1155), [\#1157](https://github.com/arborx/ArborX/pull/1157), [\#1161](https://github.com/arborx/ArborX/pull/1161)
- Hyper-dimensional geometries  [\#1100](https://github.com/arborx/ArborX/pull/1100), [\#1148](https://github.com/arborx/ArborX/pull/1148), [\#1146](https://github.com/arborx/ArborX/pull/1146), [\#1154](https://github.com/arborx/ArborX/pull/1154)

**Build Changes:**
- Require C++20 [\#1215](https://github.com/arborx/ArborX/pull/1215)
- Require Kokkos 4.5.00 [\#1233](https://github.com/arborx/ArborX/pull/1233)
- Require CMake 3.22 [\#1188](https://github.com/arborx/ArborX/pull/1188)

**Enhancements:**
- Promote `ArborX::Triangle` from the EXPERIMENTAL namespace [\#1154](https://github.com/arborx/ArborX/pull/1154)
- Add `Segment` geometry [EXPERIMENTAL] [\#1165](https://github.com/arborx/ArborX/pull/1065)
- cmake: make install path configurable [\#1174](https://github.com/arborx/ArborX/pull/1174)
- Add `DefaultIndexableGetter` [EXPERIMENTAL] [\#1181](https://github.com/arborx/ArborX/pull/1181)
- Define `ARBORX_VERSION` number macro [\#1190](https://github.com/arborx/ArborX/pull/1190)
- Allow running DBSCAN in different precision [\#1203](https://github.com/arborx/ArborX/pull/1203)
- Move `MinimumSpanningTree` to `Experimental` namespace [\#1213](https://github.com/arborx/ArborX/pull/1213)
- Add ellipsoid geometry [EXPERIMENTAL] [\#1222](https://github.com/arborx/ArborX/pull/1222)

**Backward incompatible changes:**
- Generalize geometries to support any dimension and precision [\#1100](https://github.com/arborx/ArborX/pull/1100), [\#1148](https://github.com/arborx/ArborX/pull/1148), [\#1146](https://github.com/arborx/ArborX/pull/1146), [\#1154](https://github.com/arborx/ArborX/pull/1154)
- Remove deprecated algorithms from `ArborX` namespace (e.g., `min`, `max` `lastElement`,`clone`) [\#1151](https://github.com/arborx/ArborX/pull/1151)
- Remove old interfaces (APIv1) from indexes [\#1155](https://github.com/arborx/ArborX/pull/1155), [\#1157](https://github.com/arborx/ArborX/pull/1157), [\#1161](https://github.com/arborx/ArborX/pull/1161)
- Remove `ArborX::PairIndexRank` [\#1156](https://github.com/arborx/ArborX/pull/1156)
- Remove `ArborX::PrimitivesTag` and `ArborX::PredicatesTag` from `ArborX::AccessTraits` [\#1172](https://github.com/arborx/ArborX/pull/1172)
- Switch `AnyNewerVersion` to `SameMajorVersion` in CMake [\#1188](https://github.com/arborx/ArborX/pull/1188)
- Add helper functions for predicates construction `make_{intersects,nearest}` [\#1232](https://github.com/arborx/ArborX/pull/1232)

**Fixed bugs:**
- Do not divide by zero when computing ray-AABB intersection [\#1226](https://github.com/arborx/ArborX/pull/1226)

## [1.7](https://github.com/arborx/arborx/tree/v1.7) (2024-09-03)

[Full Changelog](https://github.com/arborx/arborx/compare/v1.6...v1.7)

**New features:**
- Distributed tree: add support for a nearest query with a callback [EXPERIMENTAL] [\#1075](https://github.com/arborx/ArborX/pull/1075)
- Distributed tree: add support for a spatial query only taking a callback [\#733](https://github.com/arborx/ArborX/pull/733)

**Build Changes:**
- Require Kokkos 4.2.00 [\#1054](https://github.com/arborx/ArborX/pull/1054)

**Enhancements:**
- Implement box-triangle intersection [\#1059](https://github.com/arborx/ArborX/pull/1059)
- Add tetrahedron geometry [EXPERIMENTAL] [\#1079](https://github.com/arborx/ArborX/pull/1079)
- Expand KDOP to support different dimensions and precision [EXPERIMENTAL] [\#982](https://github.com/arborx/ArborX/pull/982)
- Add 2D KDOP geometries [EXERIMENTAL] [\#1088](https://github.com/arborx/ArborX/pull/1088)
- Improve performance of the distributed algorithms [\#1098](https://github.com/arborx/ArborX/pull/1098), [\#1103](https://github.com/arborx/ArborX/pull/1103)
- Add `make_ordered_intersects` helper function to implicitly construct ordered intersect queries for a set of geometries [EXPERIMENTAL] [\#1117](https://github.com/arborx/ArborX/pull/1117)

**Backward incompatible changes:**
- Remove deprecated `InlineCallbackTag` [\#1078](https://github.com/arborx/ArborX/pull/1078)
- Remove deprecated `Traits::Access` [\#1078](https://github.com/arborx/ArborX/pull/1078)

**Deprecations:**
- Deprecate `+=` and `reduction_identity` for `Box` [\#1082](https://github.com/arborx/ArborX/pull/1082)
- Deprecate `+=` in `KDOP` [\#982](https://github.com/arborx/ArborX/pull/982)

**Fixed bugs:**
- Fix a Kokkos bounds check warning reported when using FDBSCAN-DenseBox [\#1067](https://github.com/arborx/ArborX/pull/1067)
- Fix DBSCAN test on Intel GPUs [\#1112](https://github.com/arborx/ArborX/pull/1112)
- Fix policy warning when using CMake 3.30 with Boost [\#1123](https://github.com/arborx/ArborX/pull/1123)
- Fix geometry traits for `const` geometries [\#1129](https://github.com/arborx/ArborX/pull/1129)

## [1.6](https://github.com/arborx/arborx/tree/v1.6) (2024-04-11)

[Full Changelog](https://github.com/arborx/arborx/compare/v1.5...v1.6)

**New features:**
- Introduce new API for ArborX indexes (EXPERIMENTAL) [\#970](https://github.com/arborx/ArborX/pull/970), [\#1017](https://github.com/arborx/ArborX/pull/1017)
- Add Moving Least Squares (MLS) interpolation (EXPERIMENTAL) [\#946](https://github.com/arborx/ArborX/pull/946), [\#992](https://github.com/arborx/ArborX/pull/992), [\#1000](https://github.com/arborx/ArborX/pull/1000)

**Build Changes:**
- Require Kokkos 4.1.00 [\#973](https://github.com/arborx/ArborX/pull/973)

**Enhancements:**
- Introduce `PairValueIndex` and `Experimental::attach_indices` [\#969](https://github.com/arborx/ArborX/pull/969), [\#1016](https://github.com/arborx/ArborX/pull/1016), [\#1036](https://github.com/arborx/ArborX/pull/1036)
- Improve examples [\#1008](https://github.com/arborx/ArborX/pull/1008), [\#1009](https://github.com/arborx/ArborX/pull/1009), [\#936](https://github.com/arborx/ArborX/pull/936)
- Auto-fetch Google benchmark when not found locally [\#1039](https://github.com/arborx/ArborX/pull/1039)
- Allow default initialized distributed tree [\#1040](https://github.com/arborx/ArborX/pull/1040)
- Implement distance point-triangle [\#1046](https://github.com/arborx/ArborX/pull/1046)
- Implement nearest query for BruteForce [\#1053](https://github.com/arborx/ArborX/pull/1053)
- Add helper functions to construct predicates [\#1038](https://github.com/arborx/ArborX/pull/1038)
- Add an example of a distributed tree k-nearest neighbors search [\#724](https://github.com/arborx/ArborX/pull/724)
- Add triangulated surface distance benchmark [\#1052](https://github.com/arborx/ArborX/pull/1052)

**Deprecations:**
- Deprecate `min`, `max`, `minMax` [\#998](https://github.com/arborx/ArborX/pull/998)
- Deprecate `iota`, `exclusivePrefixSum`, `accumulate`, `adjacentDifference` [\#999](https://github.com/arborx/ArborX/pull/999)

**Fixed bugs:**
- Fixed CUDA build warning [\#1010](https://github.com/arborx/ArborX/pull/1010)
- Fixed HIP build with ROCm 6 [\#1030](https://github.com/arborx/ArborX/pull/1030)
- Fixed FDBSCAN-DenseBox issue with user provided `AccessTraits` [\#1045](https://github.com/arborx/ArborX/pull/1045)
- Fixed stream destruction order in CUDA access traits example and execution spaces benchmark [\#1050](https://github.com/arborx/ArborX/pull/1050)

## [1.5](https://github.com/arborx/arborx/tree/v1.5) (2023-12-16)

[Full Changelog](https://github.com/arborx/arborx/compare/v1.4.1...v1.5)

**Build Changes:**
- Require Kokkos 4.0.00 [\#895](https://github.com/arborx/ArborX/pull/895)

**Enhancements:**
- Add a multi-dimensional triangle (EXPERIMENTAL) [\#916](https://github.com/arborx/ArborX/pull/916)
- Add example of triangle-point intersections [\#542](https://github.com/arborx/ArborX/pull/542)
- Allow running individual queries in non-batch mode (EXPERIMENTAL) [\#917](https://github.com/arborx/ArborX/pull/917)
- Support for MSVC (core only) [\#908](https://github.com/arborx/ArborX/pull/908)

**Fixed bugs:**
- Fix compilation for CUDA-12.2 [\#933](https://github.com/arborx/ArborX/pull/933)
- Fix a bug in the dendrogram generation [\#955](https://github.com/arborx/ArborX/pull/955)
- Fix compilation for C++20 [\#884](https://github.com/arborx/ArborX/pull/884)

## [1.4.1](https://github.com/arborx/arborx/tree/v1.4.1) (2023-06-08)

[Full Changelog](https://github.com/arborx/arborx/compare/v1.4...v1.4.1)

**Build Changes:**
- Add support for Trilinos 14.0 [\#886](https://github.com/arborx/ArborX/pull/886)

## [1.4](https://github.com/arborx/arborx/tree/v1.4) (2023-05-05)

[Full Changelog](https://github.com/arborx/arborx/compare/v1.3...v1.4)

**New features:**
- Add HDBSCAN (EXPERIMENTAL) [\#777](https://github.com/arborx/ArborX/pull/777), [\#783](https://github.com/arborx/ArborX/pull/783), [\#845](https://github.com/arborx/ArborX/pull/845)

**Build changes:**
- Require Kokkos 3.7.01 [\#834](https://github.com/arborx/ArborX/pull/834), [\#749](https://github.com/arborx/ArborX/pull/749)
- Require Google Benchmark 1.5.4 when building benchmarks [\#799](https://github.com/arborx/ArborX/pull/799)

**Enhancements:**
- Add a faster implementation of the union-find for Serial [\#767](https://github.com/arborx/ArborX/pull/767), [\#780](https://github.com/arborx/ArborX/pull/780)
- Annotate fences [\#805](https://github.com/arborx/ArborX/pull/805)
- Improve performance of FDBSCAN [\#810](https://github.com/arborx/ArborX/pull/810)
- Add new facility to find "half" or "full" neighbor lists (EXPERIMENTAL) [\#809](https://github.com/arborx/ArborX/pull/809), [\#812](https://github.com/arborx/ArborX/pull/812)
- Introduce `ArborX::PairIndexRank` [\#819](https://github.com/arborx/ArborX/pull/819)
- Reduce memory consumption when sending data in `DistributedTree` [\#829](https://github.com/arborx/ArborX/pull/829)
- Avoid extra copy when sending across network in `sendAcrossNetwork` [\#830](https://github.com/arborx/ArborX/pull/830)
- Deprecate `ARBORX_USE_CUDA_AWARE_MPI` and replace it with `ARBORX_USE_GPU_AWARE_MPI` [\#828](https://github.com/arborx/ArborX/pull/828)
- Allow 7D data [\#853](https://github.com/arborx/ArborX/pull/853), [\#854](https://github.com/arborx/ArborX/pull/854)

**Backward incompatible changes:**
- Remove deprecated queryDispatch calls in the distributed tree [\#793](https://github.com/arborx/ArborX/pull/793)
- Remove volatile overloads in `ArborX::Box` and `ArborX::Point` [\#838](https://github.com/arborx/ArborX/pull/838)

**Fixed bugs:**
- Fix a bug when using distributed tree with rays [\#786](https://github.com/arborx/ArborX/pull/786)
- Fix undefined behavior when parsing boolean options in DBSCAN benchmark [\#804](https://github.com/arborx/ArborX/pull/804)
- Fix compilation for CUDA 11.0-11.3 [\#850](https://github.com/arborx/ArborX/pull/850)

## [1.3](https://github.com/arborx/arborx/tree/v1.3) (2022-10-18)

[Full Changelog](https://github.com/arborx/arborx/compare/v1.2...v1.3)

**New features:**
- Add multi-dimensional support (EXPERIMENTAL) [\#713](https://github.com/arborx/ArborX/pull/713), [\#722](https://github.com/arborx/ArborX/pull/722), [\#726](https://github.com/arborx/ArborX/pull/726), [\#731](https://github.com/arborx/ArborX/pull/731)

**Build changes:**
- Require C++17 [\#715](https://github.com/arborx/ArborX/pull/715)
- Require Kokkos 3.6 [\#748](https://github.com/arborx/ArborX/pull/748)

**Enhancements:**
- Switch to using stackless spatial traversal for all backends [\#672](https://github.com/arborx/ArborX/pull/672)
- Add ordered tree traversal (EXPERIMENTAL) [\#683](https://github.com/arborx/ArborX/pull/683)
- Add a ray-tracing example using ordered tree traversal [\#691](https://github.com/arborx/ArborX/pull/691)
- Add `distance()` function to the nearest predicate [\#700](https://github.com/arborx/ArborX/pull/700)
- Template `BruteForce` on bounding volume [\#712](https://github.com/arborx/ArborX/pull/712)
- Throw an exception when FDBSCAN-DenseBox may potentially produce wrong result [\#560](https://github.com/arborx/ArborX/pull/560)
- Migrate full DBSCAN driver from `examples/` to `benchmarks/` [\#736](https://github.com/arborx/ArborX/pull/736)
- Add a DBSCAN example [\#763](https://github.com/arborx/ArborX/pull/763)

**Backward incompatible changes:**
- Fail compilation when using the old `ArborX::Traits` interface [\#667](https://github.com/arborx/ArborX/pull/667)
- Remove deprecated `BVH` version templated on a device type [\#705](https://github.com/arborx/ArborX/pull/705)
- Remove deprecated `DistributedSearchTree` and `DistributedTree` templated on a device type [\#706](https://github.com/arborx/ArborX/pull/706)

**Fixed bugs:**
- Make sure default-initialized `BVH` and `BruteForce` do not give undefined behavior [\#692](https://github.com/arborx/ArborX/pull/692)
- Avoid use of uninitialized core distances in the minimum spanning tree [\#753](https://github.com/arborx/ArborX/pull/753)

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
