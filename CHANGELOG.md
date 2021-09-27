# Changelog

## [v1.1](https://github.com/arborx/arborx/tree/v1.1) (2021-09-23)

[Full Changelog](https://github.com/arborx/arborx/compare/v1.0...v1.1)

**Implemented enhancements:**

- Implement non-axis aligned bounding volumes [\#386](https://github.com/arborx/ArborX/pull/386)
- Compute distance sphere-box [\#495](https://github.com/arborx/ArborX/pull/495)
- Add Box-Box distance for Nearest query [\#494](https://github.com/arborx/ArborX/pull/494)
- Add benchmark using multiple execution space instances [\#538](https://github.com/arborx/ArborX/pull/538)
- Brute force algorithm for detecting collisions [\#534](https://github.com/arborx/ArborX/pull/534)
- Update FDBSCAN to use algorithm from FDBSCAN-DenseBox for single points [\#532](https://github.com/arborx/ArborX/pull/532)
- Change default DBSCAN implementation to FDBSCAN-DenseBox [\#529](https://github.com/arborx/ArborX/pull/529)
- Add and install ArborXConfigVersion.cmake [\#521](https://github.com/arborx/ArborX/pull/521)
- Add simple intersection example  [\#513](https://github.com/arborx/ArborX/pull/513)
- Introduce new algorithm \(FDBSCAN-DenseBox\) for DBSCAN [\#508](https://github.com/arborx/ArborX/pull/508)
- Add data converters for DBSCAN [\#491](https://github.com/arborx/ArborX/pull/491)

## [v1.0](https://github.com/arborx/arborx/tree/v1.0) (2021-03-13)

[Full Changelog](https://github.com/arborx/arborx/compare/v1.0-rc0...v1.0)

**Implemented enhancements:**

- Feature proposal: allow early termination of traversal [\#407](https://github.com/arborx/ArborX/issues/407)
- Add BruteForce data structure [\#468](https://github.com/arborx/ArborX/pull/468)
- Add a ray class and a test for ray-box intersection [\#414](https://github.com/arborx/ArborX/pull/414)
- Implement DBSCAN [\#331](https://github.com/arborx/ArborX/pull/331)
- Adding features to DBSCAN cmd line interface [\#490](https://github.com/arborx/ArborX/pull/490)
- Require CMake 3.16 [\#486](https://github.com/arborx/ArborX/pull/486)
- Align the meaning of min\_pts parameter in DBSCAN with literature [\#469](https://github.com/arborx/ArborX/pull/469)
- Add ray tracing example [\#461](https://github.com/arborx/ArborX/pull/461)
- Allow running BVH benchmark with SYCL and OpenMPTarget explicitly [\#455](https://github.com/arborx/ArborX/pull/455)
- Speed up DBSCAN post-processing [\#430](https://github.com/arborx/ArborX/pull/430)
- Terminate DBSCAN's neighbor counting early once threshold is reached [\#428](https://github.com/arborx/ArborX/pull/428)
- Enable all experimental backends [\#422](https://github.com/arborx/ArborX/pull/422)
- Use ropes on AMD GPUs as well [\#418](https://github.com/arborx/ArborX/pull/418)
- Change signature of the nearest callback [\#366](https://github.com/arborx/ArborX/pull/366)

## [v1.0-rc0](https://github.com/arborx/arborx/tree/v1.0-rc0) (2020-10-03)

[Full Changelog](https://github.com/arborx/arborx/compare/v0.9-beta...v1.0-rc0)

**Implemented enhancements:**

- Implement stackless tree traversal using escape index \(ropes\) [\#364](https://github.com/arborx/ArborX/pull/364)
- Enable HIP [\#236](https://github.com/arborx/ArborX/pull/236)
- Deprecate DistributedSearchTree [\#396](https://github.com/arborx/ArborX/pull/396)
- Remove deprecated nearest traversals [\#361](https://github.com/arborx/ArborX/pull/361)
- Improve performance of knn traversal [\#357](https://github.com/arborx/ArborX/pull/357)
- Add new query overloads for the distributed tree [\#356](https://github.com/arborx/ArborX/pull/356)
- Apetrei's construction algorithm [\#350](https://github.com/arborx/ArborX/pull/350)
- Allow non device type template parameter for output views in query\(\) on distributed trees [\#349](https://github.com/arborx/ArborX/pull/349)
- New BVH::query\(\) overload that only takes predicates and callback [\#329](https://github.com/arborx/ArborX/pull/329)

## [v0.9-beta](https://github.com/arborx/arborx/tree/v0.9-beta) (2020-06-10)

[Full Changelog](https://github.com/arborx/arborx/compare/v0.8-beta2...v0.9-beta)

**Implemented enhancements:**

- Add example for callbacks and lift requirement for tagging inline [\#325](https://github.com/arborx/ArborX/pull/325)
- Complete buffer optimization refactoring [\#282](https://github.com/arborx/ArborX/pull/282)
- Optional sorting of predicates [\#243](https://github.com/arborx/ArborX/pull/243)
- Allow sortResults to work for 2D and 3D [\#197](https://github.com/arborx/ArborX/pull/197)
- Enable user-defined callback in BVH search [\#166](https://github.com/arborx/ArborX/pull/166)
- Use with Trilinos' Kokkos installation [\#156](https://github.com/arborx/ArborX/pull/156)
- Add access traits CUDA example  [\#107](https://github.com/arborx/ArborX/pull/107)
- Kokkos 3.1 [\#268](https://github.com/arborx/ArborX/pull/268)
- Update examples to reflect the templated on memory space changes [\#261](https://github.com/arborx/ArborX/pull/261)
- Template distributed search tree on the memory space [\#260](https://github.com/arborx/ArborX/pull/260)
- Template BVH on memory space [\#251](https://github.com/arborx/ArborX/pull/251)
- Update tree query to take execution space argument [\#250](https://github.com/arborx/ArborX/pull/250)
- Update STL-like algorithms to take execution space as first argument [\#247](https://github.com/arborx/ArborX/pull/247)
- Take execution space as argument in tree construction [\#244](https://github.com/arborx/ArborX/pull/244)
- Halo finder [\#237](https://github.com/arborx/ArborX/pull/237)
- Use single precision floating point numbers for coordinates [\#235](https://github.com/arborx/ArborX/pull/235)
- Remove no longer necessary Trilinos-specific code to allow building against Trilinos installation of Kokkos [\#234](https://github.com/arborx/ArborX/pull/234)
- Require C++14 standard when building ArborX [\#226](https://github.com/arborx/ArborX/pull/226)
- Parallelize sortAndDetermineBufferLayout [\#172](https://github.com/arborx/ArborX/pull/172)
- CUDA-aware MPI [\#162](https://github.com/arborx/ArborX/pull/162)
- Require Kokkos CMake build [\#93](https://github.com/arborx/ArborX/pull/93)
- Logo [\#56](https://github.com/arborx/ArborX/pull/56)
- Make query sort optional [\#206](https://github.com/arborx/ArborX/issues/206)
- Friends-of-Friends Query [\#161](https://github.com/arborx/ArborX/issues/161)
- Let DistributedTree execute user payload  [\#86](https://github.com/arborx/ArborX/issues/86)

## [v0.8-beta2](https://github.com/arborx/arborx/tree/v0.8-beta2) (2019-10-10)

[Full Changelog](https://github.com/arborx/arborx/compare/97bbec21cc92dd2b4bd3a68c52a230b4c3c4643c...v0.8-beta2)

**Implemented enhancements:**

- Optimize communication within the same rank [\#134](https://github.com/arborx/ArborX/pull/134)
- Add intersects\(Point, Box\) [\#122](https://github.com/arborx/ArborX/pull/122)
- Deprecate old aliases to generate predicates [\#119](https://github.com/arborx/ArborX/pull/119)
- Add documentation [\#116](https://github.com/arborx/ArborX/pull/116)
- Refactor access traits [\#115](https://github.com/arborx/ArborX/pull/115)
- Promote access traits class template to ArborX:: namespace [\#112](https://github.com/arborx/ArborX/pull/112)
- Constexpr geometric primitives and algorithms [\#97](https://github.com/arborx/ArborX/pull/97)
- Improve benchmark for distributed tree [\#78](https://github.com/arborx/ArborX/pull/78)
- Add version\(\) and gitCommitHash\(\) [\#61](https://github.com/arborx/ArborX/pull/61)
- Make MPI dependency optional [\#42](https://github.com/arborx/ArborX/pull/42)
- Provide an option to disable tests and examples [\#31](https://github.com/arborx/ArborX/pull/31)
- Fix build system to run without TriBITS [\#14](https://github.com/arborx/ArborX/pull/14)

\* *This Changelog was automatically generated by [github_changelog_generator](https://github.com/github-changelog-generator/github-changelog-generator)*
