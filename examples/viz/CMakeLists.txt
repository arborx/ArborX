TRIBITS_ADD_EXECUTABLE_AND_TEST(
  example_viz
  SOURCES tree_visualization.cpp
  NUM_MPI_PROCS 1
  FAIL_REGULAR_EXPRESSION "data race;leak;runtime error"
  )
