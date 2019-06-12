#!/usr/bin/env bash

# Based on https://cmake.org/gitweb?p=cmake.git;a=commitdiff;h=77543bd

this_program=$(basename "$0")
usage="Usage:
  $this_program [options] -- check lower case of CMake commands

Options:
    -h --help         Print help and exit
    -q --quiet        Quiet mode (do not print the diff)
    -p --apply-patch  Apply diff patch to the source files"

verbose=1
apply_patch=0

#echo "Arguments: $# $@"

while [ $# -gt 0 ]
do
  case $1 in
  -p|--apply-patch)
      apply_patch=1
      ;;
  -q|--quiet)
      verbose=0
      ;;
  -h|--help)
      echo "$usage"
      exit 0
      ;;
  *)
      echo "$this_program: Unknown argument '$1'. See '$this_program --help'."
      exit -1
      ;;
  esac

  shift
done

# shamelessy redirecting everything to /dev/null in quiet mode
if [ $verbose -eq 0 ]; then
    exec &>/dev/null
fi

CMAKE_COMMANDS_CONVERT_FILE="cmake_convert.sed"

cmake --help-command-list | grep -v "cmake version" | while read -r c; do
    echo 's/\b'"$(echo "$c" | tr '[:lower:]' '[:upper:]')"'\(\s*\)(/'"$c"'\1(/g'
done > "$CMAKE_COMMANDS_CONVERT_FILE"

cmake_source_files=$(git ls-files -- bootstrap '*.cmake' '*.cmake.in' '*CMakeLists.txt' | grep -E -v '^(Utilities/cm|Source/kwsys/)')

unformatted_files=()
for file in $cmake_source_files; do
    diff -u \
        <(cat "$file") \
        --label "a/$file" \
        <(sed -f "$CMAKE_COMMANDS_CONVERT_FILE" "$file") \
        --label "b/$file" >&1
    if [ $? -eq 1 ]; then
        unformatted_files+=($file)
    fi
done

n_unformatted_files=${#unformatted_files[@]}
if [ "$n_unformatted_files" -ne 0 ]; then
    echo "${#unformatted_files[@]} file(s) do(es) not use case properly:"
    for file in ${unformatted_files[@]}; do
        echo "    $file"
        if [ $apply_patch -eq 1 ]; then
            sed -f "$CMAKE_COMMANDS_CONVERT_FILE" -i "$file"
        fi
    done
else
    echo "OK"
fi
rm -f "$CMAKE_COMMANDS_CONVERT_FILE"
exit "$n_unformatted_files"
