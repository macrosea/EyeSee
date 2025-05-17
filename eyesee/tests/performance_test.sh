#!/usr/bin/env bash

#python -m cProfile  performance.py > /tmp/performance.log
pyfiles=
index=0
dir=/tmp/ws/git/work/demo-project
for f in $(find $dir -name "*.py"); do
  fn=$(basename ${f})
  if [ "$fn" == *__init__.py ]; then
    continue
  fi
  if [ -z $pyfiles ]; then
    pyfiles="\b$fn"
  else
    pyfiles="${pyfiles}|\b$fn"
  fi
done
if [ -z $pyfiles ]; then
  exit -1
fi
pyfiles="\"$pyfiles\""
#echo $pyfiles
echo -e "ncall       tottime   percall  cumtime   percall    filename:lineno(function)"
grep -E $pyfiles /tmp/performance.log | grep -v "0.000    0.000    0.000    0.000"
