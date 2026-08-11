#! /usr/bin/bash

qsub -q instructional -l ncpus=1,ngpus=1 -M nicholas_storti1@baylor.edu -m bea $(realpath "$1")
