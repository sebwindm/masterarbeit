#!/bin/bash

gnuplot -e "set key autotitle columnhead; plot for [col=2:4] 'q_values_learned_results.csv' using 1:col with lines; pause mouse close; " &

gnuplot -e "set key autotitle columnhead; plot for [col=2:3] 'rewards_per_period.csv' using 1:col with points; pause mouse close; " &
