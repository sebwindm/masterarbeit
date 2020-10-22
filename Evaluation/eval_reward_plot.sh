#!/bin/bash

gnuplot -e "set key autotitle columnhead bottom right;
set title 'Rewards per period'; set ylabel 'Normalized reward';
set xlabel 'Period'; plot for [col=2:2] 'env_rewards_per_period.csv' using 1:col with points; pause mouse close; " &
