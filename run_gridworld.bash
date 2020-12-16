#!/bin/bash

for t in 50 100 500
do
    for e in 50 150 500
    do
        for w in 0.0 0.3 0.5
        do
            python mdp_irl.py --algo deep_maxnet --mdp objectworld --n_trajectories $t  --epochs $e --wind $w
            python mdp_irl.py --algo maxnet --mdp objectworld --n_trajectories $t  --epochs $e --wind $w
        done
    done
done

