#!/usr/bin/env bash
# Fetch the public datasets for the demos. All open-access.
set -e
mkdir -p data
echo "MRI  : ds000102 T1w (OpenNeuro) — https://openneuro.org/datasets/ds000102"
echo "LIGO : Gravity Spy O1 + GWOSC open data (fetched by demos/ligo_glitches.py via gwpy)"
echo "BATT : NASA PCoE Battery Data Set — https://www.nasa.gov/content/prognostics-center-of-excellence-data-set-repository"
echo "See each demo header for exact file placement under ./data/"
