# Sample Generation
This repository provides an additional SNR series generation process to the repository by Timothy Gebhard for [generating realistic synthetic gravitational-wave data](https://www.github.com/timothygebhard/ggwd/). When generating large data sets of gravitational-wave samples, an alternative script for saving data in "chunks" during the sample generation process is included.

The scripts in this repository are built on the basis of the [PyCBC software package](https://www.pycbc.org/), with the intention of providing an easy to use method for generating synthetic gravitational-wave samples in real and synthetic gaussian detector noise, and for generating SNR series for these samples using optimal filters and/or a bank of templates.

## Generating Gravitational-Wave Samples

This is only a quickstart guide, for a more detailed walkthrough and full documentation (except chunked and SNR generation), go to the [original sample generation repository by Timothy Gebhard](https://www.github.com/timothygebhard/ggwd/).

Use the 'generate_sample.py' file to generate the injection/pure noise samples. The parameters of the outputs are determined by the ranges and values that are set in both files of the 'config_files' folder. Signals are generated for the H1, L1 and V1 detectors, with detector response/delays built-in.

## Generating SNR Time-Series Samples
The script 'snr_generation.py' reads in the output .hdf file from the above sample generation process (currently only working for samples containing injections and using the optimal matched filter). It outputs a second hdf file containing the injected signal parameters and the SNR time-series data for each detector.
