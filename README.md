# Sample Generation
Gravitational Wave Injection, Noise and SNR Series Sample Generation

## Generating GW Samples
Use the 'generate_sample.py' file to generate the injection/pure noise samples. The parameters of the outputs are determined by the ranges and values that are set in both files of the 'config_files' folder. Signals are generated for the H1, L1 and V1 detectors, with detector response/delays built-in.

## Generating SNR Time-Series Samples
The script 'snr_generation.py' reads in the output .hdf file from the above sample generation process (currently only working for samples containing injections and using the optimal matched filter). It outputs a second hdf file containing the injected signal parameters and the SNR time-series data for each detector.
