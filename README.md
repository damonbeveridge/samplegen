# Sample Generation
This repository provides an additional features to the repository by Timothy Gebhard for [generating realistic synthetic gravitational-wave data](https://www.github.com/timothygebhard/ggwd/). When generating large data sets of gravitational-wave samples, an alternative script for saving data in "chunks" during the sample generation process is included. An SNR series generation process has been included to produce SNR series of both injection and pure noise samples using optimal matched filters and a template bank of filters. Additionally, the original codebase has been upgraded to work for 3 detector systems, including H1, L1 and V1, with all of the detector responses and delays built in.

The scripts in this repository are built on the basis of the [PyCBC software package](https://www.pycbc.org/), with the intention of providing an easy to use method for generating synthetic gravitational-wave samples in real and synthetic gaussian detector noise, and for generating SNR series for these samples using optimal filters and/or a bank of templates.

## Using This Repository

This project is not a package for Python, it is simply a collection of scripts that can be run from the command line. Ensure that your Python environment satisfies all the package requirements listed in `requirements.txt`.

## Generating Gravitational-Wave Samples
This is only a quickstart guide, for a more detailed walkthrough and full documentation (except chunked and SNR generation), go to the [original sample generation repository by Timothy Gebhard](https://www.github.com/timothygebhard/ggwd/).

In order to generate gravitational-wave samples you can just run either the following:
```python generate_sample.py```
```python chunked_generate.py```

The first Python script is the original script from the [ggwd repository](https://www.github.com/timothygebhard/ggwd/) and the second was created to allow for the generation of very large datafiles that would crash if using the previous file (it does this by saving to the file every 50,000 samples).

Customising the sample generation process in terms of the number of injection or pure noise samples and more can be done by editing the `./config_files/default.json` file. Adjustments to the waveform/sample parameters such as sampling rate, sample length and merger parameter ranges can be made in the `./config_files/waveform_params.ini` file.

## Generating SNR Time-Series Samples


### Generating Filter Banks
The script 'snr_generation.py' reads in the output .hdf file from the above sample generation process (currently only working for samples containing injections and using the optimal matched filter). It outputs a second hdf file containing the injected signal parameters and the SNR time-series data for each detector.
