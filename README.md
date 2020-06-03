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

The first steps in the process of generating SNR series data is to generate the samples (above) and filter templates that are going to be used.

### Generating Filter Banks

The filter bank generation file, `generate_filter_templates.py`, works much the same as the `generate_sample.py` script. In order to generate the filter bank `.hdf` file, run the following:

```python generate_filter_templates.py```

This script performs the same function as the script that generates the injection signals for the general sample generation process. This means that the two config files mentioned above will also be used in the same manner for generating the filter banks.

In the `default.json` config file, the random seed for generating waveform parameters is given its entry. This is done so that the filters in the filter bank are expected not to match the exact parameters of the injections in the generated strain samples. We do this because in the SNR series process, we generate the optimal matched filter SNR output separately to the filter bank SNR outputs such that they can be operated independently.

### Computing SNR Series

In order to compute the SNR time series for injection samples and/or pure noise samples, they must be previously generated with `generate_sample.py` or `chunked_generate.py`. If you want more than just the optimal matched filtering SNR for injection signals, you need to have generated a bank of templates using `generate_filter_templates.py`. Once these are done, you can generate the SNR time series for the samples and templates produced in the files generated with the previously mentioned processes by running the following:

```python generate_snr_series.py```

Since the multiprocessing method was created with large datasets in mind, it prints out multiple status updates for each process to the command line such that the person running the program can ensure that it is still processing. Note that at the end of the task queue, there is a pause for up to 15 seconds where it is waiting in case there are more tasks to complete.

## Generated Data Structures

<details>
<summary>Injection and Noise Gravitational Wave Samples</summary>
<br>
  
![alt text](https://github.com/damonbeveridge/samplegen/blob/master/data_structures/generate_sample.JPG "Logo Title Text 1")
![alt text](https://github.com/damonbeveridge/samplegen/blob/master/data_structures/default%20-%20injection_parameters.JPG "Injection Parameters")
![alt text](https://github.com/damonbeveridge/samplegen/blob/master/data_structures/default%20-%20injection_samples.JPG "Injection Samples")
![alt text](https://github.com/damonbeveridge/samplegen/blob/master/data_structures/default%20-%20noise_samples.JPG "Pure Noise Samples")
</details>


<details>
<summary>Matched Filtering Templates</summary>
<br>
  
![alt text](https://github.com/damonbeveridge/samplegen/blob/master/data_structures/generate_filter_templates.JPG "generate_filter_templates.py")
![alt text](https://github.com/damonbeveridge/samplegen/blob/master/data_structures/default_templates%20-%20template_samples.JPG "Template Samples")
![alt text](https://github.com/damonbeveridge/samplegen/blob/master/data_structures/default_templates%20-%20template_parameters.JPG "Template Parameters")
![alt text](https://github.com/damonbeveridge/samplegen/blob/master/data_structures/default_templates%20-%20sample_times.JPG "Template Sample Times")
</details>


<details>
<summary>SNR Time Series Samples</summary>
<br>
  
![alt text](https://github.com/damonbeveridge/samplegen/blob/master/data_structures/generate_snr_series.JPG "generate_snr_series.py")
![alt text](https://github.com/damonbeveridge/samplegen/blob/master/data_structures/default_snrs%20-%20omf_injection_snr_samples.JPG "Injection Sample Optimal SNR Series")
![alt text](https://github.com/damonbeveridge/samplegen/blob/master/data_structures/default_snrs%20-%20template_snr_samples.JPG "Noise and Injection Template SNR Series")
</details>
