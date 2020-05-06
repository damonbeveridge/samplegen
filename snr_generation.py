"""
Read in the generated injection samples to generate the
optimal matched filtering SNR time-series.
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

from __future__ import print_function

import argparse
import numpy as np
import os
import sys
import time
import h5py

from itertools import count
from multiprocessing import Process, Queue
from tqdm import tqdm
from six import iteritems

from utils.configfiles import read_ini_config, read_json_config
from utils.hdffiles import NoiseTimeline
from utils.samplefiles import SampleFile
from utils.waveforms import WaveformParameterGenerator

from lal import LIGOTimeGPS
from pycbc.psd import interpolate
from pycbc.psd.analytical import aLIGOZeroDetHighPower
from pycbc.noise import noise_from_psd
from pycbc.filter import sigma, matched_filter
from pycbc.types.timeseries import TimeSeries
from pycbc.waveform import get_td_waveform, td_approximants

# -----------------------------------------------------------------------------
# Main Code
# -----------------------------------------------------------------------------

if __name__ == '__main__':
    
    # -----------------------------------------------------------------------------
    # Preliminaries
    # -----------------------------------------------------------------------------
    
    # Disable output buffering ('flush' option is not available for Python 2)
    sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 0)
    
    # Start the stopwatch
    script_start = time.time()

    print('')
    print('GENERATE A GW SAMPLE SNR TIME-SERIES')
    print('')
    
    # -----------------------------------------------------------------------------
    # Parse the command line arguments
    # -----------------------------------------------------------------------------
    
    # Set up the parser and add arguments
    parser = argparse.ArgumentParser(description='Generate a GW data sample.')
    
    # Add arguments (and set default values where applicable)
    parser.add_argument('--input-file-path',
                        help='Name of the HDF file which contains'
                             'the generated samples.'
                             'Default: ./output/default.hdf',
                        default='./output/default.hdf')
    parser.add_argument('--output-file-path',
                        help='Name of the HDF file to write the'
                             'output snr time-series to.'
                             'Default: ./output/snr_series.hdf',
                        default='./output/snr_series.hdf')
    parser.add_argument('--delta-t',
                        help='delta-t value of input hdf file'
                             'generated samples.'
                             'Default: 1.0/2048.',
                        default=1.0/2048)
    parser.add_argument('--low-freq-cutoff',
                        help='Low frequency cutoff for the'
                             'matched filtering process.'
                             'Default: 30 Hz.',
                        default=30)
    parser.add_argument('--apx',
                        help='Name of the time-series approximant'
                             'for the matched filtering template.'
                             'Default: SEOBNRv4',
                        default='SEOBNRv4')
    parser.add_argument('--injection-samples',
                        help='Boolean expression for whether to'
                             'calculate SNRs of injection signals.'
                             'Default: True',
                        default=True)

    # Parse the arguments that were passed when calling this script
    print('Parsing command line arguments...', end=' ')
    arguments = vars(parser.parse_args())
    print('Done!')
    
    # Set up shortcut for the command line arguments
    hdf_file_path=str(arguments['input_file_path'])
    output_file_path=str(arguments['output_file_path'])
    delta_t=float(arguments['delta_t'])
    f_low=int(arguments['low_freq_cutoff'])
    apx=str(arguments['apx'])
    filter_injection_samples=bool(arguments['injection_samples'])
    
    if apx not in td_approximants():
        print("Invalid waveform approximant. Please input"
              "a valid PyCBC time-series approximant.")
        quit()
    
    # -------------------------------------------------------------------------
    # Read in the sample file
    # -------------------------------------------------------------------------

    print('Reading in HDF file...', end=' ')

    df = h5py.File(hdf_file_path, 'r')

    print('Done!')
    
    # -------------------------------------------------------------------------
    # Create dataframe column to store SNR time-series
    # -------------------------------------------------------------------------
    
    # Keep track of all the SNRs (and parameters) we have generated
    snr_samples = dict(injection_snr_samples=dict(h1_strain=dict(),l1_strain=dict(),v1_strain=dict()))

    injection_parameters = dict(mass1=dict(),mass2=dict(),spin1z=dict(),spin2z=dict(),
                 ra=dict(),dec=dict(),coa_phase=dict(),inclination=dict(),
                 polarization=dict(),injection_snr=dict())

    # Initialise list of parameters we want to save in snr_series.hdf file
    param_list = ['mass1','mass2','spin1z','spin2z','ra','dec','coa_phase','inclination','polarization','injection_snr']

    # Initialise list of all parameters required for generating template waveforms
    param_dict = dict(injections=dict(mass1=[],mass2=[],spin1z=[],spin2z=[],ra=[],dec=[],coa_phase=[],
        inclination=[],polarization=[],injection_snr=[],f_lower=f_low,
        approximant=apx,delta_t=delta_t),
        noise=dict(mass1=[],mass2=[],spin1z=[],spin2z=[],ra=[],dec=[],coa_phase=[],
        inclination=[],polarization=[],injection_snr=[],f_lower=f_low,
        approximant=apx,delta_t=delta_t))
    
    # Save event sample time to new SNR dataset for each injection sample
    snr_samples['injection_snr_samples']['event_time']=np.array(
                        df['injection_samples']['event_time'])

    # Store number of samples, should be identical for all detectors
    n_samples = len(df['injection_samples']['h1_strain'])
    
    # -------------------------------------------------------------------------
    # Compute SNR time-series
    # -------------------------------------------------------------------------

    if filter_injection_samples:
        
        print('Generating SNR time-series for injection samples...', end=' ')

        # Loop over all detectors
        for i, (det_name, det_string) in enumerate([('H1', 'h1_strain'),
                                                    ('L1', 'l1_strain'),
                                                    ('V1', 'v1_strain')]):

            # Use a tqdm context manager for the progress bar
            tqdm_args = dict(total=n_samples, ncols=80, unit='sample', desc=det_name)
            with tqdm(**tqdm_args) as progressbar:

                # Loop over all strain samples in each detector
                for i in range(n_samples):

                    # Read in detector strain data for specific sample
                    strain_sample=np.array(df['injection_samples'][det_string][i])

                    # Get injection parameters
                    for param in param_list:
                        param_dict['injections'][param] = df['injection_parameters'][param][i]

                    # Convert sample to PyCBC time series
                    strain_time_series=TimeSeries(strain_sample, delta_t=delta_t,
                                                  epoch=0, dtype=None, copy=True)
                    # Convert sample to PyCBC frequency series 
                    strain_freq_series=strain_time_series.to_frequencyseries()

                    # Generate optimal matched filtering template
                    template_hp, template_hc=get_td_waveform(**param_dict['injections'])

                    # Convert template to PyCBC frequency series
                    template_freq_series_hp=template_hp.to_frequencyseries(
                                                    delta_f=strain_freq_series.delta_f)

                    # Resize template to work with the sample
                    template_freq_series_hp.resize(len(strain_freq_series))

                    # Compute SNR time-series from optimal matched filtering template
                    snr_series = matched_filter(template_freq_series_hp,
                                                strain_freq_series.astype(complex),
                                                psd=None, low_frequency_cutoff=f_low)

                    # Save generated SNR time-series 
                    snr_samples['injection_snr_samples'][det_string][i]=np.array(abs(snr_series))

                    # Save parameters used for generating template
                    for param in injection_parameters.keys():
                        injection_parameters[param][i]= param_dict['injections'][param]

                    # Update the progress bar based on the number of results
                    progressbar.update(i+1 - progressbar.n)

    print('Done!')

    if filter_injection_samples==False:

        print('No SNR time-series generated. Please set'
              'injection-samples to True.')
    
     
    # -------------------------------------------------------------------------
    # Write dataframe to output hdf file
    # -------------------------------------------------------------------------
    
    print('Saving the results to HDF file ...', end=' ')
    
    #if not os.path.exists(output_dir):
    #    os.mkdir(output_dir)

    # Create the specified output file and generate the datastructure
    snr_file = h5py.File(output_file_path,'w')
    grp1 = snr_file.create_group('injection_snr_samples')
    subgrp1 = grp1.create_group('h1_snr')
    subgrp2 = grp1.create_group('l1_snr')
    subgrp3 = grp1.create_group('v1_snr')

    grp2 = snr_file.create_group('injection_parameters')
    subgrp4 = grp2.create_group('mass1')
    subgrp5 = grp2.create_group('mass2')
    subgrp6 = grp2.create_group('spin1z')
    subgrp7 = grp2.create_group('spin2z')
    subgrp8 = grp2.create_group('ra')
    subgrp9 = grp2.create_group('dec')
    subgrp10 = grp2.create_group('coa_phase')
    subgrp11 = grp2.create_group('inclination')
    subgrp12 = grp2.create_group('polarization')
    subgrp13 = grp2.create_group('injection_snr')

    for i in range(len(snr_samples['injection_snr_samples']['h1_strain'])):
        subgrp1.create_dataset(str(i),data=snr_samples['injection_snr_samples']['h1_strain'][i])
        subgrp2.create_dataset(str(i),data=snr_samples['injection_snr_samples']['l1_strain'][i])
        subgrp3.create_dataset(str(i),data=snr_samples['injection_snr_samples']['v1_strain'][i])

        subgrp4.create_dataset(str(i),data=
                               injection_parameters['mass1'][i])
        subgrp5.create_dataset(str(i),data=
                               injection_parameters['mass2'][i])
        subgrp6.create_dataset(str(i),data=
                               injection_parameters['spin1z'][i])
        subgrp7.create_dataset(str(i),data=
                               injection_parameters['spin2z'][i])
        subgrp8.create_dataset(str(i),data=
                               injection_parameters['ra'][i])
        subgrp9.create_dataset(str(i),data=
                               injection_parameters['dec'][i])
        subgrp10.create_dataset(str(i),data=
                               injection_parameters['coa_phase'][i])
        subgrp11.create_dataset(str(i),data=
                               injection_parameters['inclination'][i])
        subgrp12.create_dataset(str(i),data=
                               injection_parameters['polarization'][i])
        subgrp13.create_dataset(str(i),data=
                               injection_parameters['injection_snr'][i])
   
    print('Done!')
    
    # Get file size in MB and print the result
    sample_file_size = os.path.getsize(output_file_path) / 1024**2
    print('Size of resulting HDF file: {:.2f}MB'.format(sample_file_size))
    print('')
    
    # -------------------------------------------------------------------------
    # Postliminaries
    # -------------------------------------------------------------------------
    
    # Print the total run time
    print('Total runtime: {:.1f} seconds!'.format(time.time() - script_start))
    print('')
