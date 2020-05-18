"""
Read in the generated injection samples to generate the
optimal matched filtering SNR time-series.
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

from __future__ import print_function

import argparse
import os
import sys
import time
import h5py

#import multiprocessing
#import logging

from utils.configfiles import read_ini_config, read_json_config
from utils.snr_processes import InjectionsBuildFiles, FiltersBuildFiles

from pycbc.waveform import td_approximants

# -----------------------------------------------------------------------------
# Main Code
# -----------------------------------------------------------------------------

if __name__ == '__main__':
    
    # -----------------------------------------------------------------------------
    # Preliminaries
    # -----------------------------------------------------------------------------
    
    # Disable output buffering ('flush' option is not available for Python 2)
    sys.stderr = os.fdopen(sys.stderr.fileno(), 'w', 0)

    # Start the stopwatch
    script_start = time.time()

    #multiprocessing.log_to_stderr()
    #logger = multiprocessing.get_logger()
    #logger.setLevel(logging.DEBUG)

    print('')
    print('GENERATE A GW SAMPLE SNR TIME-SERIES')
    print('')
    
    # -----------------------------------------------------------------------------
    # Parse the command line arguments
    # -----------------------------------------------------------------------------
    
    # Set up the parser and add arguments
    parser = argparse.ArgumentParser(description='Generate a GW data sample.')
    
    # Add arguments (and set default values where applicable)
    parser.add_argument('--config-file',
                        help='Name of the JSON configuration file which '
                             'controls the sample generation process.',
                        default='default.json')
    parser.add_argument('--filter-injection-samples',
                        help='Boolean expression for whether to'
                             'calculate SNRs of injection signals.'
                             'Default: True',
                        default=True)
    parser.add_argument('--filter-templates',
                        help='Boolean expression for whether to calculate'
                             'SNRs of all signals using a set of templates.'
                             'Default: True',
                        default=True)

    # Parse the arguments that were passed when calling this script
    print('Parsing command line arguments...', end=' ')
    arguments = vars(parser.parse_args())
    print('Done!')
    
    # Set up shortcut for the command line arguments
    filter_injection_samples=bool(arguments['filter_injection_samples'])
    filter_templates=bool(arguments['filter_templates'])

    # -------------------------------------------------------------------------
    # Read in JSON config file specifying the sample generation process
    # -------------------------------------------------------------------------

    # Build the full path to the config file
    json_config_name = arguments['config_file']
    json_config_path = os.path.join('.', 'config_files', json_config_name)

    # Read the JSON configuration into a dict
    print('Reading and validating in JSON configuration file...', end=' ')
    config = read_json_config(json_config_path)
    print('Done!')

    # -------------------------------------------------------------------------
    # Read in INI config file specifying the static_args and variable_args
    # -------------------------------------------------------------------------

    # Build the full path to the waveform params file
    ini_config_name = config['waveform_params_file_name']
    ini_config_path = os.path.join('.', 'config_files', ini_config_name)

    # Read in the variable_arguments and static_arguments
    print('Reading and validating in INI configuration file...', end=' ')
    variable_arguments, static_arguments = read_ini_config(ini_config_path)
    print('Done!\n')

    # Check output file directory exists
    output_dir = os.path.join('.', 'output')
    if not os.path.exists(output_dir):
        print("Output folder cannot be found. Please create a folder",
              "named 'output' to store data in.")
        quit()
    # Get file names from config file
    input_file_path = os.path.join(output_dir, config['output_file_name'])
    templates_file_path = os.path.join(output_dir, config['template_output_file_name'])
    output_file_path = os.path.join(output_dir, config['snr_output_file_name'])

    # -------------------------------------------------------------------------
    # Read in the sample file
    # -------------------------------------------------------------------------

    print('Reading in samples HDF file...', end=' ')

    df = h5py.File(input_file_path, 'r')

    print('Done!')

    print('Reading in the templates HDF file...', end=' ')

    templates_df = h5py.File(templates_file_path, 'r')

    print('Done!')
    
    # -------------------------------------------------------------------------
    # Create dataframe column to store SNR time-series
    # -------------------------------------------------------------------------

    # Get approximant for generating matched filter templates from config files
    if static_arguments["approximant"] not in td_approximants():
        print("Invalid waveform approximant. Please put a valid time-series"
              "approximant in the waveform params file..")
        quit()
    apx = static_arguments["approximant"]

    # Get f-lower and delta-t from config files
    f_low = static_arguments["f_lower"]
    delta_t = 1.0 / static_arguments["target_sampling_rate"]

    # Keep track of all the SNRs (and parameters) we have generated
    injection_parameters = dict(mass1=dict(),mass2=dict(),spin1z=dict(),spin2z=dict(),
                                        ra=dict(),dec=dict(),coa_phase=dict(),inclination=dict(),
                                        polarization=dict(),injection_snr=dict())

    # Initialise list of all parameters required for generating template waveforms
    param_dict=dict(injections=dict(mass1=[],mass2=[],spin1z=[],spin2z=[],ra=[],dec=[],coa_phase=[],
                                    inclination=[],polarization=[],injection_snr=[],f_lower=f_low,
                                    approximant=apx,delta_t=delta_t))

    # Store number of injection samples, should be identical for all detectors
    n_injection_samples = config['n_injection_samples']

    # Store number of noise samples, should be identical for all detectors
    n_noise_samples = config['n_noise_samples']

    # Store number of templates
    n_templates = config['n_template_samples']

    # -------------------------------------------------------------------------
    # Compute SNR time-series
    # -------------------------------------------------------------------------

    if filter_injection_samples:

        print('Generating OMF SNR time-series for injection samples...')

        if n_injection_samples > 0:
            injections_build_files = InjectionsBuildFiles(
                output_file_path=output_file_path,
                param_dict = param_dict,
                df = df,
                n_samples = n_injection_samples
            )
            injections_build_files.run()

            print('Done!')
        else:
            print('Done! (n-samples = 0)\n')
    else:
        print('No SNR time-series generated for injections.'
              'Please set filter-injection-samples to True.')



    if filter_templates:

        print("Generating SNR time-series for injection and noise samples using a template set...")

        if n_templates == 0:
            print('Done! (n-templates = 0)'
                  'Please generate templates before running.\n')
        elif n_noise_samples > 0:
            filters_build_files = FiltersBuildFiles(
                output_file_path=output_file_path,
                df = df,
                templates_df = templates_df,
                n_noise_samples = n_noise_samples,
                n_injection_samples = n_injection_samples,
                n_templates = n_templates,
                f_low = f_low,
                delta_t = delta_t,
                filter_injection_samples = filter_injection_samples
            )
            filters_build_files.run()

            print('Done!')
        else:
            print('Done! (n-noise-samples = 0)\n')
    else:
        print('No SNR time-series generated for injections.'
              'Please set filter-templates to True.')

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
    exit()