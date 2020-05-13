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

import multiprocessing
import logging
from Queue import Empty
from tqdm import tqdm

from pycbc.filter import sigma, matched_filter
from pycbc.types.timeseries import TimeSeries
from pycbc.waveform import get_td_waveform, td_approximants

# -----------------------------------------------------------------------------
# Generate SNR function
# -----------------------------------------------------------------------------

LOGGER = multiprocessing.get_logger()

def running_consumers(consumers):
    count = 0
    for consumer in consumers:
        if consumer.is_alive():
            count += 1

    return count

class ConsumerGenerate(multiprocessing.Process):
    def __init__(
            self, task_queue, result_queue
    ):
        multiprocessing.Process.__init__(self)
        self._task_queue = task_queue
        self._result_queue = result_queue

    def run(self):

        proc_name=self.name

        while True:

            next_task=self._task_queue.get()
            if next_task is None:
                # This poison pil means shutdown
                LOGGER.info("{}: Exiting".format(proc_name))
                break

            results=list()

            mass1=next_task["mass1"]
            mass2=next_task["mass2"]
            spin1z=next_task["spin1z"]
            spin2z=next_task["spin2z"]
            ra=next_task["ra"]
            dec=next_task["dec"]
            coa_phase=next_task["coa_phase"]
            inclination=next_task["inclination"]
            polarization=next_task["polarization"]
            injection_snr=next_task["injection_snr"]
            f_low=next_task["f_low"]
            approximant=next_task["approximant"]
            delta_t=next_task["delta_t"]
            index=next_task["index"]
            det_string=next_task["det_string"]
            strain_sample = next_task["strain_sample"]

            # Convert sample to PyCBC time series
            strain_time_series=TimeSeries(strain_sample,
                                          delta_t=delta_t, epoch=0,
                                          dtype=None, copy=True)

            # Convert sample to PyCBC frequency series
            strain_freq_series=strain_time_series.to_frequencyseries()

            # Generate optimal matched filtering template
            template_hp, template_hc=get_td_waveform(
                approximant = approximant,
                mass1 = mass1,
                mass2 = mass2,
                spin1z = spin1z,
                spin2z = spin2z,
                ra = ra,
                dec = dec,
                coa_phase = coa_phase,
                inclination = inclination,
                f_lower = f_low,
                delta_t = delta_t,
            )

            # Convert template to PyCBC frequency series
            template_freq_series_hp=template_hp.to_frequencyseries(
                                    delta_f=strain_freq_series.delta_f)

            # Resize template to work with the sample
            template_freq_series_hp.resize(len(strain_freq_series))

            # Compute SNR time-series from optimal matched filtering template
            snr_series = matched_filter(template_freq_series_hp,
                                        strain_freq_series.astype(complex),
                                        psd=None, low_frequency_cutoff=f_low)

            results.append(
                {
                    "snr_strain": np.array(abs(snr_series)),
                    "mass1": mass1,
                    "mass2": mass2,
                    "spin1z": spin1z,
                    "spin2z": spin2z,
                    "ra": ra,
                    "dec": dec,
                    "coa_phase": coa_phase,
                    "inclination": inclination,
                    "polarization": polarization,
                    "injection_snr": injection_snr,
                    "index": index,
                    "det_string": det_string, # Unsure I need this for when I store files, using it and index to provide store locations later on
                }
            )

            if len(results)>= 1:
                for result in results:
                    self._result_queue.put(result)

        # Add a poison pill
        self._result_queue.put(None)

class BuildFiles(object):
    def __init__(
            self, output_file_path, param_dict, df
    ):
        self._output_file_path = output_file_path
        self._param_dict = param_dict
        self._df = df


    def run(self):

        try:
            h5_file = h5py.File(self._output_file_path, 'w')

            injection_data_group = h5_file.create_group("injection_snr_samples")
            h1_data_group = injection_data_group.create_group("h1_snr")
            l1_data_group = injection_data_group.create_group("l1_snr")
            v1_data_group = injection_data_group.create_group("v1_snr")

            tasks = multiprocessing.Queue()
            results = multiprocessing.Queue()
            num_consumers = max(int(multiprocessing.cpu_count() * 0.8), 1)

            consumers = [
                ConsumerGenerate(
                    tasks,
                    results
                )
                for _ in range(num_consumers)
            ]

            for consumer in consumers:
                consumer.start()

            # Loop over all detectors
            for i, (det_name, det_string) in enumerate([('H1', 'h1_strain'),
                                                        ('L1', 'l1_strain'),
                                                        ('V1', 'v1_strain')]):

                # Loop over all strain samples in each detector
                for j in range(n_samples):

                    # Read in detector strain data for specific sample
                    strain_sample=np.copy(self._df['injection_samples'][det_string][j])

                    # Get injection parameters
                    for param in param_list:
                        param_dict['injections'][param] = self._df['injection_parameters'][param][j]

                    LOGGER.info("Putting injection parameters.")
                    tasks.put(
                        {
                            "mass1": param_dict['injections']['mass1'],
                            "mass2": param_dict['injections']['mass2'],
                            "spin1z": param_dict['injections']['spin1z'],
                            "spin2z": param_dict['injections']['spin2z'],
                            "ra": param_dict['injections']['ra'],
                            "dec": param_dict['injections']['dec'],
                            "coa_phase": param_dict['injections']['coa_phase'],
                            "inclination": param_dict['injections']['inclination'],
                            "polarization": param_dict['injections']['polarization'],
                            "injection_snr": param_dict['injections']['injection_snr'],
                            "f_low": param_dict['injections']['f_lower'],
                            "approximant": param_dict['injections']['approximant'],
                            "delta_t": param_dict['injections']['delta_t'],
                            "index": j,
                            "det_string": det_string,
                            "strain_sample": strain_sample,
                        }
                    )

            # Poison pill for each consumer
            for _ in range(num_consumers):
                    tasks.put(None)

            while running_consumers(consumers) > 0:
                try:
                    LOGGER.info("Getting results.")
                    next_result = results.get(timeout=5)
                except Empty:
                    LOGGER.info("Nothing in the queue.")
                    next_result = None

                if next_result is None:
                    # Poison pill means a consumer shutdown
                    LOGGER.info("Next result is none. Poison pill.")
                    pass
                else:
                    snr_sample = next_result["snr_strain"]
                    injection_params = dict(
                        mass1 = next_result["mass1"],
                        mass2 = next_result["mass2"],
                        spin1z = next_result["spin1z"],
                        spin2z = next_result["spin2z"],
                        ra = next_result["ra"],
                        dec = next_result["dec"],
                        coa_phase = next_result["coa_phase"],
                        inclination = next_result["inclination"],
                        polarization = next_result["polarization"],
                        injection_snr = next_result["injection_snr"],
                        index = next_result["index"],
                        det_string = next_result["det_string"],
                    )



                    if injection_params["det_string"] == "h1_strain":
                        h1_data_group.create_dataset(
                            str(injection_params["index"]),
                            data=snr_sample,
                        )
                        print("Creating dataset for H1")
                    elif injection_params["det_string"] == "l1_strain":
                        l1_data_group.create_dataset(
                            str(injection_params["index"]),
                            data=snr_sample,
                        )
                        print("Creating dataset for L1")
                    elif injection_params["det_string"] == "v1_strain":
                        v1_data_group.create_dataset(
                            str(injection_params["index"]),
                            data=snr_sample,
                        )
                        print("Creating dataset for V1")

        finally:
            LOGGER.info("Closing file.")
            h5_file.close()

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

    multiprocessing.log_to_stderr()
    logger = multiprocessing.get_logger()
    logger.setLevel(logging.DEBUG)

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
        print('Invalid waveform approximant. Please input'
              'a valid PyCBC time-series approximant.')
        quit()
    
    # -------------------------------------------------------------------------
    # Read in the sample file
    # -------------------------------------------------------------------------

    print('Reading in HDF file...', end=' ')

    #with h5py.File(hdf_file_path, 'r') as f:
    #    df = f
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
    param_dict=dict(injections=dict(mass1=[],mass2=[],spin1z=[],spin2z=[],ra=[],dec=[],coa_phase=[],
                inclination=[],polarization=[],injection_snr=[],f_lower=f_low,
                approximant=apx,delta_t=delta_t))

    # Save event sample time to new SNR dataset for each injection sample
    snr_samples['injection_snr_samples']['event_time']=np.copy(df['injection_samples']['event_time'])

    # Store number of samples, should be identical for all detectors
    n_samples = len(df['injection_samples']['h1_strain'])

    # -------------------------------------------------------------------------
    # Compute SNR time-series
    # -------------------------------------------------------------------------

    if n_samples == 0:
        print('Done! (n-samples = 0)\n')
        exit()

    if filter_injection_samples:
        
        print('Generating SNR time-series for injection samples...', end=' ')






        build_files = BuildFiles(
            output_file_path=output_file_path,
            param_dict=param_dict,
            df = df
        )
        build_files.run()







        print('Done!')

        # Get file size in MB and print the result
        sample_file_size = os.path.getsize(output_file_path) / 1024**2
        print('Size of resulting HDF file: {:.2f}MB'.format(sample_file_size))
        print('')

    if filter_injection_samples==False:

        print('No SNR time-series generated. Please set'
              'injection-samples to True.')
    
    # -------------------------------------------------------------------------
    # Postliminaries
    # -------------------------------------------------------------------------
    
    # Print the total run time
    print('Total runtime: {:.1f} seconds!'.format(time.time() - script_start))
    print('')
    exit()

