"""
Read in configuration files and generate synthetic GW signals to be
used as matched filtering templates.
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

from multiprocessing import Process, Queue
from tqdm import tqdm
from pycbc.waveform import get_td_waveform, td_approximants
from pycbc.types.timeseries import TimeSeries

from utils.configfiles import read_ini_config, read_json_config
from utils.waveforms import WaveformParameterGenerator


# -----------------------------------------------------------------------------
# FUNCTION DEFINITIONS
# -----------------------------------------------------------------------------

def generate_waveform(static_arguments,
                      waveform_params):

    if static_arguments["approximant"] not in td_approximants():
        print("Invalid waveform approximant. Please input"
              "a valid PyCBC time-series approximant.")
        quit()

    sample_length = int(static_arguments["sample_length"] * static_arguments["target_sampling_rate"])

    # Collect all the required parameters for the simulation from the given
    # static and variable parameters
    simulation_parameters = dict(approximant=static_arguments['approximant'],
                                 coa_phase=waveform_params['coa_phase'],
                                 delta_f=static_arguments['delta_f'],
                                 delta_t=static_arguments['delta_t'],
                                 distance=static_arguments['distance'],
                                 f_lower=static_arguments['f_lower'],
                                 inclination=waveform_params['inclination'],
                                 mass1=waveform_params['mass1'],
                                 mass2=waveform_params['mass2'],
                                 spin1z=waveform_params['spin1z'],
                                 spin2z=waveform_params['spin2z'],
                                 ra=waveform_params['ra'],
                                 dec=waveform_params['dec'])

    # Perform the actual simulation with the given parameters
    h_plus, h_cross = get_td_waveform(**simulation_parameters)

    return h_plus, simulation_parameters, h_plus.sample_times



def queue_worker(arguments, results_queue):
    """
    Helper function to generate a single GW sample in a dedicated process.

    Args:
        arguments (dict): Dictionary containing the arguments that are
            passed to generate_sample().
        results_queue (Queue): The queue to which the results of this
            worker / process are passed.
    """

    # Try to generate a sample using the given arguments and store the result
    # in the given result_queue (which is shared across all worker processes).
    try:
        result = generate_waveform(**arguments)
        results_queue.put(result)
        sys.exit(0)

    # For some arguments, LALSuite crashes during the sample generation.
    # In this case, terminate with a non-zero exit code to make sure a new
    # set of argument is added to the main arguments_queue
    except RuntimeError:
        sys.exit('Runtime Error')

# -----------------------------------------------------------------------------
# MAIN CODE
# -----------------------------------------------------------------------------

if __name__ == '__main__':

    # -------------------------------------------------------------------------
    # Preliminaries
    # -------------------------------------------------------------------------

    # Disable output buffering ('flush' option is not available for Python 2)
    sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 0)

    # Start the stopwatch
    script_start = time.time()

    print('')
    print('GENERATE A GW DATA SAMPLE FILE')
    print('')

    # -------------------------------------------------------------------------
    # Parse the command line arguments
    # -------------------------------------------------------------------------

    # Set up the parser and add arguments
    parser = argparse.ArgumentParser(description='Generate a GW data sample.')
    parser.add_argument('--config-file',
                        help='Name of the JSON configuration file which '
                             'controls the sample generation process.',
                        default='default.json')

    # Parse the arguments that were passed when calling this script
    print('Parsing command line arguments...', end=' ')
    command_line_arguments = vars(parser.parse_args())
    print('Done!')

    # -------------------------------------------------------------------------
    # Read in JSON config file specifying the sample generation process
    # -------------------------------------------------------------------------

    # Build the full path to the config file
    json_config_name = command_line_arguments['config_file']
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

    # -------------------------------------------------------------------------
    # Shortcuts and random seed
    # -------------------------------------------------------------------------

    # Set the random seed for this script
    np.random.seed(config['random_seed'])

    # Define some useful shortcuts
    random_seed = config['random_seed']
    max_runtime = config['max_runtime']
    bkg_data_dir = config['background_data_directory']

    # -------------------------------------------------------------------------
    # Construct a generator for sampling waveform parameters
    # -------------------------------------------------------------------------

    # Initialize a waveform parameter generator that can sample injection
    # parameters from the distributions specified in the config file
    waveform_parameter_generator = \
        WaveformParameterGenerator(config_file=ini_config_path,
                                   random_seed=random_seed)

    # Wrap it in a generator expression so that we can we can easily sample
    # from it by calling next(waveform_parameters)
    waveform_parameters = \
        (waveform_parameter_generator.draw() for _ in iter(int, 1))

    # -------------------------------------------------------------------------
    # Define a convenience function to generate arguments for the simulation
    # -------------------------------------------------------------------------

    def generate_arguments():

        # Sample waveform parameters
        waveform_params = next(waveform_parameters)

        # Return all necessary arguments as a dictionary
        return dict(static_arguments=static_arguments,
                    waveform_params=waveform_params)

    # -------------------------------------------------------------------------
    # Finally: Create our samples!
    # -------------------------------------------------------------------------

    # Keep track of all the samples (and parameters) we have generated
    samples = dict(template_samples=[])
    injection_parameters = dict(template_samples=[])
    sample_times = dict(template_samples=[])

    print('Generating template samples...')
    n_samples = config['n_template_samples']
    arguments_generator = \
        (generate_arguments() for _ in iter(int, 1))

    sample_type = "template_samples"

    # ---------------------------------------------------------------------
    # If we do not need to generate any samples, skip ahead:
    # ---------------------------------------------------------------------

    if n_samples == 0:
        print('Done! (n_samples=0)\n')
        exit()

    # ---------------------------------------------------------------------
    # Initialize queues for the simulation arguments and the results
    # ---------------------------------------------------------------------

    # Initialize a Queue and fill it with as many arguments as we
    # want to generate samples
    arguments_queue = Queue()
    for i in range(n_samples):
        arguments_queue.put(next(arguments_generator))

    # Initialize a Queue and a list to store the generated samples
    results_queue = Queue()
    results_list = []

    # ---------------------------------------------------------------------
    # Use process-based multiprocessing to generate samples in parallel
    # ---------------------------------------------------------------------

    # Use a tqdm context manager for the progress bar
    tqdm_args = dict(total=n_samples, ncols=80, unit='sample')
    with tqdm(**tqdm_args) as progressbar:

        # Keep track of all running processes
        list_of_processes = []

        # While we haven't produced as many results as desired, keep going
        while len(results_list) < n_samples:

            # -------------------------------------------------------------
            # Loop over processes to see if anything finished or got stuck
            # -------------------------------------------------------------

            for process_dict in list_of_processes:

                # Get the process object and its current runtime
                process = process_dict['process']
                runtime = time.time() - process_dict['start_time']

                # Check if the process is still running when it should
                # have terminated already (according to max_runtime)
                if process.is_alive() and (runtime > max_runtime):

                    # Kill process that's been running too long
                    process.terminate()
                    process.join()
                    list_of_processes.remove(process_dict)

                    # Add new arguments to queue to replace the failed ones
                    new_arguments = next(arguments_generator)
                    arguments_queue.put(new_arguments)

                # If process has terminated already
                elif not process.is_alive():

                    # If the process failed, add new arguments to queue
                    if process.exitcode != 0:
                        new_arguments = next(arguments_generator)
                        arguments_queue.put(new_arguments)

                    # Remove process from the list of running processes
                    list_of_processes.remove(process_dict)

            # -------------------------------------------------------------
            # Start new processes if necessary
            # -------------------------------------------------------------

            # Start new processes until the arguments_queue is empty, or
            # we have reached the maximum number of processes
            while (arguments_queue.qsize() > 0 and
                    len(list_of_processes) < config['n_processes']):

                # Get arguments from queue and start new process
                arguments = arguments_queue.get()
                p = Process(target=queue_worker,
                            kwargs=dict(arguments=arguments,
                            results_queue=results_queue))

                # Remember this process and its starting time
                process_dict = dict(process=p, start_time=time.time())
                list_of_processes.append(process_dict)

                # Finally, start the process
                p.start()

            # -------------------------------------------------------------
            # Move results from results_queue to results_list
            # -------------------------------------------------------------

            # Without this part, the results_queue blocks the worker
            # processes so that they won't terminate
            while results_queue.qsize() > 0:
                results_list.append(results_queue.get())

            # Update the progress bar based on the number of results
            progressbar.update(len(results_list) - progressbar.n)

            # Sleep for some time before we check the processes again
            time.sleep(0.5)

    # ---------------------------------------------------------------------
    # Process results in the results_list
    # ---------------------------------------------------------------------

    # Separate the samples and the injection parameters
    samples[sample_type], injection_parameters[sample_type], sample_times[sample_type] = \
        zip(*results_list)

    print('Sample generation completed!\n')

    # -------------------------------------------------------------------------
    # Create a SampleFile dict from list of samples and save it as an HDF file
    # -------------------------------------------------------------------------

    print('Saving the results to HDF file ...', end=' ')

    # Initialize the dictionary that we use to create a SampleFile object
    sample_file_dict = dict(template_samples=dict(),
                            sample_times=dict(),
    )

    for i in range(n_samples):
        if samples[sample_type]:
            value = samples[sample_type][i]
            value_times = sample_times[sample_type][i]
        else:
            value = None
            value_times = None
            mass1 = None
        sample_file_dict[sample_type][i] = value
        sample_file_dict["sample_times"][i] = value_times

    # Construct the path for the output HDF file
    output_dir = os.path.join('.', 'output')
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    sample_file_path = os.path.join(output_dir, config['template_output_file_name'])

    # Create the specified output file and generate the datastructure
    template_file = h5py.File(sample_file_path,'w')
    grp1 = template_file.create_group('template_samples')
    grp2 = template_file.create_group('sample_times')
    grp3 = template_file.create_group('template_parameters')
    for i in range(n_samples):
        grp1.create_dataset(str(i),data=sample_file_dict[sample_type][i])
        grp2.create_dataset(str(i),data=sample_file_dict['sample_times'][i])

        parameter_group = grp3.create_group(str(i))
        parameter_group.create_dataset("mass1",data=np.copy(injection_parameters[sample_type][i]["mass1"]))
        parameter_group.create_dataset("mass2",data=np.copy(injection_parameters[sample_type][i]["mass2"]))
        parameter_group.create_dataset("spin1z",data=np.copy(injection_parameters[sample_type][i]["spin1z"]))
        parameter_group.create_dataset("spin2z",data=np.copy(injection_parameters[sample_type][i]["spin2z"]))
        parameter_group.create_dataset("ra",data=np.copy(injection_parameters[sample_type][i]["ra"]))
        parameter_group.create_dataset("dec",data=np.copy(injection_parameters[sample_type][i]["dec"]))
        parameter_group.create_dataset("coa_phase",data=np.copy(injection_parameters[sample_type][i]["coa_phase"]))
        parameter_group.create_dataset("inclination",data=np.copy(injection_parameters[sample_type][i]["inclination"]))
        parameter_group.create_dataset("approximant",data=np.copy(injection_parameters[sample_type][i]["approximant"]))
        parameter_group.create_dataset("f_lower",data=np.copy(injection_parameters[sample_type][i]["f_lower"]))
        parameter_group.create_dataset("delta_t",data=np.copy(injection_parameters[sample_type][i]["delta_t"]))

    print('Done!')

    # Get file size in MB and print the result
    sample_file_size = os.path.getsize(sample_file_path) / 1024**2
    print('Size of resulting HDF file: {:.2f}MB'.format(sample_file_size))
    print('')

    # -------------------------------------------------------------------------
    # Postliminaries
    # -------------------------------------------------------------------------

    # PyCBC always create a copy of the waveform parameters file, which we
    # can delete at the end of the sample generation process
    duplicate_path = os.path.join('.', config['waveform_params_file_name'])
    if os.path.exists(duplicate_path):
        os.remove(duplicate_path)

    # Print the total run time
    print('Total runtime: {:.1f} seconds!'.format(time.time() - script_start))
    print('')