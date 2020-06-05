"""
Plot the results produced by the generate_sample.py script.
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

import matplotlib.pyplot as plt

from utils.configfiles import read_ini_config, read_json_config


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
    script_start_time = time.time()
    print('')
    print('PLOT A GENERATED SNR SAMPLE (WITH AN INJECTION)')
    print('')

    # -------------------------------------------------------------------------
    # Parse the command line arguments
    # -------------------------------------------------------------------------

    # Set up the parser
    parser = argparse.ArgumentParser(description='Plot a generated sample.')

    # Add arguments (and set default values where applicable)
    parser.add_argument('--plot-template-sample',
                        help='Boolean value determining whether to plot '
                             'template or optimal matched filtering SNR.'
                             'Default: False.',
                        default=True)
    parser.add_argument('--plot-injection-sample',
                        help='Boolean value determining whether to plot '
                             'injection sample or noise sample SNR.'
                             'Note: If False, noise sample SNR will be plotted.'
                             'Default: True.',
                        default=True)
    parser.add_argument('--sample-id',
                        help='ID of the sample to be viewed (an integer '
                             'between 0 and n_injection_samples).'
                             'Default: 0.',
                        default=0)
    parser.add_argument('--noise-id',
                        help='ID of the sample to be viewed (an integer '
                             'between 0 and n_noise_samples).'
                             'Default: 0.',
                        default=0)
    parser.add_argument('--template-id',
                        help='ID of the matched filteringtemplate to be viewed (an '
                             'integer between 0 and n_template_samples).'
                             'Note: Not applicable if plot-template-sample is False.'
                             'Default: 0.',
                        default=0)
    parser.add_argument('--seconds-before',
                        help='Seconds to plot before the event_time. '
                             'Default: 0.2.',
                        default=0.20)
    parser.add_argument('--seconds-after',
                        help='Seconds to plot after the event_time. '
                             'Default: 0.05.',
                        default=0.05)
    parser.add_argument('--plot-path',
                        help='Where to save the plot of the sample. '
                             'Default: ./snr_sample.pdf.',
                        default='snr_sample.png')
    parser.add_argument('--config-file',
                        help='Name of the JSON configuration file which '
                             'controls the sample generation process.',
                        default='default.json')

    # Parse the arguments that were passed when calling this script
    print('Parsing command line arguments...', end=' ')
    arguments = vars(parser.parse_args())
    print('Done!')

    # Set up shortcuts for the command line arguments
    plot_template_sample = bool(arguments['plot_template_sample'])
    plot_injection_sample = bool(arguments['plot_injection_sample'])
    sample_id = int(arguments['sample_id'])
    noise_id = int(arguments['noise_id'])
    template_id = int(arguments['template_id'])
    seconds_before = float(arguments['seconds_before'])
    seconds_after = float(arguments['seconds_after'])
    plot_path = str(arguments['plot_path'])

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
    hdf_file_path = os.path.join(output_dir, config['snr_output_file_name'])
    template_file_path = os.path.join(output_dir, config['template_output_file_name'])
    injections_file_path = os.path.join(output_dir, config['output_file_name'])

    # -------------------------------------------------------------------------
    # Read in the sample file
    # -------------------------------------------------------------------------

    print('Reading in SNR samples file...', end=' ')

    df = h5py.File(hdf_file_path,'r')

    print('Done!')

    if plot_template_sample:
        print('Reading in templates file...', end=' ')

        templates = h5py.File(template_file_path, 'r')

        print('Done!\n')

    if plot_injection_sample:
        print('Reading in injections file...', end=' ')

        injections = h5py.File(injections_file_path, 'r')

        print('Done!\n')

    # -------------------------------------------------------------------------
    # Plot the desired sample
    # -------------------------------------------------------------------------

    if plot_template_sample:
        if plot_injection_sample:

            print('Plotting injection sample filtered with a template...', end=' ')

            label = "template" + str(template_id) + ",sample" + str(sample_id)

            # Select the sample (i.e., the row from the data frame of samples)
            try:
                sample = df['template_snr_samples']['injection']['H1'][label]
            except KeyError:
                raise KeyError('Given sample_id or template_id is too big!'
                               'Maximum template_id value = {t}. Maximum sample_id value = {s}'.
                   format(t = len(np.copy(templates['template_samples'])) - 1,
                          s = len(np.copy(injections['injection_samples']['h1_strain']))))

            # Create subplots for H1, L1 and V1
            fig, axes1 = plt.subplots(nrows=3)

            for i, (det_name, det_string) in enumerate([('H1', 'h1_snr'),
                                                        ('L1', 'l1_snr'),
                                                        ('V1', 'v1_snr')]):

                # axes1[i].plot(grid, sample[det_string], color='C0')
                axes1[i].plot(df['template_snr_samples']['injection'][det_name][label], color='C0')
                # axes1[i].set_xlim(-seconds_before, seconds_after)
                # axes1[i].set_ylim(-150, 150)
                axes1[i].tick_params('y', colors='C0', labelsize=8)
                axes1[i].set_ylabel('{} SNR'
                                    .format(det_name), color='C0', fontsize=8)

            # Add the injection parameters to the title
            template_mass1 = np.copy(templates['template_parameters'][str(template_id)]['mass1'])
            template_mass2 = np.copy(templates['template_parameters'][str(template_id)]['mass2'])
            template_spin1z = np.copy(templates['template_parameters'][str(template_id)]['spin1z'])
            template_spin2z = np.copy(templates['template_parameters'][str(template_id)]['spin2z'])
            template_ra = np.copy(templates['template_parameters'][str(template_id)]['ra'])
            template_dec = np.copy(templates['template_parameters'][str(template_id)]['dec'])
            template_coa_phase = np.copy(templates['template_parameters'][str(template_id)]['coa_phase'])
            template_inclination = np.copy(templates['template_parameters'][str(template_id)]['inclination'])

            injection_mass1 = np.copy(injections['injection_parameters']['mass1'][sample_id])
            injection_mass2 = np.copy(injections['injection_parameters']['mass2'][sample_id])
            injection_spin1z = np.copy(injections['injection_parameters']['spin1z'][sample_id])
            injection_spin2z = np.copy(injections['injection_parameters']['spin2z'][sample_id])
            injection_ra = np.copy(injections['injection_parameters']['ra'][sample_id])
            injection_dec = np.copy(injections['injection_parameters']['dec'][sample_id])
            injection_coa_phase = np.copy(injections['injection_parameters']['coa_phase'][sample_id])
            injection_inclination = np.copy(injections['injection_parameters']['inclination'][sample_id])
            injection_polarization = np.copy(injections['injection_parameters']['polarization'][sample_id])
            injection_snr = np.copy(injections['injection_parameters']['injection_snr'][sample_id])

            template_string = str('mass1 = {:.2f}, mass2 = {:.2f}, spin1z = {:.2f}, spin2z = {:.2f}, ra = {:.2f}, dec = {:.2f}, coa_phase = {:.2f}, inclination = {:.2f}'
                                   .format(template_mass1,template_mass2,template_spin1z,template_spin2z,template_ra,template_dec,template_coa_phase,template_inclination))
            injection_string = str('mass1 = {:.2f}, mass2 = {:.2f}, spin1z = {:.2f}, spin2z = {:.2f}, ra = {:.2f}, dec = {:.2f}, coa_phase = {:.2f}, inclination = {:.2f}, polarization = {:.2f}, injection_snr = {:.2f}'
                                   .format(injection_mass1,injection_mass2,injection_spin1z,injection_spin2z,injection_ra,injection_dec,injection_coa_phase,injection_inclination,injection_polarization,injection_snr))

            plt.figtext(0.5, 0.923, 'Injection Parameters:\n' + injection_string, fontsize=8,ha='center')
            plt.figtext(0.5, 0.88, 'Template Parameters:\n' + template_string, fontsize=8,ha='center')

            # Set x-labels
            axes1[0].set_xticklabels([])
            axes1[1].set_xlabel('Time from event time (in seconds)')

            # Adjust the size and spacing of the subplots
            plt.gcf().set_size_inches(12, 6, forward=True)
            plt.tight_layout(rect=[0, 0, 1, 0.9])
            plt.subplots_adjust(wspace=0, hspace=0)

            # Add a title
            plt.suptitle('Sample #{s}, Template #{t}'.format(s=sample_id, t=template_id), y=0.99)

            # Save the plot at the given location
            plt.savefig(plot_path, bbox_inches='tight', pad_inches=0)

            print('Done!')





        else:

            print('Plotting noise sample filtered with a template...', end=' ')

            label = "template" + str(template_id) + ",sample" + str(noise_id)

            # Select the sample (i.e., the row from the data frame of samples)
            try:
                sample = df['template_snr_samples']['noise']['H1'][label]
            except KeyError:
                raise KeyError('Given noise_id or template_id is too big!'
                               'Maximum template_id value = {t}. Maximum sample_id value = {s}'.
                               format(t = len(np.copy(templates['template_samples'])) - 1,
                                      s = len(np.copy(injections['noise_samples']['h1_strain']))))

            # Create subplots for H1, L1 and V1
            fig, axes1 = plt.subplots(nrows=3)

            for i, (det_name, det_string) in enumerate([('H1', 'h1_snr'),
                                                        ('L1', 'l1_snr'),
                                                        ('V1', 'v1_snr')]):

                # axes1[i].plot(grid, sample[det_string], color='C0')
                axes1[i].plot(df['template_snr_samples']['noise'][det_name][label], color='C0')
                # axes1[i].set_xlim(-seconds_before, seconds_after)
                # axes1[i].set_ylim(-150, 150)
                axes1[i].tick_params('y', colors='C0', labelsize=8)
                axes1[i].set_ylabel('{} SNR'
                                    .format(det_name), color='C0', fontsize=8)

            # Add the injection parameters to the title
            template_mass1 = np.copy(templates['template_parameters'][str(template_id)]['mass1'])
            template_mass2 = np.copy(templates['template_parameters'][str(template_id)]['mass2'])
            template_spin1z = np.copy(templates['template_parameters'][str(template_id)]['spin1z'])
            template_spin2z = np.copy(templates['template_parameters'][str(template_id)]['spin2z'])
            template_ra = np.copy(templates['template_parameters'][str(template_id)]['ra'])
            template_dec = np.copy(templates['template_parameters'][str(template_id)]['dec'])
            template_coa_phase = np.copy(templates['template_parameters'][str(template_id)]['coa_phase'])
            template_inclination = np.copy(templates['template_parameters'][str(template_id)]['inclination'])

            template_string = str('mass1 = {:.2f}, mass2 = {:.2f}, spin1z = {:.2f}, spin2z = {:.2f}, ra = {:.2f}, dec = {:.2f}, coa_phase = {:.2f}, inclination = {:.2f}'
                                  .format(template_mass1,template_mass2,template_spin1z,template_spin2z,template_ra,template_dec,template_coa_phase,template_inclination))

            plt.figtext(0.5, 0.9, 'Template Parameters:\n' + template_string, fontsize=8,ha='center')

            # Set x-labels
            axes1[0].set_xticklabels([])
            axes1[1].set_xlabel('Time from event time (in seconds)')

            # Adjust the size and spacing of the subplots
            plt.gcf().set_size_inches(12, 6, forward=True)
            plt.tight_layout(rect=[0, 0, 1, 0.9])
            plt.subplots_adjust(wspace=0, hspace=0)

            # Add a title
            plt.suptitle('Sample #{s}, Template #{t}'.format(s=sample_id, t=template_id), y=0.975)

            # Save the plot at the given location
            plt.savefig(plot_path, bbox_inches='tight', pad_inches=0)

            print('Done!')





    else:
        if plot_injection_sample:

            print('Plotting injection sample with optimal filter...', end=' ')

            # Select the sample (i.e., the row from the data frame of samples)
            try:
                sample = df['omf_injection_snr_samples']['h1_snr']['0']
            except KeyError:
                raise KeyError('Given noise_id or template_id is too big!'
                               'Maximum template_id value = {t}. Maximum sample_id value = {s}'.
                               format(t = len(np.copy(templates['template_samples'])) - 1,
                                      s = len(np.copy(injections['noise_samples']['h1_strain']))))

            # Create subplots for H1, L1 and V1
            fig, axes1 = plt.subplots(nrows=3)

            for i, (det_name, det_string) in enumerate([('H1', 'h1_snr'),
                                                        ('L1', 'l1_snr'),
                                                        ('V1', 'v1_snr')]):

                # axes1[i].plot(grid, sample[det_string], color='C0')
                axes1[i].plot(df['omf_injection_snr_samples'][det_string][str(sample_id)], color='C0')
                # axes1[i].set_xlim(-seconds_before, seconds_after)
                # axes1[i].set_ylim(-150, 150)
                axes1[i].tick_params('y', colors='C0', labelsize=8)
                axes1[i].set_ylabel('{} SNR'
                                    .format(det_name), color='C0', fontsize=8)

            # Add the injection parameters to the title
            injection_mass1 = np.copy(injections['injection_parameters']['mass1'][sample_id])
            injection_mass2 = np.copy(injections['injection_parameters']['mass2'][sample_id])
            injection_spin1z = np.copy(injections['injection_parameters']['spin1z'][sample_id])
            injection_spin2z = np.copy(injections['injection_parameters']['spin2z'][sample_id])
            injection_ra = np.copy(injections['injection_parameters']['ra'][sample_id])
            injection_dec = np.copy(injections['injection_parameters']['dec'][sample_id])
            injection_coa_phase = np.copy(injections['injection_parameters']['coa_phase'][sample_id])
            injection_inclination = np.copy(injections['injection_parameters']['inclination'][sample_id])
            injection_polarization = np.copy(injections['injection_parameters']['polarization'][sample_id])
            injection_snr = np.copy(injections['injection_parameters']['injection_snr'][sample_id])

            injection_string = str('mass1 = {:.2f}, mass2 = {:.2f}, spin1z = {:.2f}, spin2z = {:.2f}, ra = {:.2f}, dec = {:.2f}, coa_phase = {:.2f}, inclination = {:.2f}, polarization = {:.2f}, injection_snr = {:.2f}'
                                   .format(injection_mass1,injection_mass2,injection_spin1z,injection_spin2z,injection_ra,injection_dec,injection_coa_phase,injection_inclination,injection_polarization,injection_snr))

            plt.figtext(0.5, 0.9, 'Injection Parameters:\n' + injection_string, fontsize=8,ha='center')

            # Set x-labels
            axes1[0].set_xticklabels([])
            axes1[1].set_xlabel('Time from event time (in seconds)')

            # Adjust the size and spacing of the subplots
            plt.gcf().set_size_inches(12, 6, forward=True)
            plt.tight_layout(rect=[0, 0, 1, 0.9])
            plt.subplots_adjust(wspace=0, hspace=0)

            # Add a title
            plt.suptitle('Sample #{}'.format(sample_id), y=0.975)

            # Save the plot at the given location
            plt.savefig(plot_path, bbox_inches='tight', pad_inches=0)

            print('Done!')





        else:

            print('No SNR samples exist with no template and no injection.'
                  'Please adjust plot-template-sample and plot-injection-sample.')

    # -------------------------------------------------------------------------
    # Postliminaries
    # -------------------------------------------------------------------------

    # Print the total run time
    print('')
    print('Total runtime: {:.1f} seconds!'
          .format(time.time() - script_start_time))
    print('')
