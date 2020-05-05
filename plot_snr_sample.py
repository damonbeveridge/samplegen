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

import matplotlib.pyplot as plt  # noqa


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
    parser.add_argument('--hdf-file-path',
                        help='Path to the HDF sample file (generated with '
                             'matched_filter.py) to be used. '
                             'Default: ./output/snr_series.hdf.',
                        default='./output/snr_series.hdf')
    parser.add_argument('--sample-id',
                        help='ID of the sample to be viewed (an integer '
                             'between 0 and n_injection_samples).'
                             'Default: 0.',
                        default=0)
    parser.add_argument('--seconds-before',
                        help='Seconds to plot before the event_time. '
                             'Default: 5.5.',
                        default=0.20)
    parser.add_argument('--seconds-after',
                        help='Seconds to plot after the event_time. '
                             'Default: 2.5.',
                        default=0.05)
    parser.add_argument('--plot-path',
                        help='Where to save the plot of the sample. '
                             'Default: ./snr_sample.pdf.',
                        default='snr_sample.pdf')

    # Parse the arguments that were passed when calling this script
    print('Parsing command line arguments...', end=' ')
    arguments = vars(parser.parse_args())
    print('Done!')

    # Set up shortcuts for the command line arguments
    hdf_file_path = str(arguments['hdf_file_path'])
    sample_id = int(arguments['sample_id'])
    seconds_before = float(arguments['seconds_before'])
    seconds_after = float(arguments['seconds_after'])
    plot_path = str(arguments['plot_path'])

    # -------------------------------------------------------------------------
    # Read in the sample file
    # -------------------------------------------------------------------------

    print('Reading in HDF file...', end=' ')

    df = h5py.File(hdf_file_path,'r')
    data = df['injection_snr_samples']
    parameters = df['injection_parameters']

    print('Done!')

    # -------------------------------------------------------------------------
    # Plot the desired sample
    # -------------------------------------------------------------------------

    print('Plotting sample...', end=' ')
    
    # Select the sample (i.e., the row from the data frame of samples)
    try:
        sample = data['h1_snr'][str(sample_id)]
    except KeyError:
        raise KeyError('Given sample_id is too big! Maximum value = {}'.
                       format(len(np.array(data['h1_snr'])) - 1))

    # Create a grid on which the sample can be plotted so that the
    # event_time is at position 0
#    grid = np.linspace(0 - seconds_before_event, 0 + seconds_after_event,
#                       int(target_sampling_rate * sample_length))

    # Create subplots for H1, L1 and V1
    fig, axes1 = plt.subplots(nrows=3)

    # Plot the strains for H1 and L1
    for i, (det_name, det_string) in enumerate([('H1', 'h1_snr'),
                                                ('L1', 'l1_snr'),
                                                ('V1', 'v1_snr')]):

#        axes1[i].plot(grid, sample[det_string], color='C0')
        axes1[i].plot(data[det_string][str(sample_id)], color='C0')
#        axes1[i].set_xlim(-seconds_before, seconds_after)
#        axes1[i].set_ylim(-150, 150)
        axes1[i].tick_params('y', colors='C0', labelsize=8)
        axes1[i].set_ylabel('{} SNR'
                            .format(det_name), color='C0', fontsize=8)

    # Add the injection parameters to the title
    mass1=np.array(parameters['mass1'][str(sample_id)])
    mass2=np.array(parameters['mass2'][str(sample_id)])
    spin1z=np.array(parameters['spin1z'][str(sample_id)])
    spin2z=np.array(parameters['spin2z'][str(sample_id)])
    ra=np.array(parameters['ra'][str(sample_id)])
    dec=np.array(parameters['dec'][str(sample_id)])
    coa_phase=np.array(parameters['coa_phase'][str(sample_id)])
    inclination=np.array(parameters['inclination'][str(sample_id)])
    polarization=np.array(parameters['polarization'][str(sample_id)])
    injection_snr=np.array(parameters['injection_snr'][str(sample_id)])

    string = str('mass1 = {:.2f}, mass2 = {:.2f}, spin1z = {:.2f}, spin2z = {:.2f}, ra = {:.2f}, dec = {:.2f}, coa_phase = {:.2f}, inclination = {:.2f}, polarization = {:.2f}, injection_snr = {:.2f}, '.format(mass1,mass2,spin1z,spin2z,ra,dec,coa_phase,inclination,polarization,injection_snr))

    plt.figtext(0.5, 0.9, 'Injection Parameters:\n' + string, fontsize=8,ha='center')

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

    # -------------------------------------------------------------------------
    # Postliminaries
    # -------------------------------------------------------------------------

    # Print the total run time
    print('')
    print('Total runtime: {:.1f} seconds!'
          .format(time.time() - script_start_time))
    print('')
