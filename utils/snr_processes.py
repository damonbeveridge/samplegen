"""
Read in the generated injection samples to generate the
optimal matched filtering SNR time-series.
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

from __future__ import print_function

import numpy as np
import h5py
import random

import multiprocessing
from Queue import Empty

from pycbc.filter import sigma, matched_filter
from pycbc.types.timeseries import TimeSeries
from pycbc.waveform import get_td_waveform

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

class InjectionsConsumerGenerate(multiprocessing.Process):
    def __init__(
            self, task_queue, result_queue
    ):
        multiprocessing.Process.__init__(self)
        self._task_queue = task_queue
        self._result_queue = result_queue

    def run(self):

        proc_name=self.name

        while True:

            # Get next task to be completed from the queue
            next_task=self._task_queue.get()
            if next_task is None:
                # This poison pil means shutdown
                LOGGER.info("{}: Exiting".format(proc_name))
                break

            results=list()

            # Initialise parameters from task queue to generate SNR time-series
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

            print("Generating optimal SNR time series: " + det_string + " - sample" + str(index))

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

            # Time shift the template so that the SNR peak matches the merger time
            template_freq_series_hp = template_freq_series_hp.cyclic_time_shift(template_freq_series_hp.start_time)

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

            # Put the results on the results queue
            if len(results)>= 1:
                for result in results:
                    self._result_queue.put(result)

        # Add a poison pill
        self._result_queue.put(None)

class InjectionsBuildFiles(object):
    def __init__(
            self, output_file_path, param_dict, df, n_samples,
            trim_output, trim_cutoff_low, trim_cutoff_high, trim_cutoff_variation
    ):
        self._output_file_path = output_file_path
        self._param_dict = param_dict
        self._df = df
        self._n_samples = n_samples
        self._trim_output = trim_output
        self._trim_cutoff_low = trim_cutoff_low
        self._trim_cutoff_high = trim_cutoff_high
        self._trim_cutoff_variation = trim_cutoff_variation


    def run(self):

        try:

            # Open output file path to generate file structure
            h5_file = h5py.File(self._output_file_path, 'w')

            # Setup basic hdf file structure for output
            omf_injection_data_group = h5_file.create_group("omf_injection_snr_samples")
            omf_h1_data_group = omf_injection_data_group.create_group("h1_snr")
            omf_l1_data_group = omf_injection_data_group.create_group("l1_snr")
            omf_v1_data_group = omf_injection_data_group.create_group("v1_snr")

            tasks = multiprocessing.Queue()
            results = multiprocessing.Queue()

            # Limit core usage to nearest integer below 80% of CPU cores
            num_consumers = max(int(multiprocessing.cpu_count() * 0.8), 1)

            # Initialise consumers
            consumers = [
                InjectionsConsumerGenerate(
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
                for j in range(self._n_samples):

                    # Read in detector strain data for specific sample
                    strain_sample=np.copy(self._df['injection_samples'][det_string][j])

                    # Initialise list of parameters we want to save in snr_series.hdf file
                    param_list = ['mass1','mass2','spin1z','spin2z','ra','dec',
                                  'coa_phase','inclination','polarization','injection_snr']
                    # Get injection parameters
                    for param in param_list:
                        self._param_dict['injections'][param] = self._df['injection_parameters'][param][j]

                    #LOGGER.info("Putting injection parameters: " + det_name + "-" + str(j) + ".")
                    print("Putting injection parameters: " + det_name + " - sample" + str(j) + ".")
                    tasks.put(
                        {
                            "mass1": self._param_dict['injections']['mass1'],
                            "mass2": self._param_dict['injections']['mass2'],
                            "spin1z": self._param_dict['injections']['spin1z'],
                            "spin2z": self._param_dict['injections']['spin2z'],
                            "ra": self._param_dict['injections']['ra'],
                            "dec": self._param_dict['injections']['dec'],
                            "coa_phase": self._param_dict['injections']['coa_phase'],
                            "inclination": self._param_dict['injections']['inclination'],
                            "polarization": self._param_dict['injections']['polarization'],
                            "injection_snr": self._param_dict['injections']['injection_snr'],
                            "f_low": self._param_dict['injections']['f_lower'],
                            "approximant": self._param_dict['injections']['approximant'],
                            "delta_t": self._param_dict['injections']['delta_t'],
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
                    next_result = results.get(timeout=15)
                except Empty:
                    LOGGER.info("Nothing in the queue.")
                    next_result = None

                # Store results from queue for writing to file
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

                    # Write result SNR data to HDF file in correct groups
                    if self._trim_output:

                        rand = random.randint(-self._trim_cutoff_variation,self._trim_cutoff_variation)
                        low = self._trim_cutoff_low + rand
                        high = self._trim_cutoff_high + rand
                        
                        if injection_params["det_string"] == "h1_strain":
                            print("Creating dataset for H1 - sample" + str(injection_params["index"]))
                            omf_h1_data_group.create_dataset(
                                str(injection_params["index"]),
                                data=snr_sample[low:high],
                            )
                        elif injection_params["det_string"] == "l1_strain":
                            print("Creating dataset for L1 - sample" + str(injection_params["index"]))
                            omf_l1_data_group.create_dataset(
                                str(injection_params["index"]),
                                data=snr_sample[low:high],
                            )
                        else:
                            print("Creating dataset for V1 - sample" + str(injection_params["index"]))
                            omf_v1_data_group.create_dataset(
                                str(injection_params["index"]),
                                data=snr_sample[low:high],
                            )
                    else:
                        if injection_params["det_string"] == "h1_strain":
                            print("Creating dataset for H1 - sample" + str(injection_params["index"]))
                            omf_h1_data_group.create_dataset(
                                str(injection_params["index"]),
                                data=snr_sample,
                            )
                        elif injection_params["det_string"] == "l1_strain":
                            print("Creating dataset for L1 - sample" + str(injection_params["index"]))
                            omf_l1_data_group.create_dataset(
                                str(injection_params["index"]),
                                data=snr_sample,
                            )
                        else:
                            print("Creating dataset for V1 - sample" + str(injection_params["index"]))
                            omf_v1_data_group.create_dataset(
                                str(injection_params["index"]),
                                data=snr_sample,
                            )

        finally:
            LOGGER.info("Closing file.")
            h5_file.close()

        return











class FiltersConsumerGenerate(multiprocessing.Process):
    def __init__(
            self, template_task_queue, template_result_queue
    ):
        multiprocessing.Process.__init__(self)
        self._template_task_queue = template_task_queue
        self._template_result_queue = template_result_queue

    def run(self):

        proc_name=self.name

        while True:

            next_task=self._template_task_queue.get()
            if next_task is None:
                # This poison pil means shutdown
                LOGGER.info("{}: Exiting".format(proc_name))
                break

            template_results=list()

            f_low=next_task["f_low"]
            delta_t=next_task["delta_t"]
            template=next_task["template"]
            sample_index=next_task["sample_index"]
            template_index=next_task["template_index"]
            det_string=next_task["det_string"]
            strain_sample=next_task["strain_sample"]
            sample_type=next_task["sample_type"]
            delta_f=next_task["delta_f"]
            template_start_time=next_task["template_start_time"]

            print("Generating SNR time series: " + det_string + " - sample" + str(sample_index) + ", template" + str(template_index))

            template_time_series = TimeSeries(template,
                                              delta_t=delta_t, epoch=0,
                                              dtype=None, copy=True)
            template_freq_series = template_time_series.to_frequencyseries(delta_f=delta_f)

            strain_sample_time_series = TimeSeries(strain_sample,
                                                   delta_t=delta_t, epoch=0,
                                                   dtype=None, copy=True)
            strain_freq_series = strain_sample_time_series.to_frequencyseries(delta_f=delta_f)

            template_freq_series.resize(len(strain_freq_series))

            # Time shift the template so that the SNR peak matches the merger time
            template_freq_series = template_freq_series.cyclic_time_shift(template_start_time)

            # Compute SNR time-series from optimal matched filtering template
            snr_series = matched_filter(template_freq_series,
                                        strain_freq_series.astype(complex),
                                        psd=None, low_frequency_cutoff=f_low)

            template_results.append(
                {
                    "snr_strain": np.array(abs(snr_series)),
                    "sample_index": sample_index,
                    "template_index": template_index,
                    "det_string": det_string,
                    "sample_type": sample_type,
                }
            )

            if len(template_results)>= 1:
                for result in template_results:
                    self._template_result_queue.put(result)

        # Add a poison pill
        self._template_result_queue.put(None)

class FiltersBuildFiles(object):
    def __init__(
            self, output_file_path, df, templates_df, n_injection_samples, n_noise_samples,
            n_templates, f_low, delta_t, filter_injection_samples, delta_f,
            trim_output, trim_cutoff_low, trim_cutoff_high, trim_cutoff_variation
    ):
        self._output_file_path = output_file_path
        self._df = df
        self._template_df = templates_df
        self._n_injection_samples = n_injection_samples
        self._n_noise_samples = n_noise_samples
        self._n_templates = n_templates
        self._f_low = f_low
        self._delta_t = delta_t
        self._filter_injection_samples = filter_injection_samples
        self._delta_f = delta_f
        self._trim_output = trim_output
        self._trim_cutoff_low = trim_cutoff_low
        self._trim_cutoff_high = trim_cutoff_high
        self._trim_cutoff_variation = trim_cutoff_variation

    def run(self):

        try:
            # Overwriting HDF file because we aren't generating SNR's for injection samples
            if self._filter_injection_samples is False:
                h5_file = h5py.File(self._output_file_path, 'w')
            # Appending HDF file because we have already generated SNR's for injection samples
            else:
                h5_file = h5py.File(self._output_file_path, 'a')

            template_data_group = h5_file.create_group("template_snr_samples")

            injection_data_group = template_data_group.create_group("injection")
            noise_data_group = template_data_group.create_group("noise")

            h1_injection_data_group = injection_data_group.create_group("H1")
            l1_injection_data_group = injection_data_group.create_group("L1")
            v1_injection_data_group = injection_data_group.create_group("V1")
            h1_noise_data_group = noise_data_group.create_group("H1")
            l1_noise_data_group = noise_data_group.create_group("L1")
            v1_noise_data_group = noise_data_group.create_group("V1")

            template_tasks = multiprocessing.Queue()
            template_results = multiprocessing.Queue()
            num_consumers = max(int(multiprocessing.cpu_count() * 0.8), 1)

            consumers = [
                FiltersConsumerGenerate(
                    template_tasks,
                    template_results
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
                for j in range(self._n_injection_samples):

                    # Loop over all templates
                    for k in range(self._n_templates):

                        #LOGGER.info("Putting injection sample and template parameters: " + det_name + "-" + str(j) + ", Template-" + str(k) + ".")
                        print("Putting injection sample and template parameters: " + det_name + " - sample" + str(j) + ", Template-" + str(k) + ".")
                        template_tasks.put(
                            {
                                "f_low": self._f_low,
                                "delta_t": self._delta_t,
                                "template": np.copy(self._template_df['template_samples'][str(k)]),
                                "template_start_time": np.copy(self._template_df['template_parameters'][str(k)]["start_time"]),
                                "sample_index": j,
                                "template_index": k,
                                "det_string": det_string,
                                "strain_sample": np.copy(self._df["injection_samples"][det_string][j]),
                                "sample_type": "injection_samples",
                                "delta_f": self._delta_f,
                            }
                        )

                # Loop over all strain samples in each detector
                for j in range(self._n_noise_samples):

                    # Loop over all templates
                    for k in range(self._n_templates):

                        #LOGGER.info("Putting noise sample and template parameters: " + det_name + "-" + str(j) + ", Template-" + str(k) + ".")
                        print("Putting noise sample and template parameters: " + det_name + " - sample" + str(j) + ", Template-" + str(k) + ".")
                        template_tasks.put(
                            {
                                "f_low": self._f_low,
                                "delta_t": self._delta_t,
                                "template": np.copy(self._template_df['template_samples'][str(k)]),
                                "sample_index": j,
                                "template_index": k,
                                "det_string": det_string,
                                "strain_sample": np.copy(self._df["noise_samples"][det_string][j]),
                                "sample_type": "noise_samples",
                                "delta_f": self._delta_f,
                            }
                        )

            # Poison pill for each consumer
            for _ in range(num_consumers):
                template_tasks.put(None)

            while running_consumers(consumers) > 0:
                try:
                    LOGGER.info("Getting results.")
                    next_result = template_results.get(timeout=15)
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
                        sample_index = next_result["sample_index"],
                        template_index = next_result["template_index"],
                        det_string = next_result["det_string"],
                        sample_type = next_result["sample_type"],
                    )

                    # Write result SNR data to HDF file in correct groups
                    if self._trim_output:

                        rand = random.randint(-self._trim_cutoff_variation,self._trim_cutoff_variation)
                        low = self._trim_cutoff_low + rand
                        high = self._trim_cutoff_high + rand
                        
                        if injection_params['sample_type'] == "injection_samples":
                            if injection_params['det_string'] == "h1_strain":
                                label = "template" + str(injection_params["template_index"]) + \
                                        ",sample" + str(injection_params["sample_index"])
                                print("Creating dataset for H1 - " + label)
                                h1_injection_data_group.create_dataset(label, data=snr_sample[low:high])
                            elif injection_params['det_string'] == "l1_strain":
                                label = "template" + str(injection_params["template_index"]) + \
                                        ",sample" + str(injection_params["sample_index"])
                                print("Creating dataset for L1 - " + label)
                                l1_injection_data_group.create_dataset(label, data=snr_sample[low:high])
                            else:
                                label = "template" + str(injection_params["template_index"]) + \
                                        ",sample" + str(injection_params["sample_index"])
                                print("Creating dataset for V1 - " + label)
                                v1_injection_data_group.create_dataset(label, data=snr_sample[low:high])
                        else:
                            if injection_params['det_string'] == "h1_strain":
                                label = "template" + str(injection_params["template_index"]) + \
                                        ",sample" + str(injection_params["sample_index"])
                                print("Creating dataset for H1 - " + label)
                                h1_noise_data_group.create_dataset(label, data=snr_sample[low:high])
                            elif injection_params['det_string'] == "l1_strain":
                                label = "template" + str(injection_params["template_index"]) + \
                                        ",sample" + str(injection_params["sample_index"])
                                print("Creating dataset for L1 - " + label)
                                l1_noise_data_group.create_dataset(label, data=snr_sample[low:high])
                            else:
                                label = "template" + str(injection_params["template_index"]) + \
                                        ",sample" + str(injection_params["sample_index"])
                                print("Creating dataset for V1 - " + label)
                                v1_noise_data_group.create_dataset(label, data=snr_sample[low:high])
                    else:
                        if injection_params['sample_type'] == "injection_samples":
                            if injection_params['det_string'] == "h1_strain":
                                label = "template" + str(injection_params["template_index"]) + \
                                        ",sample" + str(injection_params["sample_index"])
                                print("Creating dataset for H1 - " + label)
                                h1_injection_data_group.create_dataset(label, data=snr_sample)
                            elif injection_params['det_string'] == "l1_strain":
                                label = "template" + str(injection_params["template_index"]) + \
                                        ",sample" + str(injection_params["sample_index"])
                                print("Creating dataset for L1 - " + label)
                                l1_injection_data_group.create_dataset(label, data=snr_sample)
                            else:
                                label = "template" + str(injection_params["template_index"]) + \
                                        ",sample" + str(injection_params["sample_index"])
                                print("Creating dataset for V1 - " + label)
                                v1_injection_data_group.create_dataset(label, data=snr_sample)
                        else:
                            if injection_params['det_string'] == "h1_strain":
                                label = "template" + str(injection_params["template_index"]) + \
                                        ",sample" + str(injection_params["sample_index"])
                                print("Creating dataset for H1 - " + label)
                                h1_noise_data_group.create_dataset(label, data=snr_sample)
                            elif injection_params['det_string'] == "l1_strain":
                                label = "template" + str(injection_params["template_index"]) + \
                                        ",sample" + str(injection_params["sample_index"])
                                print("Creating dataset for L1 - " + label)
                                l1_noise_data_group.create_dataset(label, data=snr_sample)
                            else:
                                label = "template" + str(injection_params["template_index"]) + \
                                        ",sample" + str(injection_params["sample_index"])
                                print("Creating dataset for V1 - " + label)
                                v1_noise_data_group.create_dataset(label, data=snr_sample)

        finally:
            LOGGER.info("Closing file.")
            h5_file.close()

        return
