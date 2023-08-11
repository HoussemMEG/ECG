import time
from datetime import datetime
from multiprocessing import Queue, Process
from multiprocessing import current_process

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import winsound

import utils
from featuregen import DFG
from plot import Plotter
from preprocess import Preprocess
from reader import Reader
from saver import Saver

pd.set_option('display.max_columns', None)
pd.set_option('display.expand_frame_repr', False)
matplotlib.use('QT5agg')


def compute_SDF(working_q, output_q):
    verbose = current_process().name == 'Process-1'  # print enabled only for the first process
    feature_gen = DFG(method='LARS',
                      f_sampling=400,
                      version=1,
                      find_alpha=False,
                      alpha=[0.00016, 0.00018, 0.00012, 0.00016, 0.00011, 0.00013, 0.00017, 0.00021, 0.00026, 0.00034, 0.00032, 0.00025],
                      model_freq=np.linspace(3, 45, 35, endpoint=True),
                      normalize=True,
                      damping=None,  # (under-damped 0.008 / over-damped 0.09)
                      fit_path=True, ols_fit=True,
                      fast=False,
                      selection=[],  # 0.02
                      selection_alpha=None,
                      omit=None,  # omit either 'x0' or 'u' from y_hat computation
                      plot=(False, True), show=True, fig_name="fig name", save_fig=True,
                      verbose=verbose)

    while True:
        temp = working_q.get()
        if temp is not None and not isinstance(temp, str):
            exam_id = temp[0]
            y = temp[-1]
            features, x0 = feature_gen.generate(y)
            features, x0 = utils.compress(features), utils.ndarray_to_list(x0)
            output_q.put((exam_id, features, x0))
            plt.show()

        else:
            output_q.put(None)
            if working_q.empty():
                output_q.put(feature_gen.parameters)
            break


if __name__ == '__main__':
    # Parameters
    cpu_count = 3
    conditions = ['HEALTHY']  # ['1dAVb', 'RBBB', 'LBBB', 'SB', 'AF', 'ST', 'HEALTHY']
    random = True

    # Init the class instances'
    session = datetime.now().strftime("%Y-%m-%d %H;%M")
    session = "2800 per category _ 35 freq _ long window"
    set_name = 'test'  # ['train', 'validation', 'test', 'learning']
    reader = Reader(batch_size=1, n=1, stratified=True, set_name=set_name)  # 699
    plotter = Plotter(show=False, save=False)
    preprocess = Preprocess(fs=400, before=0.2, after=0.4)  # after = 0.1 or 0.4
    saver = Saver(session=session, set_name=set_name, conditions=Reader.ALL_CONDITIONS, save=False, verbose=True)

    tic = time.perf_counter()
    # Read data
    for x, meta_data in reader.read(condition=conditions, random=random):
        # Plot the raw signal before preprocessing
        plotter.plot_signal(x, meta_data=meta_data, select_lead=[])

        # Pre-processing
        x, r_peaks, epochs, meta_data = preprocess.process(x, meta_data)

        # Plot the signal after preprocessing and the detected R-peaks
        plotter.plot_signal(x, meta_data=meta_data, r_peaks=r_peaks, select_lead=[])

        # Plot the extracted epochs
        plotter.plot_epochs(epochs, meta_data=meta_data, select_lead=[])  # plot epochs

        # Prepare containers and queues
        processes = []
        exams_id = meta_data['exam_id'].to_numpy()
        working_q = utils.init_working_q(epochs, exams_id, cpu_count)
        output_q = Queue()

        # Start <cpu_count> processes
        for i in range(cpu_count):
            p = Process(target=compute_SDF, args=(working_q, output_q))
            p.start()
            utils.print_c('Process {:} has started'.format(i + 1), 'green', bold=True)
            processes.append(p)

        # Handle output data after SDF generation and store the parameters of the feature generator
        feat_gen_param = utils.output_q_handler(output_q, meta_data, cpu_count, saver, reader.ALL_CONDITIONS)

        # Wait for all process to complete their job and end their work
        for i, p in enumerate(processes):
            p.join()
            utils.print_c('Process {:} ended'.format(i + 1), 'green', bold=True)
        plt.show()
        break
    # Save the parameters of the preprocessing and the feature generator for replication purposes
    saver.save_json(preprocess.parameters, file_name='preprocessing_parameters')
    saver.save_json(feat_gen_param, file_name='DFG_parameters')

    plt.show()  # Show the plots
    print('Running time {:.2f}'.format(time.perf_counter() - tic), flush=True)

    winsound.Beep(440, 1000)

"""
# to be removed at the end
    # Test data
    test_data, annotation = reader.read_test(condition=condition, index=5)  # index = 5 is the gold standard
    x, r_peaks, epochs = preprocess.process(test_data)
    plotter.plot_signal(x[0:1, ...], r_peaks=r_peaks[0:1], select_lead=[])
    plotter.plot_epochs(epochs[0:1], select_lead=[])
"""

"""
alpha for np.linspace(3, 45, 40, endpoint=True) 
    (-0.2, 0.4)
        [0.00016, 0.00018, 0.00012, 0.00016, 0.00011, 0.00013, 0.00018, 0.00021, 0.00027, 0.00033, 0.00033, 0.00026],


alpha for np.linspace(3, 45, 70, endpoint=True) 
    (-0.2, 0.4)
        [0.00016, 0.00018, 0.00013, 0.00016, 0.00012, 0.00013, 0.00019, 0.00023, 0.00027, 0.00036, 0.00033, 0.00026],
   
    
alpha for np.linspace(3, 45, 10, endpoint=True) 
    (-0.2, 0.4)
        [0.00003, 0.00004, 0.00003, 0.00003, 0.00003, 0.00003, 0.00005, 0.00005, 0.00006, 0.00007, 0.00007, 0.00005], 

alpha for np.linspace(3, 45, 70, endpoint=True) 
    (-0.2, 0.4)
    (-0.2, 0.1)
        [0.00028, 0.00033, 0.00023, 0.00028, 0.00018, 0.00025, 0.00035, 0.00043, 0.00049, 0.00057, 0.00058, 0.00048],
        
        
alpha for np.linspace(3, 45, 35, endpoint=True) 
    (-0.2, 0.4)
        [0.00016, 0.00018, 0.00012, 0.00016, 0.00011, 0.00013, 0.00017, 0.00021, 0.00026, 0.00034, 0.00032, 0.00025],
    (-0.2, 0.1)
        [0.00033, 0.00039, 0.00023, 0.00034, 0.00022, 0.00027, 0.00038, 0.00046, 0.00060, 0.00067, 0.00066, 0.00056],

alpha for np.linspace(3, 45, 25, endpoint=True) 
    (-0.2, 0.4)
    (-0.2, 0.1) 
        [0.00030, 0.00034, 0.00022, 0.00032, 0.00020, 0.00025, 0.00036, 0.00043, 0.00050, 0.00062, 0.00063, 0.00050],


alpha for np.linspace(3, 45, 10, endpoint=True) 
    (-0.2, 0.4)
    (-0.2, 0.1) 
        [0.00017, 0.00020, 0.00014, 0.00018, 0.00012, 0.00015, 0.00021, 0.00025, 0.00031, 0.00036, 0.00039, 0.00031],
"""
