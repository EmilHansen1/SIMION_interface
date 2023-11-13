##################################################### 
#   AUTHOR: Emil Hansen                             #
#   SUMMARY: Program for controlling SIMION         #
#   via the command line in linux. Contains         #
#   functions for creating geometries, making       #
#   potential arrays, flying ions and performing    #
#   optimization of voltages and geometry.          #
#####################################################


import numpy as np
import matplotlib.pyplot as plt
import subprocess
import os
import pandas as pd
import time
from scipy.optimize import minimize, fmin



def run_lua_script():
    ''' Runs a lua script in SIMION - example so I don't forget. Don't use the interactive flag!!! '''
    subprocess.run(['/bin/bash', '-i', '-c', 'simion --nogui lua make_geometry.lua'])


def simion_command(command):
    simion_path_exe = 'wine /home/emil/.wine/drive_c/"Program Files (x86)"/SIMION-8.0/simion.exe'
    cmd = ['/bin/bash', '-c', simion_path_exe + ' ' + command]
    subprocess.call(cmd)


def electrode_string(electrode_index, a, b, c, d):
    ''' Creates an electrode for the .GEM file in the style of a corner box'''
    return f'electrode({electrode_index}) {{ fill {{ within  {{ corner_box({a}, {b}, {c}, {d}) }} }} }}\n'


def make_geometry(total_length, radius, lens_positions, lens_outer_diameter, lens_inner_diameter, electrode_width=1, filename='geometry'):
    # Create file
    gem_file = open(filename + '.gem', 'w')

    # Define workspace
    gem_file.write(f'pa_define({total_length}, {radius}, 1, cylindrical, electrostatic)\n')

    for i in range(len(lens_positions)):
        # Get the z-position of the i'th (current) electrode
        current_length = np.sum([lens_positions[j] for j in range(i+1)])

        gem_file.write(electrode_string(i+1,                                            # Electrode index (from 1)
                                        current_length,                                 # Start point
                                        int(lens_inner_diameter[i]/2),                  # Inner radius
                                        electrode_width,                                # Width (z-direction)
                                        int(radius - lens_inner_diameter[i]/2 - 2)))    # Radial extent
    gem_file.close()
    print('Geometry file created\n')

    # Create .PA-files from .GEM
    simion_command(f'--nogui gem2pa {filename}.gem {filename}.pa#')
    print('PA#-file created\n')
    
    # Refine 
    simion_command(f'--nogui refine --convergence=1e-3 {filename}.pa#')
    print('----- GEOEMTRY AND PAs ALL DONE -----\n')


def fly(workspace_filename, rec_filename, output_filename):
    ''' Fly ions with SIMION given a workspace, .REC-file and filename out output file '''
    # Remove possibly old data
    if os.path.exists(f'{output_filename}.txt'):
        os.remove(f'{output_filename}.txt')

    # Fly and save
    simion_command(f'--nogui fly --particles=ions.fly2 --restore-potentials=0 --recording={rec_filename}.rec --recording-output={output_filename}.txt {workspace_filename}.iob')


def load_flight_data(output_filename):
    ''' Read flight data into pandas dataframe '''
    # Get the index for the dataframe
    df_index = []
    with open(f'{output_filename}.txt') as file:
        for line in file:
            if '"Ion N"' in line:                   # Only the index line contains this substring
               df_index = line.split(',')
               for i, idx in enumerate(df_index):
                    if '\n' in idx:
                        df_index[i] = idx[0:-1]     # Remove \n from last letter
                    df_index[i] = idx[1:-1]         # Remove " from both sides of index
               break

    # Load numerical data and create dataframe
    flight_data = np.loadtxt(f'{output_filename}.txt', skiprows=12, delimiter=',')
    df = pd.DataFrame(flight_data, columns=df_index)
    initial_conditions = df[::2].reset_index()
    final_conditions = df[1::2].reset_index()

    return initial_conditions, final_conditions


def vmi_cost_function(voltages, workspace_filename, rec_filename, data_filename):
    ''' The cost function for velovity map imaging focus to be minimized by adjusting volages '''

    # First perform fast adjust of voltages
    simion_command(f'--nogui fastadj {workspace_filename}.pa0 1={4500},2={voltages[0]},3={voltages[1]},4=0,5=0')

    # Then fly the ions
    fly(workspace_filename, rec_filename, data_filename)
    df_ic, df_res = load_flight_data(data_filename)
    subprocess.call(['/bin/bash', '-c', 'rm *.tmp'])


    # Calculate the penalty (same velocity must have same y-values)
    # We group particles together by initial energy and minimize dy
    penalty = 0
    initial_energies = df_ic['KE'].unique()
    for energy in initial_energies:
        y_pos = df_res.loc[df_ic['KE'] == energy]['Y']
        penalty += (y_pos.max() - y_pos.min())**2 

    for i in range(len(voltages) - 1):
        if voltages[i+1] > voltages[i]:
            penalty += 500#(voltages[i+1] - voltages[i])**2

    
    print('THE COST FOR THIS RUN IS: ', penalty, '\n')
    return penalty



if __name__ == '__main__':
    # Create electrode property arrays
    lengths = [1, 14, 21, 29, 300]
    diameters = [6, 12, 32, 50, 0]
    n_electrodes = 5
    radius = 100 + 1  # Plus one for spacing
    electrode_width = 1
    total_length = np.sum(lengths) + electrode_width + 2  # Includes one free point on either side

    # Make geometry file and potential arrays
    #make_geometry(total_length, radius, lengths, diameters, diameters, electrode_width=electrode_width, filename='test')

    # Fly ions with test.iob and save results in flights/result.rec
    #fly('test', 'data', 'data')

    # Load in the data
    #df_ic, df_res = load_flight_data('data')
    # Voltage optimization
    voltages_guess = [3000, 100]
    res = minimize(vmi_cost_function, voltages_guess, args=('test', 'data', 'data'), method='Nelder-Mead')
    #res = fmin(vmi_cost_function, voltages_guess, args=('test', 'data', 'data'), maxiter=20)
    print(res)
    #vmi_cost_function(voltages_guess, 'test', 'data', 'data')




