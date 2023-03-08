from email.mime import base
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as dates
import pandas as pd
import csv
from datetime import datetime
import os
from scipy.optimize import curve_fit as cf
import re

def get_master_temp_arrays(dirname):
    """
    Loop over all the Bluefors Lakeshore log folders in folder dirname (with name of the form YY-MM-DD)
    and concatenate all of the data for:
    
    CH6 (Mixing Chamber) Temperature 
    CH9 (RuOx-5869) Resistance 
    CH10 (RuOx-5870) Resistance
    CH11 (RuOx-4221) Resistance
    CH12 (RuOx-5871) Resistance 

    dirname: Directory where all the Lakeshore log folders are stored
    Return: One pair of time and temperature/resistance arrays for Ch6, 9, 10, 11, 12 over all the folders in dirname
    """
    
    subdirs = [d for d in os.listdir(dirname) if os.path.isdir(os.path.join(dirname, d))]
    subdirs.sort(key=lambda x:re.findall('\d+', x))
    
    t_6_arrays, Temp_6_arrays = [], []
    t_9_arrays, R_9_arrays = [], []
    t_10_arrays, R_10_arrays = [], []
    t_11_arrays, R_11_arrays = [], []
    t_12_arrays, R_12_arrays = [], []
    
    for subdir in subdirs:
        f = os.path.join(dirname, subdir)
        
        if os.path.isdir(f):
            ch_6_name_string = f + "/CH6 T " + subdir + ".log" 
            ch_9_name_string = f + "/CH9 R " + subdir + ".log" 
            ch_10_name_string = f + "/CH10 R " + subdir + ".log"
            ch_11_name_string = f + "/CH11 R " + subdir + ".log"
            ch_12_name_string = f + "/CH12 R " + subdir + ".log"
        
            t_6, Temp_6 = get_time_and_value_arrays(ch_6_name_string)
            t_9, R_9 = get_time_and_value_arrays(ch_9_name_string)
            t_10, R_10 = get_time_and_value_arrays(ch_10_name_string)
            t_11, R_11 = get_time_and_value_arrays(ch_11_name_string)
            t_12, R_12 = get_time_and_value_arrays(ch_12_name_string)
        
        
            t_6_arrays.append(t_6)
            Temp_6_arrays.append(Temp_6)

            t_9_arrays.append(t_9)
            R_9_arrays.append(R_9)

            t_10_arrays.append(t_10)
            R_10_arrays.append(R_10)

            t_11_arrays.append(t_11)
            R_11_arrays.append(R_11)

            t_12_arrays.append(t_12)
            R_12_arrays.append(R_12)
        

    t_6_arrays = np.concatenate(t_6_arrays)
    Temp_6_arrays = np.concatenate(Temp_6_arrays)

    t_9_arrays = np.concatenate(t_9_arrays)
    R_9_arrays = np.concatenate(R_9_arrays)

    t_10_arrays = np.concatenate(t_10_arrays)
    R_10_arrays = np.concatenate(R_10_arrays)
    
    t_11_arrays = np.concatenate(t_11_arrays)
    R_11_arrays = np.concatenate(R_11_arrays)

    t_12_arrays = np.concatenate(t_12_arrays)
    R_12_arrays = np.concatenate(R_12_arrays)
    
    Temp_11_arrays = calibration_function(R_11_arrays)
    
    
    return t_6_arrays, Temp_6_arrays, t_11_arrays, Temp_11_arrays, t_9_arrays, R_9_arrays, t_10_arrays, R_10_arrays, t_12_arrays, R_12_arrays

def get_time_and_value_arrays(fname):
    """
    Get time and value (resistance/temperature) data from the 
    Snoopy Lakeshore Log Files

    fname: Log file name
    Return: Time and value data numpy arrays
    """
    df = pd.read_csv(fname, sep='\s\s+', engine='python')
    
    # create dummy csv file from log file to readout data
    csv_fname = fname.rstrip(".log") + ".csv"
    df.to_csv(csv_fname, index=None)
    
    time_values, y_axis_values = [], []
    
    with open(csv_fname) as file:
        reader = csv.reader(file)
        for row in reader:
            split_outputs = row[0].split(",")
            date_string = split_outputs[0]
            time_string = split_outputs[1]
            
            datetime_string = date_string + " " + time_string
            time = datetime.strptime(datetime_string, "%d-%m-%y %H:%M:%S")

            value_string = split_outputs[2]

            time_values.append(time)
            y_axis_values.append(float(value_string))
    
    # close and delete dummy csv file
    file.close()
    os.remove(csv_fname)

    return np.array(time_values), np.array(y_axis_values)

def cut_arrays_by_datetime(time, values, ref_date_lower, ref_date_upper):
    """
    Cut a set of time and resistance/temperature arrays by an upper and
    lower datetime

    time: Input time array
    values: Input value array (resistance or temperature)
    ref_date_lower: The earlier datetime 
    ref_date_upper: The later datetime
    Return: the time and value arrays within the window of ref_date_lower/upper
    """
    assert ref_date_upper > ref_date_lower
    values = values[(ref_date_lower < time) & (time < ref_date_upper)]
    time = time[(ref_date_lower < time) & (time < ref_date_upper)]
    return time, values

def get_arrays_from_file_cut_by_time(fname, ref_date_lower, ref_date_upper):
    """
    Cut arrays by datetime but using a log file as input

    fname: The log file in which the arrays exist
    ref_date_lower: The earlier datetime 
    ref_date_upper: The later datetime
    Return: the time and value arrays in fname within the window of ref_date_lower/upper
    """
    t, v = get_time_and_value_arrays(fname)
    t, v = cut_arrays_by_datetime(t, v, ref_date_lower, ref_date_upper)
    return t, v

def calibration_function(r):
    """
    RuOx-4221 Calibration Function to convert resistance to temperature

    r: Resistance in ohms
    Return: temperature in Kelvin
    """
    log_r = np.log10(r)
    log_t = 1182.9999631011303 + \
            log_r * -1608.0259474126617 + \
            ((log_r)**2) * 907.8127700668347 + \
            ((log_r)**3) * -271.96429769488213 + \
            ((log_r)**4) * 45.52802670739256 + \
            ((log_r)**5) * -4.034719629444096 + \
            ((log_r)**6) * 0.14779818043050427
    return 10**log_t

## C/G Time Functions

def warmup(t, A, tau):
    """
    Fit function for exponential warmup T = A(1-exp(-t/tau))

    t: Elapsed time 
    A: The asymptote value of temperature
    tau: C/G time constant
    Return: The temperature after time t
    """
    return A*(1-np.exp(-t/tau))

def cooldown(t, A, tau):
    """
    Fit function for exponential cooldown T = A*exp(-t/tau)

    t: Elapsed time 
    A: The asymptote value of temperature
    tau: C/G time constant
    Return: The temperature after time t
    """
    return A*np.exp(-t/tau)

def get_cg_time(start, end, time_array, temp_array, iswarmup):
    """
    Get the C/G time from a set of time-temperature data
    
    start: Datetime object indicating start of run
    end: Datetime object indicating end of run
    time_array: Time data
    temp_array: Temperature dat
    iswarmup: Boolean flag: true if warmup, false if cooldown
    Return: C/G Time for the given cooldown/warmup data.
    """

    t, temp = cut_arrays_by_datetime(time_array, 
                                    temp_array, 
                                    start, 
                                    end)
    if iswarmup:
        function = warmup 
        base_temp = temp[0]
        p0=[temp[-1], 300]
    else:
        function = cooldown 
        base_temp = temp[-1]
        p0=[temp[0], 300]

    t = np.array([(date-start).total_seconds() for date in t])
    temp_delta = temp - base_temp

    popt, pcov = cf(function, t, temp_delta, p0) 
 
    tau = popt[1]
    
    return tau, t, temp_delta, popt, base_temp