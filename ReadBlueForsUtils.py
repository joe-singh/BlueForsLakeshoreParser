from email.mime import base
from operator import truediv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as dates
import pandas as pd
import csv
from datetime import datetime
import os
from scipy.optimize import curve_fit as cf
import re

def get_master_arrays(dirname, temps=False, ch_nos=[10, 11, 12]):
    """
    Loop over all the Bluefors Lakeshore log folders in folder dirname (with name of the form YY-MM-DD)
    and concatenate all of the data for:
    
    CH6 (Mixing Chamber) Temperature 
    CH9 Resistance 
    CH10 Resistance
    CH11 Resistance 
    CH12 Resistance 
    CH13 Resistance
    CH14 Resistance

    dirname: Directory where all the Lakeshore log folders are stored
    temps: Flag for if you want temperature or resistance arrays. The raw data is resistance arrays but if temps is True then
           the function applies the resistance to temperature calibration function and returns the temperature instead of resistance. 
           Set to false by default, so you can apply a calibration function yourself. 
    ch_nos: Channel Numbers. Expected to be 9, 10, 11, 12, 13, 14. Set by default to 10, 11, 12 since that's what I use most often
    Return: Time, temperature and resistance arrays for Ch6 and time and either temperature or resistance (depending on temps) 
    arrays for the channels in ch_nos (9, 10, 11, 12, 13, 14) 
    """
    
    subdirs = [d for d in os.listdir(dirname) if os.path.isdir(os.path.join(dirname, d))]
    subdirs.sort(key=lambda x:re.findall('\d+', x))

    nine_flag=True
    ten_flag=True 
    eleven_flag=True
    twelve_flag=True 
    thirteen_flag=True
    fourteen_flag=True


    if 9 not in ch_nos:
        nine_flag = False
    if 10 not in ch_nos:
        ten_flag = False 
    if 11 not in ch_nos:
        eleven_flag = False 
    if 12 not in ch_nos:
        twelve_flag = False 
    if 13 not in ch_nos:
        thirteen_flag = False
    if 14 not in ch_nos:
        fourteen_flag = False
    
    t_6_arrays, Temp_6_arrays, R_6_arrays = [], [], []
    t_9_arrays, R_9_arrays = [], []
    t_10_arrays, R_10_arrays = [], []
    t_11_arrays, R_11_arrays = [], []
    t_12_arrays, R_12_arrays = [], []
    t_13_arrays, R_13_arrays = [], []
    t_14_arrays, R_14_arrays = [], []
    
    for subdir in subdirs:
        f = os.path.join(dirname, subdir)
        
        if os.path.isdir(f):
            ch_6_name_string = f + "/CH6 T " + subdir + ".log"
            ch_6_resistance_string = f + "/CH6 R " + subdir + ".log"
            t_6, Temp_6 = get_time_and_value_arrays(ch_6_name_string)
            _, R_6 = get_time_and_value_arrays(ch_6_resistance_string)  
            t_6_arrays.append(t_6)
            Temp_6_arrays.append(Temp_6)
            R_6_arrays.append(R_6)

            if nine_flag: 
                ch_9_name_string = f + "/CH9 R " + subdir + ".log"
                t_9, R_9 = get_time_and_value_arrays(ch_9_name_string)
                t_9_arrays.append(t_9)
                R_9_arrays.append(R_9)
            if ten_flag:
                ch_10_name_string = f + "/CH10 R " + subdir + ".log"
                t_10, R_10 = get_time_and_value_arrays(ch_10_name_string)
                t_10_arrays.append(t_10)
                R_10_arrays.append(R_10)
            if eleven_flag:
                ch_11_name_string = f + "/CH11 R " + subdir + ".log"
                t_11, R_11 = get_time_and_value_arrays(ch_11_name_string)
                t_11_arrays.append(t_11)
                R_11_arrays.append(R_11)
            if twelve_flag:
                ch_12_name_string = f + "/CH12 R " + subdir + ".log"
                t_12, R_12 = get_time_and_value_arrays(ch_12_name_string)
                t_12_arrays.append(t_12)
                R_12_arrays.append(R_12)
            if thirteen_flag:
                ch_13_name_string = f + "/CH13 R " + subdir + ".log"
                t_13, R_13 = get_time_and_value_arrays(ch_13_name_string)
                t_13_arrays.append(t_13)
                R_13_arrays.append(R_13)
            if fourteen_flag:
                ch_14_name_string = f + "/CH14 R " + subdir + ".log"
                t_14, R_14 = get_time_and_value_arrays(ch_14_name_string)
                t_14_arrays.append(t_14)
                R_14_arrays.append(R_14)

    t_6_arrays = np.concatenate(t_6_arrays)
    Temp_6_arrays = np.concatenate(Temp_6_arrays)
    R_6_arrays = np.concatenate(R_6_arrays)

    if nine_flag:
        t_9_arrays = np.concatenate(t_9_arrays)
        R_9_arrays = np.concatenate(R_9_arrays)
    
    if ten_flag:
        t_10_arrays = np.concatenate(t_10_arrays)
        R_10_arrays = np.concatenate(R_10_arrays)
    
    if eleven_flag:
        t_11_arrays = np.concatenate(t_11_arrays)
        R_11_arrays = np.concatenate(R_11_arrays)
    
    if twelve_flag:
        t_12_arrays = np.concatenate(t_12_arrays)
        R_12_arrays = np.concatenate(R_12_arrays)
    
    if thirteen_flag:
        t_13_arrays = np.concatenate(t_13_arrays)
        R_13_arrays = np.concatenate(R_13_arrays)
    
    if fourteen_flag:
        t_14_arrays = np.concatenate(t_14_arrays)
        R_14_arrays = np.concatenate(R_14_arrays)

    
    return_arrs = []
    return_arrs.append(t_6_arrays) 
    return_arrs.append(Temp_6_arrays) 
    return_arrs.append(R_6_arrays)

    if nine_flag:
        return_arrs.append(t_9_arrays)
        if temps:
            Temp_9_arrays = calibration_function_9(R_9_arrays)
            return_arrs.append(Temp_9_arrays)
        else:
            return_arrs.append(R_9_arrays)
    if ten_flag:
        return_arrs.append(t_10_arrays)
        if temps:
            Temp_10_arrays = calibration_function_10(R_10_arrays)
            return_arrs.append(Temp_10_arrays)
        else:
            return_arrs.append(R_10_arrays)
    if eleven_flag:
        return_arrs.append(t_11_arrays)
        if temps:
            Temp_11_arrays = calibration_function_11(R_11_arrays)
            return_arrs.append(Temp_11_arrays)
        else:
            return_arrs.append(R_11_arrays)
    if twelve_flag:
        return_arrs.append(t_12_arrays)
        if temps:
            Temp_12_arrays = calibration_function_12(R_12_arrays)
            return_arrs.append(Temp_12_arrays)
        else:
            return_arrs.append(R_12_arrays)
    if thirteen_flag:
        return_arrs.append(t_13_arrays)
        if temps:
            pass # No calibration function for channel 13
            #Temp_13_arrays = calibration_function_13(R_12_arrays)
            #return_arrs.append(Temp_12_arrays)
        else:
            return_arrs.append(R_13_arrays)
    if fourteen_flag:
        return_arrs.append(t_14_arrays)
        if temps:
            pass # No calibration function for channel 14
            #Temp_13_arrays = calibration_function_13(R_12_arrays)
            #return_arrs.append(Temp_12_arrays)
        else:
            return_arrs.append(R_14_arrays)

    return return_arrs #t_6_arrays, Temp_6_arrays, R_6_arrays, t_9_arrays, Temp_9_arrays, t_10_arrays, Temp_10_arrays, t_11_arrays, Temp_11_arrays, t_12_arrays, Temp_12_arrays


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


def bluefors_temp_error(temps):
    """Put error bars on the Bluefors MXC temperature. From Pradheesh's email on 9/20 where he says error between 15-100 mK < 5% but below 10 mK it can be 10%"""
    errors = []

    for temp in temps:
        if temp < 15e-3:
            errors.append(0.1*temp)
        else: 
            errors.append(0.05*temp)
    
    return np.array(errors)

def calibration_function_9(r):
    """
    RuOx-5870 Calibration Function to convert resistance to temperature

    r: Resistance in ohms
    Return: temperature in Kelvin
    """
    log_r = np.log10(r)
    log_t = -1.010359743562858 * log_r**6 + \
            24.98874761505759 * log_r**5 + \
            -256.5233885355524 * log_r**4 + \
            1398.7550424544854 * log_r**3 + \
            -4271.57638623511 * log_r**2 + \
            6923.456319400351 * log_r + \
            -4649.869041441679
    return 10**(log_t)

def calibration_function_10(r):
    """
    RuOx-5869 Calibration Function to convert resistance to temperature

    r: Resistance in ohms
    Return: temperature in Kelvin
    """
    log_r = np.log10(r)
    log_t = -0.6500383015723634 * log_r**6 + \
            16.4860065626999 * log_r**5 + \
            -173.36739219882602 * log_r**4 + \
            967.3618693239875 * log_r**3 + \
            -3019.5631210780202 * log_r**2 + \
            4996.025196409633 * log_r + \
            -3420.2207405545496
    return 10**(log_t)


def calibration_function_11(r):
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

def calibration_function_12(r):
    """
    RuOx-5871 Calibration Function to convert resistance to temperature

    r: Resistance in ohms
    Return: temperature in Kelvin
    """
    log_r = np.log10(r)
    log_t = -0.5792896802925668 * log_r**6 + \
            14.550215238428517 * log_r**5 + \
            -151.514259636263 * log_r**4 + \
            836.9939029985673 * log_r**3 + \
            -2585.8799685039194 * log_r**2 + \
            4232.9337322469 * log_r + \
            -2865.176346538847
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