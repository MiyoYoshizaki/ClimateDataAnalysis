from tkinter import filedialog as fd
import numpy as np
import pandas as pd
from netCDF4 import Dataset
import os
import matplotlib.pyplot as plt
import glob
import datetime
import math
import pymannkendall as mk
import statsmodels.api as sm
from scipy.stats import pearsonr
import seaborn as sns
from pathlib import Path
import matplotlib.ticker as mticker
import matplotlib.dates as dates

def fileDir():
    fileDir = fd.askdirectory()
    return fileDir

def createfileDir(fileDir):
    newfoldername = 'stat_analysis'
    os.chdir(fileDir)
    p = Path(newfoldername)
    p.mkdir(exist_ok=True)
    newfileDir = os.path.join(fileDir, newfoldername)
    return newfileDir
    
def fileName(fileDir, fileCondition):
    os.chdir(fileDir)
    fileName = glob.glob(fileCondition)
    return fileName

def fileRead(fileDir, fileName, item):
    os.chdir(fileDir)
    #Anglian = 0
    file = Dataset(fileName[0])
    #Total rainfall (mm)
    dataAll = file[item][:]
    dataAnglian = dataAll[:,0]
    #Hours since 1800-01-01 00:00:00
    time = file['time'][:]
    dtStart = datetime.datetime(year=1800, month=1, day=1, hour=0)
    dtTime = np.empty(len(time),dtype='M8[D]')
    n = 0
    m = 0
    y = 0
    for i in range(len(time)):
        deltaDays = int(math.floor(time[i]/24))
        deltaHour = int(time[i] - deltaDays*24)
        dtTime[i] = dtStart +  datetime.timedelta(days=deltaDays, hours=deltaHour)
        if dtTime[i] < np.datetime64('1980','Y'):
            n = n + 1
            years = dtTime[i].astype('datetime64[Y]').astype(int) + 1970
            if years > y:
                y = years
                m = m + 1
    dtTime = dtTime[0:n]
    dataAnglian = dataAnglian[0:n]

    # Generate  dataframe
    dataSet = pd.DataFrame({'time':dtTime, item:dataAnglian}, dtype=object)

    return dataSet

def dataDistribution(path, data, item, figName):
    os.chdir(path)
    data.rename({item: figName}, axis=1, inplace=True)
    sns.displot(data[figName], kind='kde', height=3, aspect=1.2)
    filename = 'stat_distribution_' + item + '.png'
    plt.savefig(os.path.join(path, filename))
    plt.close()
    data.rename({figName: item}, axis=1, inplace=True)
    return

def autocorrelation(path, data, item):
    fig, ax = plt.subplots(figsize=(12, 8))
    sm.graphics.tsa.plot_acf(data[item], lags=20, ax=ax)
    filename = 'autocorrelation_' + item + '.png'
    plt.savefig(os.path.join(path, filename))
    plt.close()
    return

def originalMK(data, item):
    # Original M-K test and trend line generation
    res = mk.original_test(data[item], alpha=0.05)
    trend_line = np.arange(len(data[item])) * res.slope + res.intercept
    return res, trend_line

def seasonalMK(data, item, period):
    # Seasonal M-K test and trend line generation
    res = mk.seasonal_test(data[item], period=period, alpha=0.05)
    trend_line = np.arange(len(data[item])) / period * res.slope + res.intercept
    return res, trend_line

def savetxt(path, data, item, testName):
    # Output the test result to txt file
    txtName = testName + item + '.txt'
    with open(os.path.join(path, txtName), 'w') as f:
        f.write(str(data))
    return

def singlePlot(path, data, trend_line, item, testName):
    # Add trend_line in dataframe
    data['trend_line'] = trend_line.tolist()
    # Plot the test result to png file
    fig, ax = plt.subplots(figsize=(12, 8))
    data.plot(ax=ax, x='time', y=[item, 'trend_line'])
    ax.legend(['data', 'trend line'])
    filename = testName + item + '.png'
    plt.savefig(os.path.join(path, filename))
    plt.close()
    data = data.drop('trend_line', axis=1)
    return

def multiPlot(path, data, trend_line, seasonLabel, item, testName):
    # Add trend_line in dataframe
    for i in range(0,4):
        data[i]['trend_line'] = trend_line[i].tolist()
    # Plot the test result to png file
    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(15, 12))
    plt.subplots_adjust(hspace=0.5)
    fig.suptitle("Seasonal Trend - " + item, fontsize=18, y=0.95)
    axsList = [[0, 0], [0, 1], [1, 0], [1, 1]]
    for i in range(0,4):
        data[i].plot(ax=axs[axsList[i][0]][axsList[i][1]])
        axs[axsList[i][0]][axsList[i][1]].set_title(seasonLabel[i], loc='left')
    filename = testName + item + '-season.png'
    plt.savefig(os.path.join(path, filename))
    plt.close()
    return

def prepSeasonalData(data, item, season, seasonLabel):
    df={}
    for i in range(0,4):
        # Prepare seasonal dataset
        df[i] = data.loc[(pd.DatetimeIndex(data['time']).month == season[i][0]) | \
            (pd.DatetimeIndex(data['time']).month == season[i][1]) | \
            (pd.DatetimeIndex(data['time']).month == season[i][2])]
        df[i] = df[i].groupby(pd.PeriodIndex(df[i]['time'], freq="Y"))[item].mean().reset_index()
        df[i].rename({item: item + '-' + seasonLabel[i]}, axis=1, inplace=True)
    return df

def correlation(precipitation, precLabel, temperature, tasLabel):
    temperature['time'] = temperature['time'].dt.to_timestamp('s').dt.strftime('%Y-%m-%d %H:%M:%S')
    temperature = temperature.loc[pd.DatetimeIndex(temperature['time']).year >= 1891]
    correl = pearsonr(precipitation[precLabel], temperature[tasLabel])
    return correl

def plotThesis(precipDataset, tasDataset):

    time = tasDataset.loc[pd.DatetimeIndex(tasDataset['time']).year >= 1891].reset_index()
    precipDataset['time'] = time['time']

    #define subplot layout
    fig, axes = plt.subplots(nrows=1, ncols=2)

    #add DataFrames to subplots
    # M-K test
    res, trend_line = originalMK(precipDataset, 'rainfall')
    # Add trend_line in dataframe
    precipDataset['trend_line'] = trend_line.tolist()
    # Plot
    plot01 = precipDataset.plot(x='time', y=['rainfall','trend_line'], lw=0.5, legend=False, ax=axes[0])
    plot01.set_title('(a)',loc ='left')
    plot01.set_xlabel("Year")
    plot01.set_ylabel("Precipitation (mm/month)")
    plot01.set_xlim([datetime.date(1880, 1, 1), datetime.date(1980, 1, 1)])
    plot01.set_ylim(0, 200)

    # M-K test
    res, trend_line = seasonalMK(tasDataset, 'tas', 12)
    # Add trend_line in dataframe
    tasDataset['trend_line'] = trend_line.tolist()
    # Plot
    plot02 = tasDataset.plot(x='time', y=['tas','trend_line'], lw=0.5, legend=False, ax=axes[1])
    plot02.set_title('(b)',loc ='left')
    plot02.set_xlabel("Year")
    plot02.set_ylabel("Mean air temperature at 1.5m (degC)")
    plot02.set_xlim([datetime.date(1880, 1, 1), datetime.date(1980, 1, 1)])
    plot02.set_ylim(-5, 25)

    plt.show()
    return

def MKtest():
    filePath = fileDir()
    resultPath = createfileDir(filePath)

    # Read daily precipitation dataset
    inputfileName = fileName(filePath, 'rainfall_*.nc')
    precipDataset = fileRead(filePath, inputfileName, 'rainfall')
    
    # Read monthly mean temperature dataset
    inputfileName = fileName(filePath, 'tas_*.nc')
    tasDataset = fileRead(filePath, inputfileName, 'tas')

    # MK test example available @ below URL
    # https://github.com/mmhs013/pyMannKendall/blob/master/Examples/Example_pyMannKendall.ipynb
    
    # Check data distribution
    dataDistribution(resultPath, precipDataset, 'rainfall', 'precipitation (mm/day)')
    dataDistribution(resultPath, tasDataset, 'tas', 'monthly mean temperature (degree C)')

    # Modify daily precipitation to monthy for avoiding error in M-K test 
    # (Coen et al., 2020 (https://doi.org/10.5194/amt-13-6945-2020))
    precipDataset = precipDataset.groupby(pd.PeriodIndex(precipDataset['time'], freq="M"))['rainfall'].sum().reset_index()
    precipDataset['time'] = precipDataset['time'].dt.to_timestamp('s').dt.strftime('%Y-%m-%d %H:%M:%S')

    # Autocorrelation check 
    autocorrelation(resultPath, precipDataset, 'rainfall')
    autocorrelation(resultPath, tasDataset, 'tas')
    
    ## As a result of autocorrelation check & seasonality
    ## rainfall data --> original M-K test, seasonal M-K test
    ## mean temperature data --> seasonal M-K test

    # Original Mann Kendall test
    result, trend_line = originalMK(precipDataset, 'rainfall')
    savetxt(resultPath, result, 'rainfall', 'originalMK_')
    singlePlot(resultPath, precipDataset, trend_line, 'rainfall', 'originalMK_')

    # Seasonal Mann Kendall test (monthly dataset --> period = 12)
    result, trend_line = seasonalMK(precipDataset, 'rainfall', 12)
    savetxt(resultPath, result, 'rainfall', 'seasonalMK_')
    singlePlot(resultPath, precipDataset, trend_line, 'rainfall', 'seasonalMK_')
    
    result, trend_line = seasonalMK(tasDataset, 'tas', 12)
    savetxt(resultPath, result, 'tas', 'seasonalMK_')
    singlePlot(resultPath, tasDataset, trend_line, 'tas', 'seasonalMK_')
    
    # Further check on seasonal dataset
    # Season - MetOffice ###########################
    # spring (March, April, May), 
    # summer (June, July, August), 
    # autumn (September, October, November) 
    # and winter (December, January, February)
    season = [[3,4,5],[6,7,8],[9,10,11],[12,1,2]]
    seasonLabel = ["spring","summer","autumn","winter"]

    # Prepare seasonal dataset
    precipDatasetSeason = prepSeasonalData(precipDataset, 'rainfall', season, seasonLabel)
    tasDatasetSeason = prepSeasonalData(tasDataset, 'tas', season, seasonLabel)

    # Autocorrelation check 
    for i in range(0,4):
        autocorrelation(resultPath, precipDatasetSeason[i], 'rainfall-'+ seasonLabel[i])
        autocorrelation(resultPath, tasDatasetSeason[i], 'tas-'+ seasonLabel[i])
    
    ## As a result of autocorrelation check & seasonality
    ## all seasonal dataset --> Original Mann Kendall test
    result =  {}
    trend_line = {}
    for i in range(0,4):
        result[i], trend_line[i] = originalMK(precipDatasetSeason[i], 'rainfall-'+ seasonLabel[i])
        savetxt(resultPath, result[i], 'rainfall-'+ seasonLabel[i], 'originalMK_')
    multiPlot(resultPath, precipDatasetSeason, trend_line, seasonLabel, 'rainfall', 'originalMK_')

    result =  {}
    trend_line = {}
    for i in range(0,4):
        result[i], trend_line[i] = originalMK(tasDatasetSeason[i], 'tas-'+ seasonLabel[i])
        savetxt(resultPath, result[i], 'tas-'+ seasonLabel[i], 'originalMK_')
    multiPlot(resultPath, tasDatasetSeason, trend_line, seasonLabel, 'tas', 'originalMK_')

    correl =  {}
    for i in range(0,4):
        correl[i] = correlation(precipDatasetSeason[i], 'rainfall-'+ seasonLabel[i], tasDatasetSeason[i], 'tas-'+ seasonLabel[i])
    savetxt(resultPath, correl, '-correlation', 'pearson')

    plotThesis(precipDataset, tasDataset)

    return


#############################################
if __name__ == "__main__":
    MKtest()