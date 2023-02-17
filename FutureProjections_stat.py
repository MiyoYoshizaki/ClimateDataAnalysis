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

def readDataset(fileDir, fileName, item, dateFormat):
    os.chdir(fileDir)
    file = Dataset(fileName[0])
    dfAll = file[item][:]
    #Anglian = 0
    dfAnglian = dfAll[0,:,0]
    #Conceptual date in climate model
    YMD = file[dateFormat][:]
    return dfAnglian, YMD

def modDailyDataset(dfAnglian, YMD):
    year = np.empty(len(YMD[:,0]),dtype="<U4")
    month = np.empty(len(YMD[:,0]),dtype="<U2")
    day = np.empty(len(YMD[:,0]),dtype="<U2")
    y_str = np.empty(len(YMD[:,0]),dtype="<U10")
    for j in range(len(YMD[:,0])):
        for i in range(len(YMD[0,:])):
            if i < 4:
                year[j] = year[j] + str(YMD[j,i],'UTF-8')
            elif i < 6:
                month[j] = month[j] + str(YMD[j,i],'UTF-8')
            elif i < 8:
                day[j] = day[j] + str(YMD[j,i],'UTF-8')
        y_str[j] = year[j] + '-' + month[j] + '-' + day[j]
    dateStart = datetime.date(int(year[0]), int(month[0]), int(day[0]))
    dateEnd = datetime.date(int(year[len(YMD[:,0])-1]), int(month[len(YMD[:,0])-1]), int(day[len(YMD[:,0])-1]))
    diff = dateEnd - dateStart
    dtTime = np.empty(diff.days + 1,dtype='M8[D]')
    dfAnglianModif = np.zeros(diff.days+ 1)
    days31 = ["01","03","05","07","08","10","12"]
    days28 = ["02"]
    n = 0
    for j in range(len(y_str)):
        if month[j] in days31:
            if day[j] == "30":
                dtTime[j+n] = datetime.date(int(year[j]), int(month[j]), int(day[j]))
                dfAnglianModif[j+n] = dfAnglian[j]
                n = n + 1
                dtTime[j+n] = datetime.date(int(year[j]), int(month[j]), 31)
                dfAnglianModif[j+n] = (dfAnglian[j]+dfAnglian[j+1])/2
            elif int(day[j]) < 30:
                dtTime[j+n] = datetime.date(int(year[j]), int(month[j]), int(day[j]))
                dfAnglianModif[j+n] = dfAnglian[j]
        elif month[j] in days28:
            if ((int(year[j])-1976)/4).is_integer():
                if day[j] == "29":
                    dtTime[j+n] = datetime.date(int(year[j]), int(month[j]), int(day[j]))
                    dfAnglianModif[j+n] = dfAnglian[j]
                    n = n - 1
                elif int(day[j]) < 29:
                    dtTime[j+n] = datetime.date(int(year[j]), int(month[j]), int(day[j]))
                    dfAnglianModif[j+n] = dfAnglian[j]
            elif day[j] == "28":
                dtTime[j+n] = datetime.date(int(year[j]), int(month[j]), int(day[j]))
                dfAnglianModif[j+n] = dfAnglian[j]
                n = n - 2
            elif int(day[j]) < 28:
                dtTime[j+n] = datetime.date(int(year[j]), int(month[j]), int(day[j]))
                dfAnglianModif[j+n] = dfAnglian[j]
        else:
            dtTime[j+n] = datetime.date(int(year[j]), int(month[j]), int(day[j]))
            dfAnglianModif[j+n] = dfAnglian[j]
    outData = list(zip(dtTime,dfAnglianModif))
    return outData

def modMonthlyDataset(dfAnglian, YMD):
    year = np.empty(len(YMD[:,0]),dtype="<U4")
    month = np.empty(len(YMD[:,0]),dtype="<U2")
    y_str = np.empty(len(YMD[:,0]),dtype="<U10")
    for j in range(len(YMD[:,0])):
        for i in range(len(YMD[0,:])):
            if i < 4:
                year[j] = year[j] + str(YMD[j,i],'UTF-8')
            elif i < 6:
                month[j] = month[j] + str(YMD[j,i],'UTF-8')
        y_str[j] = year[j] + '-' + month[j] + '-' + str(16)
    outData = list(zip(y_str,dfAnglian))
    return outData

def saveDataset(path, df, item, header, format):
    os.chdir(path)
    np.savetxt(item, df, fmt=format, delimiter=",", header = ', '.join(header))  
    return

def prFile(filePath, condition):
    # Read daily dataset
    # Precipitation rate (mm/day)
    inputfileName = fileName(filePath, condition)
    dfAnglian, YMD = readDataset(filePath, inputfileName, 'pr', 'yyyymmdd')
    dfAnglian = modDailyDataset(dfAnglian, YMD)

    # Save generated dataset in csv
    header = ['time',"Total rainfall (mm/day)"]
    format = '%s', '%1.5f'
    saveDataset(filePath, dfAnglian, "rainfall_RCP85.csv", header, format)
    return dfAnglian

def tasFile(filePath, condition):
    # Read monthly dataset
    # Mean air temperature at 1.5m (degC)
    inputfileName = fileName(filePath, condition)
    dfAnglian, YMD = readDataset(filePath, inputfileName, 'tas', 'yyyymm')
    dfAnglian = modMonthlyDataset(dfAnglian, YMD)

    # Save generated dataset in csv
    header = ['time',"Mean air temperature at 1.5m (degC)"]
    format = '%s', '%s'
    saveDataset(filePath, dfAnglian, "tas_RCP85.csv", header, format)
    return dfAnglian

def readData(condition):
    filePath = fileDir()
    dfRain = prFile(filePath, condition[0])
    dfTas = tasFile(filePath, condition[1])
    dfRain = pd.DataFrame(dfRain).set_axis(['time', 'rainfall'], axis=1)
    dfTas = pd.DataFrame(dfTas).set_axis(['time', 'tas'], axis=1)
    return dfRain, dfTas, filePath

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

def MKtest(dfRain, dfTas, filePath):
    resultPath = createfileDir(filePath)

    # MK test example available @ below URL
    # https://github.com/mmhs013/pyMannKendall/blob/master/Examples/Example_pyMannKendall.ipynb
    
    # Check data distribution
    dataDistribution(resultPath, dfRain, 'rainfall', 'precipitation (mm/day)')
    dataDistribution(resultPath, dfTas, 'tas', 'monthly mean temperature (degree C)')

    # Modify daily precipitation to monthy for avoiding error in M-K test 
    # (Coen et al., 2020 (https://doi.org/10.5194/amt-13-6945-2020))
    dfRain = dfRain.groupby(pd.PeriodIndex(dfRain['time'], freq="M"))['rainfall'].sum().reset_index()
    dfRain['time'] = dfRain['time'].dt.to_timestamp('s').dt.strftime('%Y-%m-%d %H:%M:%S')

    # Autocorrelation check 
    autocorrelation(resultPath, dfRain, 'rainfall')
    autocorrelation(resultPath, dfTas, 'tas')

    ## As a result of autocorrelation check & seasonality
    ## rainfall data --> original M-K test, seasonal M-K test
    ## mean temperature data --> seasonal M-K test

    # Original Mann Kendall test
    result, trend_line = originalMK(dfRain, 'rainfall')
    savetxt(resultPath, result, 'rainfall', 'originalMK_')
    singlePlot(resultPath, dfRain, trend_line, 'rainfall', 'originalMK_')

    # Seasonal Mann Kendall test (monthly dataset --> period = 12)
    result, trend_line = seasonalMK(dfRain, 'rainfall', 12)
    savetxt(resultPath, result, 'rainfall', 'seasonalMK_')
    singlePlot(resultPath, dfRain, trend_line, 'rainfall', 'seasonalMK_')
    
    result, trend_line = seasonalMK(dfTas, 'tas', 12)
    savetxt(resultPath, result, 'tas', 'seasonalMK_')
    singlePlot(resultPath, dfTas, trend_line, 'tas', 'seasonalMK_')

    # Further check on seasonal dataset
    # Season - MetOffice ###########################
    # spring (March, April, May), 
    # summer (June, July, August), 
    # autumn (September, October, November) 
    # and winter (December, January, February)
    season = [[3,4,5],[6,7,8],[9,10,11],[12,1,2]]
    seasonLabel = ["spring","summer","autumn","winter"]

    # Prepare seasonal dataset
    precipDatasetSeason = prepSeasonalData(dfRain, 'rainfall', season, seasonLabel)
    tasDatasetSeason = prepSeasonalData(dfTas, 'tas', season, seasonLabel)

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
    
    return 

#############################################
if __name__ == "__main__":
    dfRain, dfTas, path = readData(['pr_rcp85_*.nc', 'tas_rcp85_*.nc'])
    MKtest(dfRain, dfTas, path)
