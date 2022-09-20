import pandas as pd
import numpy as np

import holidays


def smape(y_true, y_pred):
    
    smape = abs(y_true - y_pred) / (abs(y_true) + abs(y_pred))
    smape = smape.mean() * 200
    
    return smape


def is_holiday(train, test):
    
    ## Extracting holidays
    be_holidays = holidays.BE(years = [2017, 2018, 2019, 2020, 2021])
    fr_holidays = holidays.FR(years = [2017, 2018, 2019, 2020, 2021])
    de_holidays = holidays.DE(years = [2017, 2018, 2019, 2020, 2021])
    it_holidays = holidays.IT(years = [2017, 2018, 2019, 2020, 2021])
    pl_holidays = holidays.PL(years = [2017, 2018, 2019, 2020, 2021])
    es_holidays = holidays.ES(years = [2017, 2018, 2019, 2020, 2021])

    train_list = list()
    test_list = list()
    countries = ['Belgium', 'France', 'Germany', 'Italy', 'Poland', 'Spain']
    black_fridays = [pd.to_datetime('2017-11-24'), pd.to_datetime('2018-11-23'), pd.to_datetime('2019-11-29'), pd.to_datetime('2020-11-27'), pd.to_datetime('2021-11-26')]

    for i in range(0, len(countries)):
    
        train_temp = train[train['country'] == countries[i]].reset_index(drop = True)
        train_temp['is_holiday'] = np.nan
        train_temp['holiday_season'] = np.nan
        train_temp['black_friday'] = 0
     
        test_temp = test[test['country'] == countries[i]].reset_index(drop = True)
        test_temp['is_holiday'] = np.nan
        test_temp['holiday_season'] = np.nan
        test_temp['black_friday'] = 0
        
        if (i == 0):
        
            holiday_to_use = be_holidays
        
        elif (i == 1):

            holiday_to_use = fr_holidays

        elif (i == 2):

            holiday_to_use = de_holidays

        elif (i == 3):

            holiday_to_use = it_holidays

        elif (i == 4):

            holiday_to_use = pl_holidays

        else:

            holiday_to_use = es_holidays
    
        for j in range(0, train_temp.shape[0]):

            train_temp['is_holiday'][j] = np.where(train_temp['date'][j] in holiday_to_use, 1, 0)
            
            train_temp['black_friday'][j] = np.where(np.isin(train_temp['date'][j], black_fridays), 1, 0)
            
            
            if ((train_temp['date'][j] >= pd.to_datetime('2017-12-01')) & (train_temp['date'][j] <= pd.to_datetime('2017-12-31'))):

                train_temp['holiday_season'][j] = 1

            elif ((train_temp['date'][j] >= pd.to_datetime('2018-12-01')) & (train_temp['date'][j] <= pd.to_datetime('2018-12-31'))):    

                train_temp['holiday_season'][j] = 1

            elif ((train_temp['date'][j] >= pd.to_datetime('2019-12-01')) & (train_temp['date'][j] <= pd.to_datetime('2019-12-31'))):      

                train_temp['holiday_season'][j] = 1

            elif ((train_temp['date'][j] >= pd.to_datetime('2020-12-01')) & (train_temp['date'][j] <= pd.to_datetime('2020-12-31'))):

                train_temp['holiday_season'][j] = 1

            else:

                train_temp['holiday_season'][j] = 0

        train_list.append(train_temp)
    
        for k in range(0, test_temp.shape[0]):

            test_temp['is_holiday'][k] = np.where(test_temp['date'][k] in holiday_to_use, 1, 0)

            test_temp['black_friday'][k] = np.where(np.isin(test_temp['date'][k], black_fridays), 1, 0)
            
            if ((test_temp['date'][k] >= pd.to_datetime('2017-12-01')) & (test_temp['date'][k] <= pd.to_datetime('2017-12-31'))):

                test_temp['holiday_season'][j] = 1

            elif ((test_temp['date'][k] >= pd.to_datetime('2018-12-01')) & (test_temp['date'][k] <= pd.to_datetime('2018-12-31'))):    

                test_temp['holiday_season'][j] = 1

            elif ((test_temp['date'][k] >= pd.to_datetime('2019-12-01')) & (test_temp['date'][k] <= pd.to_datetime('2019-12-31'))):      

                test_temp['holiday_season'][j] = 1

            elif ((test_temp['date'][k] >= pd.to_datetime('2020-12-01')) & (test_temp['date'][k] <= pd.to_datetime('2020-12-31'))):

                test_temp['holiday_season'][k] = 1

            else:

                test_temp['holiday_season'][k] = 0

        test_list.append(test_temp)
    
    ## Putting train and test in the right format
    train = pd.concat(train_list)
    train['is_holiday'] = train['is_holiday'].astype(int)
    train['holiday_season'] = train['holiday_season'].astype(int)

    test = pd.concat(test_list)
    test['is_holiday'] = test['is_holiday'].astype(int)
    test['holiday_season'] = test['holiday_season'].astype(int)    
    
    return [train, test]