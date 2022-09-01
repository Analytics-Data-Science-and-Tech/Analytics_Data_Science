import pandas as pd
import numpy as np
from Run_Ensemble_Help_Funs import Run_Ensemble


#########################
## Validation Datasets ##
#########################

val_1 = pd.read_csv('Evan_results_validation.csv')
val_1.columns = ['row_id', 'time', 'x', 'y', 'direction', 'congestion', 'congestion_pred_1']

val_2 = pd.read_csv('Gabriel_Ferreira_results_validation.csv')
val_2 = val_2[['row_id', 'congestion_pred']]
val_2.columns = ['row_id', 'congestion_pred_2']

val_3 = pd.read_csv('Gabriel_De_Medeiros_results_validation.csv')
val_3.columns = ['row_id', 'congestion_pred_3']

val_4 = pd.read_csv('Oscar_results_validation.csv')
val_4 = val_4[['row_id', 'congestion_pred']]
val_4.columns = ['row_id', 'congestion_pred_4']

validation = pd.merge(val_1, val_2, on = ['row_id'], how = 'left')
validation = pd.merge(validation, val_3, on = ['row_id'], how = 'left')
validation = pd.merge(validation, val_4, on = ['row_id'], how = 'left')

###################
## Test Datasets ##
###################

test_1 = pd.read_csv('Evan_results_test.csv')
test_1.columns = ['row_id', 'time', 'x', 'y', 'direction', 'congestion_pred_1']

test_2 = pd.read_csv('Gabriel_Ferreira_results_test.csv')
test_2 = test_2[['row_id', 'congestion_pred']]
test_2.columns = ['row_id', 'congestion_pred_2']

test_3 = pd.read_csv('Gabriel_De_Medeiros_results_test.csv')
test_3.columns = ['row_id', 'congestion_pred_3']

test_4 = pd.read_csv('Oscar_results_test.csv')
test_4 = test_4[['row_id', 'congestion_pred']]
test_4.columns = ['row_id', 'congestion_pred_4']

test = pd.merge(test_1, test_2, on = ['row_id'], how = 'left')
test = pd.merge(test, test_3, on = ['row_id'], how = 'left')
test = pd.merge(test, test_4, on = ['row_id'], how = 'left')

## Modeling 
results = Run_Ensemble(validation, test, 'RF')

## Storing results
results[0].to_csv('ensemble_results_validation.csv', index = False)
results[1].to_csv('ensemble_results_test.csv', index = False)