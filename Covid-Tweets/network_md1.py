import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.multiclass import OneVsRestClassifier
from sklearn.ensemble import RandomForestClassifier

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

test_id = test['Id']
test = test.drop(columns = ['Id', 'text', 'reply_to_screen_name', 'hashtags'], axis = 1)

## Defining input and target
X = train.drop(columns = ['text', 'reply_to_screen_name', 'hashtags', 'country'], axis = 1)
Y = train['country']
Y = np.where(Y == 'us', 0, 
             np.where(Y == 'uk', 1, 
                      np.where(Y == 'canada', 2, 
                               np.where(Y == 'australia', 3,
                                        np.where(Y == 'ireland', 4, 5)))))

## Splitting the data 
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size = 0.2)

## Scaling the data 
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)
test = scaler.fit_transform(test)

## Defining model 
model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(10, input_dim =  15, activation = 'relu'),
#         tf.keras.layers.Dense(10, input_dim =  15, activation = 'relu'),
        tf.keras.layers.Dense(6, activation = 'softmax')
])

model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

## Fitting model 
history = model.fit(X_train, tf.keras.utils.to_categorical(Y_train, num_classes = 6), epochs = 10, batch_size = 32, validation_data = (X_test, tf.keras.utils.to_categorical(Y_test, num_classes = 6)))

## Predicting on test
nn_pred = model.predict(test)
nn_pred = np.argmax(nn_pred, axis = 1)

## Defining data to be exported
data_out = pd.DataFrame({'Id': test_id, 'Category': nn_pred})
data_out['Category'] = np.where(data_out['Category'] == 0, 'us',
                                np.where(data_out['Category'] == 1, 'uk',
                                         np.where(data_out['Category'] == 2, 'canada',
                                                  np.where(data_out['Category'] == 3, 'australia',
                                                           np.where(data_out['Category'] == 4, 'ireland', 'new_zealand')))))
data_out.to_csv('nn_submission_md1.csv', index = False)