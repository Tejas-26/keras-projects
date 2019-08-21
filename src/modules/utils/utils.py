from logs import logDecorator as lD
import jsonref
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import csv

from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split as tts
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.datasets import load_files

from keras import backend as K
from keras.models import Model, Sequential
from keras.layers.core import Activation
from keras.layers import Input, Embedding, Flatten, dot, Dense
from keras.optimizers import Adam

from psycopg2.sql import SQL, Identifier, Literal
from lib.databaseIO import pgIO
from collections import Counter
from textwrap import wrap


config = jsonref.load(open('../config/config.json'))
logBase = config['logging']['logBase'] + '.modules.utils.utils'
t1_config = jsonref.load(open('../config/tejasT1.json'))

@lD.log(logBase + '.createDF_allRaces_anySUD')
def createDF_allRaces_anySUD(logger):
    '''Creates dataframe for total sample, dependent variable = any sud

    This function creates a dataframe for the total sample, where the
    dependent variable is any sud and the independent variables are:
    race, age, sex and setting.

    Decorators:
        lD.log

    Arguments:
        logger {logging.Logger} -- logs error information
    '''

    try:

        query = '''
        SELECT * from tejas.restofusers_t3_p1
        '''
        data = pgIO.getAllData(query)
        sud_data = [d[0] for d in data]
        race_data = [d[1] for d in data]
        age_data = [d[2] for d in data]
        sex_data = [d[3] for d in data]
        setting_data = [d[4] for d in data]

        d = {'sud': sud_data, 'race': race_data, 'age': age_data, 'sex': sex_data, 'setting': setting_data}

        main = pd.DataFrame(data=d)
        df = main.copy()

        # Change sud column to binary, dummify the other columns
        df.replace({False:0, True:1}, inplace=True)

        dummy_races = pd.get_dummies(main['race'])
        df = df[['sud']].join(dummy_races.ix[:, 'MR':])
        main.replace(to_replace=list(range(12, 18)), value="12-17", inplace=True)
        main.replace(to_replace=list(range(18, 35)), value="18-34", inplace=True)
        main.replace(to_replace=list(range(35, 50)), value="35-49", inplace=True)
        main.replace(to_replace=list(range(50, 100)), value="50+", inplace=True)
        dummy_ages = pd.get_dummies(main['age'])
        df = df[['sud', 'MR', 'NHPI']].join(dummy_ages.ix[:, :'50+'])
        dummy_sexes = pd.get_dummies(main['sex'])
        df = df[['sud', 'MR', 'NHPI', '12-17', '18-34', '35-49', '50+']].join(dummy_sexes.ix[:, 'M':])
        dummy_setting = pd.get_dummies(main['setting'])
        df = df[['sud', 'MR', 'NHPI', '12-17', '18-34', '35-49', 'M']].join(dummy_setting.ix[:, :'Inpatient'])
        df['intercept'] = 1.0

    except Exception as e:
        logger.error('createDF_allRaces_anySUD failed because of {}'.format(e))
    return df

@lD.log(logBase + '.logRegress')
def nnClassify(logger, layers):
    '''Performs classification with hidden layer NN
    Decorators:
        lD.log

    Arguments:
        logger {logging.Logger} -- logs error information
        df {dataframe} -- input dataframe where first column is 'sud'
    '''
    try:
        print("Performing classification with hidden layer NN...")
        query = '''
        SELECT * from tejas.restofusers_t3_p1
        '''
        data = pgIO.getAllData(query)#returns list of tuples (T/F,.......)
        csvfile = '../data/firstThouSUDorNah.csv'
        with open(csvfile,'w+') as f:
            csv_out=csv.writer(f)
            csv_out.writerow(['sud','race','age','sex','setting'])
            csv_out.writerows(data)
        f.close()
        dataset = pd.read_csv(csvfile)
        # print(dataset)
        X = dataset.iloc[:,1:].values #X now takes everything but sud
        y = dataset.iloc[:,0].values #sud saved for y
        lab_enc_race = LabelEncoder()
        X[:,0] = lab_enc_race.fit_transform(X[:,0])
        lab_enc_sex = LabelEncoder()
        X[:,2] = lab_enc_sex.fit_transform(X[:,2])
        lab_enc_setting = LabelEncoder()
        X[:,3] = lab_enc_setting.fit_transform(X[:,3])
        # sex and setting are binary variables
        # must create dummy variable for race since race = 'aa', 'nhpi' or 'mr'
        onehotencoder = OneHotEncoder(categorical_features = [0])
        X = onehotencoder.fit_transform(X).toarray()
        #80-20 train-test split
        X_train, X_test, y_train, y_test = tts(X, y, test_size = 0.2)
        #feature scaling
        sc = StandardScaler()
        X_train = sc.fit_transform(X_train)
        X_test = sc.transform(X_test)

        #Initializing Neural Network
        classifier = Sequential()
        buildClassifier(classifier, layers)
        # # Adding the output layer
        # classifier.add(Dense(output_dim = 1, init = 'uniform',
        #                      activation = 'sigmoid'))
        # Compiling Neural Network
        classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy',
                             metrics = ['accuracy'])
        # Fitting our model
        # Number of epochs and batch sizes are hyperparameters
        history = classifier.fit(X_train, y_train, epochs = 40,
                                 verbose = 0, batch_size = 100,
                                 validation_data=(X_test,y_test))
        trainL = history.history['loss']
        testL = history.history['val_loss']
        epoch_count = range(1,1+len(trainL))

        plt.plot(epoch_count,trainL,'r--')
        plt.plot(epoch_count,testL,'b-')
        plt.legend(['Training Loss','Test Loss'])
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.show()
        # Predicting the Test set results
        y_pred = classifier.predict(X_test)
        y_pred = (y_pred > 0.5)
        # # Creating the Confusion Matrix
        acc = accuracy_score(y_test, y_pred)
        print('\n%.2f'%(100*acc))
        # accuracy = (cm[0][0]+cm[1][1])/(cm[0][0]+cm[0][1]+cm[1][0]+cm[1][1])
        # accuracy *=100
    except Exception as e:
        logger.error('logRegress failed because of {}'.format(e))
    return

@lD.log(logBase + '.createDF_allRaces_morethan2SUD')
def createDF_allRaces_morethan2SUD(logger):
    '''Creates dataframe for total sample, dependent variable = at least 2 sud

    This function creates a dataframe for the total sample, where the
    dependent variable is >=2 sud and the independent variables are:
    race, age, sex and setting.

    Decorators:
        lD.log

    Arguments:
        logger {logging.Logger} -- logs error information
    '''

    try:

        query = '''
        SELECT morethan2sud, race, age, sex, visit_type
        FROM tejas.sud_race_age
        WHERE age BETWEEN 12 AND 100
        '''

        data = pgIO.getAllData(query)
        sud_data = [d[0] for d in data]
        race_data = [d[1] for d in data]
        age_data = [d[2] for d in data]
        sex_data = [d[3] for d in data]
        setting_data = [d[4] for d in data]

        d = {'sud': sud_data, 'race': race_data, 'age': age_data, 'sex': sex_data, 'setting': setting_data}
        main = pd.DataFrame(data=d)
        df = main.copy()

        # Change sud column to binary, dummify the other columns
        df.replace({False:0, True:1}, inplace=True)

        dummy_races = pd.get_dummies(main['race'])
        df = df[['sud']].join(dummy_races.ix[:, 'MR':])

        main.replace(to_replace=list(range(12, 18)), value="12-17", inplace=True)
        main.replace(to_replace=list(range(18, 35)), value="18-34", inplace=True)
        main.replace(to_replace=list(range(35, 50)), value="35-49", inplace=True)
        main.replace(to_replace=list(range(50, 100)), value="50+", inplace=True)
        dummy_ages = pd.get_dummies(main['age'])
        df = df[['sud', 'MR', 'NHPI']].join(dummy_ages.ix[:, :'35-49'])

        dummy_sexes = pd.get_dummies(main['sex'])
        df = df[['sud', 'MR', 'NHPI', '12-17', '18-34', '35-49']].join(dummy_sexes.ix[:, 'M':])

        dummy_setting = pd.get_dummies(main['setting'])
        df = df[['sud', 'MR', 'NHPI', '12-17', '18-34', '35-49', 'M']].join(dummy_setting.ix[:, :'Inpatient'])

        df['intercept'] = 1.0

    except Exception as e:
        logger.error('createDF_allRaces_morethan2SUD failed because of {}'.format(e))

    return df

@lD.log(logBase + '.doSomeShit')
def buildClassifier(logger, classifier, n):
    if n==1:
        # one hidden layer
        classifier.add(Dense(output_dim = 12, init = 'uniform',
                             activation = 'relu', input_dim = 6))
        classifier.add(Dense(output_dim = 1, init = 'uniform',
                             activation = 'sigmoid'))
    elif n==2:
        # two hidden layers
        classifier.add(Dense(output_dim = 8, init = 'uniform',
                             activation = 'relu', input_dim = 6))
        classifier.add(Dense(output_dim = 12, init = 'uniform',
                             activation = 'sigmoid'))
        classifier.add(Dense(output_dim = 1, init = 'uniform',
                             activation = 'sigmoid'))
    elif n==3:
        # two hidden layers
        classifier.add(Dense(output_dim = 18, init = 'uniform',
                             activation = 'relu', input_dim = 6))
        classifier.add(Dense(output_dim = 12, init = 'uniform',
                             activation = 'sigmoid'))
        classifier.add(Dense(output_dim = 6, init = 'uniform',
                             activation = 'sigmoid'))
        classifier.add(Dense(output_dim = 1, init = 'uniform',
                             activation = 'sigmoid'))
    else:
        classifier.add(Dense(output_dim = 1, init = 'uniform',
                             activation = 'sigmoid', input_dim = 6))
    return

@lD.log(logBase + '.load_imdb_data')
def load_imdb_data(logger, datadir):
    # read in training and test corpora
    categories= ['pos', 'neg']
    print("yeet 1")
    train_b = load_files(datadir+'/train', shuffle=True,
                         categories=categories)
    print("yeet 2")
    test_b = load_files(datadir+'/test', shuffle=True,
                         categories=categories)
    train_b.data = [x.decode('utf-8') for x in train_b.data]
    test_b.data =  [x.decode('utf-8') for x in test_b.data]
    veczr =  CountVectorizer(ngram_range=(1,3), binary=True,
                             token_pattern=r'\w+',
                             max_features=800000)
    dtm_train = veczr.fit_transform(train_b.data)
    dtm_test = veczr.transform(test_b.data)
    y_train = train_b.target
    y_test = test_b.target
    print("DTM shape (training): (%s, %s)" % (dtm_train.shape))
    print("DTM shape (test): (%s, %s)" % (dtm_train.shape))
    num_words = len([v for k,v in veczr.vocabulary_.items()]) + 1
    print('vocab size:%s' % (num_words))

    return (dtm_train, dtm_test), (y_train, y_test), num_words
