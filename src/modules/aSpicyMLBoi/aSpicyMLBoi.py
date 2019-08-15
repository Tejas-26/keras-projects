import jsonref
from numpy import loadtxt
from keras.models import Sequential
from keras.layers import Dense

config = jsonref.load(open('../config/config.json'))
logBase = config['logging']['logBase'] + '.modules.aSpicyMLBoi.aSpicyMLBoi'

# @lD.log(logBase + '.main')
def main(logger, resultDict):
    '''
    main function for aSpicyMLBoi

    This function finishes all the tasks for the
    main function. This is a way in which a
    particular module is going to be executed.

    Parameters
    ----------
    logger : {logging.Logger}
        The logger used for logging error information
    resultsDict: {dict}
        A dintionary containing information about the
        command line arguments. These can be used for
        overwriting command line arguments as needed.
    '''
    print('='*30)
    print('Main function of aSpicyMLBoi module')
    print('='*30)
    #load the dataset
    dataset = loadtxt('../data/raw_data/diabetes.csv', delimiter=',')
    X = dataset[:,0:8]
    y = dataset[:,8]

    # define the keras model
    model = Sequential()
    model.add(Dense(12, input_dim=8, activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    # compile the keras model
    # adam stochastic gradient descent algorithm automatically tunes itself
    # SGD: iterative
    # gives good results in large range of problems
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    # fit keras model on dataset, define hyperparameters
    # epoch = one pass throuh all training dataset rows
    # batch = at least one sample that model considers within an epoch before updating weights
    # in this case
    model.fit(X,y,epochs=150,batch_size=10) # 15 batches here
    _, accuracy = model.evaluate(X,y) # determine accuracy
    print('The boi has an accuracy of %.2f'%(accuracy*100))
    print('Getting out of aSpicyMLBoi module')
    print('-'*30)
    return
