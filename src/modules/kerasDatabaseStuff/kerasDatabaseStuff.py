from logs import logDecorator as lD
import jsonref
from numpy import loadtxt
from keras.models import Sequential
from keras.layers import Dense
from modules.utils import utils as u

config = jsonref.load(open('../config/config.json'))
logBase = config['logging']['logBase'] + '.modules.kerasDatabaseStuff.kerasDatabaseStuff'

@lD.log(logBase + '.main')
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
    print('Main function of kerasDatabaseStuff module')
    print('='*30)
    #nnclassify(number of first HL's nodes, snumber of 2nd HL's nodes, test size)
    acc = u.nnClassify(9,6,0.3)
    print(acc)
    print('Getting out of kerasDatabaseStuff module')
    print('-'*30)
    return
