from logs import logDecorator as lD
import jsonref
from modules.utils import utils as u

config = jsonref.load(open('../config/config.json'))
logBase = config['logging']['logBase'] + '.modules.kerasDatabaseStuff.kerasDatabaseStuff'
PATH_TO_IMDB = r'../data/aclImdb'

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
    u.doSomeShit()
    # dict_accs = {}
    # #(number of first HL's nodes, number of 2nd HL's nodes, test size, epochs)
    # for i in range(10,100,10):
    #     acc = u.nnClassify(9,6,0.2,i)
    #     dict_accs['{}'.format(i)] = acc
    # print(dict_accs)
    # (dtm_train, dtm_test), (y_train, y_test), num_words = u.load_imdb_data(PATH_TO_IMDB)
    print('Getting out of kerasDatabaseStuff module')
    print('-'*30)
    return
