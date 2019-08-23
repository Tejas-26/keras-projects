from logs import logDecorator as lD
import jsonref
import numpy as np
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
    # u.nnClassify(1)
    (dtm_train, dtm_test), (y_train, y_test), num_words = u.load_imdb_data(PATH_TO_IMDB)
    print(num_words)
    maxlen = 2000
    x_train = u.dtm2wid(dtm_train, maxlen)
    x_test = u.dtm2wid(dtm_test, maxlen)
    nbratios = np.log(u.pr(dtm_train, y_train, 1)/u.pr(dtm_train,
                                               y_train, 0))
    nbratios = np.squeeze(np.asarray(nbratios))
    model = u.get_model(num_words, maxlen, nbratios=nbratios)
    model.fit(x_train, y_train,
              batch_size=32,
              epochs=3,
              validation_data=(x_test, y_test))
    print('Getting out of kerasDatabaseStuff module')
    print('-'*30)
    return
