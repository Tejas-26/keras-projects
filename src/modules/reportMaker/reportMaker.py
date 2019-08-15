from logs import logDecorator as lD
import jsonref

config = jsonref.load(open('../config/config.json'))
logBase = config['logging']['logBase'] + '.modules.reportMaker.reportMaker'

@lD.log(logBase + '.main')
def main(logger, resultDict):
    '''
    main function for module1

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
    print('Main function of reportMaker module')
    print('='*30)

    print('Getting out of reportMaker module')
    print('-'*30)
    return
