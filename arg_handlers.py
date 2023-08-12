import sys

def model_is_singlestep(model_type):
    single_step_models = ['RNN', 'RNNPROB', 'MLP', 'LSTNET']
    return (model_type in single_step_models)

def model_type_cmd_arg():

    model_types = ['RNN', 'RNNPROB', 'MLP', 'S2S', 'S2SGEN', 'S2SATTN', 'S2SATTNGEN', 'DARNN', 'DARNNGEN', 'DARNNAF', 'DARNNAFGEN', 'GARNN', 'LSTNET', 'DSTPRNN', 'STEMGNN']

    model_type = model_types[0]

    if '--model' in sys.argv:

        index = sys.argv.index('--model')

        if index < len(sys.argv)-1:
            if sys.argv[index+1] in model_types:
                model_type = sys.argv[index+1]

    print('model_type: ' + model_type)

    return model_type


def predict_type_cmd_arg():

    predict_types = ['forecast', 'classify']

    predict_type = predict_types[0]

    if '--predict' in sys.argv:

        index = sys.argv.index('--predict')

        if index < len(sys.argv)-1:
            if sys.argv[index+1] in predict_types:
                predict_type = sys.argv[index+1]

    print('predict_type: ' + predict_type)

    return predict_type


def desc_cmd_arg():

    desc_all = ['All', 'polarity', 'score']

    desc = desc_all[0]

    if '--desc' in sys.argv:

        index = sys.argv.index('--desc')

        if index < len(sys.argv)-1:
            if sys.argv[index+1] in desc_all:
                desc = sys.argv[index+1]

    print('desc: ' + desc)

    return desc


def party_cmd_arg():

    party_all = ['All', 'Rep', 'Dem']

    party = party_all[0]

    if '--party' in sys.argv:

        index = sys.argv.index('--party')

        if index < len(sys.argv)-1:
            if sys.argv[index+1] in party_all:
                party = sys.argv[index+1]

    print('party: ' + party)

    return party

def horizon_cmd_arg():

    horizon = 1

    if '--horizon' in sys.argv:

        index = sys.argv.index('--horizon')

        if index < len(sys.argv)-1:
            if sys.argv[index+1].isnumeric():
                h = int(sys.argv[index+1])
                if h>0:
                    horizon = h

    print('horizon: ' + str(horizon))

    return horizon

    

def dataset_cmd_arg():

    dataset_all = ['ETD','SML2010','AirQuality','energyco','NASDAQ100', 'poll']

    dataset = dataset_all[0]

    if '--dataset' in sys.argv:

        index = sys.argv.index('--dataset')

        if index < len(sys.argv)-1:
            if sys.argv[index+1] in dataset_all:
                dataset = sys.argv[index+1]

    print('dataset: ' + dataset)

    return dataset