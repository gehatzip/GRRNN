import sys


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



def mode_cmd_arg():
    modes_all = ['optimize', 'train', 'test']

    mode = modes_all[0]

    if '--mode' in sys.argv:

        index = sys.argv.index('--mode')

        if index < len(sys.argv)-1:
            if sys.argv[index+1] in modes_all:
                mode = sys.argv[index+1]

    print('mode: ' + mode)

    return mode