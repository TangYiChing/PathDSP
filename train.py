import candle
import os
import sys
#import json
#from json import JSONEncoder
from preprocess_new import mkdir, preprocess
#sys.path.append("/usr/local/PathDSP/PathDSP")
sys.path.append("/usr/local/PathDSP/PathDSP")
sys.path.append(os.getcwd() + "/PathDSP")
import FNN_new

file_path = os.path.dirname(os.path.realpath(__file__))
# This should be set outside as a user environment variable
#os.environ['CANDLE_DATA_DIR'] = os.environ['HOME'] + '/improve_data_dir/'
required = None
additional_definitions = None

# initialize class
class PathDSP_candle(candle.Benchmark):
    def set_locals(self):
        '''
        Functionality to set variables specific for the benchmark
        - required: set of required parameters for the benchmark.
        - additional_definitions: list of dictionaries describing the additional parameters for the benchmark.
        '''
        if required is not None: 
            self.required = set(required)
        if additional_definitions is not None:
            self.additional_definitions = additional_definitions

def initialize_parameters():
    preprocessor_bmk = PathDSP_candle(file_path,
        'PathDSP_params.txt',
        'pytorch',
        prog='PathDSP_candle',
        desc='Data Preprocessor'
    )
    #Initialize parameters
    gParameters = candle.finalize_parameters(preprocessor_bmk)
    return gParameters

# class CustomData:
#     def __init__(self, name, value):
#         self.name = name
#         self.value = value

# class CustomEncoder(json.JSONEncoder):
#     def default(self, o):
#             return o.__dict__


# def run(params):
#     params['data_type'] = str(params['data_type'])
#     json_out = params['output_dir']+'/params.json'
#     print(params)

#     with open (json_out, 'w') as fp:
#         json.dump(params, fp, indent=4, cls=CustomEncoder)

#     scores = main(params)
#     with open(params['output_dir'] + "/scores.json", "w", encoding="utf-8") as f:
#         json.dump(scores, f, ensure_ascii=False, indent=4)
# #    print('IMPROVE_RESULT RMSE:\t' + str(scores['rmse']))


def candle_main():
    params = initialize_parameters()
    data_dir = os.environ['CANDLE_DATA_DIR'] + '/' + '/Data/'
    params =  preprocess(params, data_dir)
    FNN_new.main(params)


if __name__ == "__main__":
    candle_main()
