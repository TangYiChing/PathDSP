def preprocess(params):
    fname='input_for_Nick.txt'
    origin=params['data_url']
    # Download and unpack the data in CANDLE_DATA_DIR
    candle.file_utils.get_file(fname, origin)
    params['train_data'] = os.environ['CANDLE_DATA_DIR'] + '/common/Data/'+params['train_data']
    #params['val_data'] = os.environ['CANDLE_DATA_DIR'] + '/common/Data/'+params['val_data']
    #params['gep_filepath'] = os.environ['CANDLE_DATA_DIR'] + '/common/Data/'+params['gep_filepath']
    #params['smi_filepath'] = os.environ['CANDLE_DATA_DIR'] + '/common/Data/'+params['smi_filepath']
    #params['gene_filepath'] = os.environ['CANDLE_DATA_DIR'] + '/common/Data/'+params['gene_filepath']
    #params['smiles_language_filepath'] = os.environ['CANDLE_DATA_DIR'] + '/common/Data/'+params['smiles_language_filepath']
    """ 
    params["train_data"] = candle.get_file(params['train_data'], origin, datadir=params['data_dir'], cache_subdir=None)
    params["val_data"] = candle.get_file(params['val_data'], origin, datadir=params['data_dir'], cache_subdir=None)
    params["gep_filepath"] = candle.get_file(params['gep_filepath'], origin, datadir=params['data_dir'], cache_subdir=None)
    params["smi_filepath"] = candle.get_file(params['smi_filepath'], origin, datadir=params['data_dir'], cache_subdir=None)
    params["gene_filepath"] = candle.get_file(params['gene_filepath'], origin, datadir=params['data_dir'], cache_subdir=None)
    params["smiles_language_filepath"] = candle.get_file(params['smiles_language_filepath'], origin, datadir=params['data_dir'], cache_subdir=None) """
    return params

def run(params):
    params['data_type'] = str(params['data_type'])
    with open ((params['output_dir']+'/params.json'), 'w') as outfile:
        json.dump(params, outfile)
    scores = main(params)
    with open(params['output_dir'] + "/scores.json", "w", encoding="utf-8") as f:
        json.dump(scores, f, ensure_ascii=False, indent=4)
    print('IMPROVE_RESULT RMSE:\t' + str(scores['rmse']))

def candle_main():
    params = initialize_parameters()
    params = preprocess(params)
    run(params)

if __name__ == "__main__":
    candle_main()
