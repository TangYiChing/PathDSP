import candle
import os

# Assumes CANDLE_DATA_DIR is an environment variable
os.environ['CANDLE_DATA_DIR'] = '/tmp/data_dir'

fname='input_txt_Nick.txt'
origin='http://chia.team/IMPROVE_data/input_txt_Nick.txt'

# Download and unpack the data in CANDLE_DATA_DIR
candle.file_utils.get_file(fname, origin)

# Do it again to confirm it's not re-downloading
candle.file_utils.get_file(fname, origin)
