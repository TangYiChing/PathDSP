#!/bin/bash

#The point of this is to test if the thing works at all

python get_test_data.py
python ./PathDSP/FNN.py -i tmp/common/input_txt_Nick.txt -o ./output_prefix

