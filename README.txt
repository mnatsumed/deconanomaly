# Data sets
## SMAP and MSL
Please read description in
https://github.com/khundman/telemanom

## SMD
Please read description in
https://github.com/NetManAIOps/OmniAnomaly

## PAMAP2
Please read description in
http://archive.ics.uci.edu/ml/datasets/PAMAP2+Physical+Activity+Monitoring

## VAM
This data set is collected from a vinyl acetate monomer production plant simulator simulating a procedure of start-up operations.
It consists of five stages as follows.
1. Starting Column & Absorber
2. Ethylene make up
3. Starting the vaporizer
4. Connecting lines between reactor and column
5. O2 Feed
An checkpoint exist at the end of the stage.

For more detail, please read manuals for Visual Modeler Trial Version.
https://www.omegasim.co.jp/contents_e/product/vm/trial/

# Data collection and preparation
## SMAP and MSL
1. Download data.zip following the instruction in https://github.com/khundman/telemanom
2. Download https://github.com/khundman/telemanom/blob/master/labeled_anomalies.csv
3. Unzip data.zip and place train and test under DeconAnomaly/test_data/JPL
4. Place labeled_anomalies.csv under DeconAnomaly/test_data/JPL

## SMD
1. Download https://github.com/NetManAIOps/OmniAnomaly/tree/master/ServerMachineDataset
2. Place ServerMachineDataset under DeconAnomaly/test_data

## PAMAP2
1. Download https://archive.ics.uci.edu/ml/machine-learning-databases/00231/PAMAP2_Dataset.zip
2. Unzip PAMAP2_Dataset.zip
3. Move all files in Protocol to DeconAnomaly/test_data/PAMAP2
4. Add the suffix, _op, to each file name in Optional, e.g. subject101_op.dat
5. Move all files in Optional to DeconAnomaly/test_data/PAMAP2

# install dependencies with Python 3.6
pip3 install -r requirements.txt

# Processing
1. python3 ./prep.py
2. python3 ./prep_pub_data.py

# Output directories
## SMAP
training set: ./test_data/smap_tr
test set: ./test_data/smap_ts

## MSL
training set: ./test_data/msl_tr
test set: ./test_data/msl_ts

## SMD
training set: ./test_data/smd_tr
test set: ./test_data/smd_ts

## PAMAP2
training set: ./test_data/pamap2_tr
test set: ./test_data/pamap2_ts

## VAM
training set: ./test_data/vam_tr
test set: ./test_data/vam_ts

# Structure of the outputs
Each directory contains a directory for each region.
The directry name follows the rule, namely sensorID_operatingID. The sensorID is the index of the sensor group. The operatingID is the index of the operating modes.
A directory for each region contains CSV files and text for labels. As for SMAP, MSL and SMD, they are 0.csv and label.txt. As for PAMAP2 and VAM, they are i.csv and label_i.txt, where i is the index of the time series segment.
