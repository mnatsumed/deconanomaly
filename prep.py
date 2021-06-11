import numpy as np
import pandas as pd

import glob

from tqdm import tqdm
import os

def output_sub_dir(output_dir, seg_idx, sensor_g_idx):
    path_ = os.path.join(output_dir, str(sensor_g_idx)+'___'+str(seg_idx))
    os.makedirs(path_ , exist_ok=True)
    return path_

def data_preprocessing(tss_csv_list, sensor_group_id, output_dir):
    for k, _tss_csv_path in tqdm(enumerate(tss_csv_list), total=len(tss_csv_list)):
        df = pd.read_csv(_tss_csv_path, header=0)
        event_time_table_path = get_event_time_table_path(_tss_csv_path)
        event_time_table = np.loadtxt(event_time_table_path)
        for i in range(len(event_time_table)-1):
            s_time = event_time_table[i]
            e_time = event_time_table[i+1]

            if s_time < e_time:
                _mask = (df['Time'] >= s_time) & (df['Time'] < e_time)
                df_seg = df[_mask]
                for j, sensor_g_id in enumerate(sensor_group_id):
                    selected_column_idx = [0]
                    selected_column_idx.extend(sensor_g_id)
                    selected_column_name = df_seg.columns[selected_column_idx]
                    output_path = os.path.join(output_sub_dir(output_dir, i, j), str(k)+'.csv')
                    df_seg.to_csv(output_path, columns=selected_column_name, index=False)

def get_event_time_table_path(tss_csv_path):
    f_name = os.path.splitext(os.path.basename(tss_csv_path))[0]+'_time.txt'
    return os.path.join(os.path.dirname(tss_csv_path), f_name)

def add_labels_all_normal(data_dir, n_normal_trajectories):
    dirs = os.listdir(data_dir)
    dirs = [os.path.join(data_dir, f) for f in dirs if os.path.isdir(os.path.join(data_dir, f))]

    for _dir in dirs:
        for i in range(n_normal_trajectories):
            _path_tss = os.path.join(_dir, str(i)+'.csv')
            _df_seg = pd.read_csv(_path_tss)
            # for label
            _out_path_label = os.path.join(_dir, 'label_'+str(i)+'.txt')
            _n_samples_ts = _df_seg.shape[0]
            _label = np.zeros((_n_samples_ts, 1))
            np.savetxt(_out_path_label, _label, delimiter=',')

def data_preprocessing_period(examples, sensor_group_id, output_dir, initi_index_dict):
    for _tss_path, _target_period, _abnormal_period, _target_seg_id in tqdm(examples, total=len(examples)):
        current_index = initi_index_dict[str(_target_seg_id)]

        df = pd.read_csv(_tss_path, header=0)

        s_time = _target_period[0]
        e_time = _target_period[1]
        assert s_time < e_time

        s_time_ab = _abnormal_period[0]
        e_time_ab = _abnormal_period[1]
        assert s_time_ab < e_time_ab
        assert s_time_ab >= s_time
        assert e_time_ab <= e_time

        _mask = (df['Time'] >= s_time) & (df['Time'] < e_time)
        df_seg = df[_mask]
        _time_stamps = df_seg['Time'].values
        _label = np.zeros((len(_time_stamps), 1))
        _label[(_time_stamps>=s_time_ab) & (_time_stamps<=e_time_ab)] = 1.0
        for j, sensor_g_id in enumerate(sensor_group_id):
            selected_column_idx = [0]
            selected_column_idx.extend(sensor_g_id)
            selected_column_name = df_seg.columns[selected_column_idx]
            _dir = output_sub_dir(output_dir, _target_seg_id, j)
            output_path = os.path.join(_dir, str(current_index)+'.csv')
            df_seg.to_csv(output_path, columns=selected_column_name, index=False)

            # for label
            _out_path_label = os.path.join(_dir, 'label_'+str(current_index)+'.txt')
            _n_samples_ts = df_seg.shape[0]
            np.savetxt(_out_path_label, _label, delimiter=',')
        initi_index_dict[str(_target_seg_id)] += 1

def to_hours(hh, mm, ss):
    return hh + mm/60.0 + ss/60.0/60.0

if __name__ == '__main__':
    SENSOR_GROUP_ID = [1, 2, 3, 4, 5, 6, 7]
    n_segments = 5

    sensor_info_path = './test_data/VAM/sensor_groups_wo_sw.csv'
    sensors_info = pd.read_csv(sensor_info_path, header=0)
    sensor_groups = sensors_info.groupby('group_index')
    sensor_group_id = []
    for i in SENSOR_GROUP_ID:
        s_group_id = sensor_groups.get_group(i)['index'].values.tolist()
        sensor_group_id.append(s_group_id)

    ## training data
    output_dir_tr = './test_data/vam_tr'
    tss_csv_path1 = './test_data/VAM/VMctrl_startup1.csv'
    tss_csv_path2 = './test_data/VAM/VMctrl_startup2.csv'
    tss_csv_path3 = './test_data/VAM/VMctrl_startup3.csv'
    tss_csv_path4 = './test_data/VAM/VMctrl_startup4.csv'
    tss_csv_path5 = './test_data/VAM/VMctrl_startup5.csv'

    tss_csv_path_list = [tss_csv_path1, tss_csv_path2, tss_csv_path3, tss_csv_path4, tss_csv_path5]
    data_preprocessing(tss_csv_list=tss_csv_path_list, sensor_group_id=sensor_group_id, output_dir=output_dir_tr)

    # normal testing data
    output_dir_ts = './test_data/vam_ts'
    tss_csv_path_list_ts = ['./test_data/VAM/VMctrl_startup6.csv']
    n_normal_trajectories = len(tss_csv_path_list_ts)
    data_preprocessing(tss_csv_list=tss_csv_path_list_ts, sensor_group_id=sensor_group_id, output_dir=output_dir_ts)
    add_labels_all_normal(output_dir_ts, n_normal_trajectories)

    # abnormal data for testing
    tss_csv_MAL18 = './test_data/VAM/VMctrl_MAL18.csv'
    target_period_MAL18 = [to_hours(0, 0, 6), to_hours(6, 5, 57)]
    abnormal_period_MAL18 = [to_hours(5, 35, 57), to_hours(6, 5, 57)]
    target_seg_id_MAL18 = 0
    tss_csv_MAL27 = './test_data/VAM/VMctrl_MAL27.csv'
    target_period_MAL27 = [to_hours(0, 0, 6), to_hours(0, 30, 6)]
    abnormal_period_MAL27 = [to_hours(0, 0, 6), to_hours(0, 30, 6)]
    target_seg_id_MAL27 = 0
    tss_csv_MAL06 = './test_data/VAM/VMctrl_MAL06.csv'
    target_period_MAL06 = [to_hours(0, 0, 6), to_hours(5, 46, 39)]
    abnormal_period_MAL06 = [to_hours(5, 16, 39), to_hours(5, 46, 39)]
    target_seg_id_MAL06 = 0

    tss_csv_MAL19 = './test_data/VAM/VMctrl_MAL19.csv'
    target_period_MAL19 = [to_hours(6, 6, 11), to_hours(8, 25, 46)]
    abnormal_period_MAL19 = [to_hours(7, 55, 46), to_hours(8, 25, 46)]
    target_seg_id_MAL19 = 1
    tss_csv_MAL22 = './test_data/VAM/VMctrl_MAL22.csv'
    target_period_MAL22 = [to_hours(6, 6, 11), to_hours(6, 36, 11)]
    abnormal_period_MAL22 = [to_hours(6, 6, 11), to_hours(6, 36, 11)]
    target_seg_id_MAL22 = 1
    tss_csv_MAL13 = './test_data/VAM/VMctrl_MAL13.csv'
    target_period_MAL13 = [to_hours(6, 6, 11), to_hours(6, 36, 11)]
    abnormal_period_MAL13 = [to_hours(6, 6, 11), to_hours(6, 36, 11)]
    target_seg_id_MAL13 = 1

    tss_csv_MAL20 = './test_data/VAM/VMctrl_MAL20.csv'
    target_period_MAL20 = [to_hours(9, 10, 2), to_hours(9, 40, 2)]
    abnormal_period_MAL20 = [to_hours(9, 10, 2), to_hours(9, 40, 2)]
    target_seg_id_MAL20 = 2
    tss_csv_MAL26 = './test_data/VAM/VMctrl_MAL26.csv'
    target_period_MAL26 = [to_hours(8, 51, 19), to_hours(9, 21, 19)]
    abnormal_period_MAL26 = [to_hours(8, 51, 19), to_hours(9, 21, 19)]
    target_seg_id_MAL26 = 2
    tss_csv_MAL14 = './test_data/VAM/VMctrl_MAL14.csv'
    target_period_MAL14 = [to_hours(9, 10, 2), to_hours(9, 40, 2)]
    abnormal_period_MAL14 = [to_hours(9, 10, 2), to_hours(9, 40, 2)]
    target_seg_id_MAL14 = 2

    tss_csv_MAL16 = './test_data/VAM/VMctrl_MAL16.csv'
    target_period_MAL16 = [to_hours(11, 29, 34), to_hours(11, 59, 34)]
    abnormal_period_MAL16 = [to_hours(11, 29, 34), to_hours(11, 59, 34)]
    target_seg_id_MAL16 = 3
    tss_csv_MAL21 = './test_data/VAM/VMctrl_MAL21.csv'
    target_period_MAL21 = [to_hours(11, 29, 34), to_hours(11, 59, 34)]
    abnormal_period_MAL21 = [to_hours(11, 29, 34), to_hours(11, 59, 34)]
    target_seg_id_MAL21 = 3
    tss_csv_MAL07 = './test_data/VAM/VMctrl_MAL07.csv'
    target_period_MAL07 = [to_hours(11, 29, 34), to_hours(11, 59, 34)]
    abnormal_period_MAL07 = [to_hours(11, 29, 34), to_hours(11, 59, 34)]
    target_seg_id_MAL07 = 3

    tss_csv_MAL15 = './test_data/VAM/VMctrl_MAL15.csv'
    target_period_MAL15 = [to_hours(14, 37, 11), to_hours(21, 8, 41)]
    abnormal_period_MAL15 = [to_hours(20, 38, 41), to_hours(21, 8, 41)]
    target_seg_id_MAL15 = 4
    tss_csv_MAL17 = './test_data/VAM/VMctrl_MAL17.csv'
    target_period_MAL17 = [to_hours(14, 37, 11), to_hours(21, 8, 41)]
    abnormal_period_MAL17 = [to_hours(20, 38, 41), to_hours(21, 8, 41)]
    target_seg_id_MAL17 = 4
    tss_csv_MAL05 = './test_data/VAM/VMctrl_MAL05.csv'
    target_period_MAL05 = [to_hours(14, 37, 11), to_hours(15, 7, 11)]
    abnormal_period_MAL05 = [to_hours(14, 37, 11), to_hours(15, 7, 11)]
    target_seg_id_MAL05 = 4

    initi_index_dict = {}
    for i in range(n_segments):
        initi_index_dict[str(i)] = n_normal_trajectories


    examples = [
    (tss_csv_MAL18, target_period_MAL18, abnormal_period_MAL18, target_seg_id_MAL18),
    (tss_csv_MAL27, target_period_MAL27, abnormal_period_MAL27, target_seg_id_MAL27),
    (tss_csv_MAL06, target_period_MAL06, abnormal_period_MAL06, target_seg_id_MAL06),
    (tss_csv_MAL19, target_period_MAL19, abnormal_period_MAL19, target_seg_id_MAL19),
    (tss_csv_MAL22, target_period_MAL22, abnormal_period_MAL22, target_seg_id_MAL22),
    (tss_csv_MAL13, target_period_MAL13, abnormal_period_MAL13, target_seg_id_MAL13),
    (tss_csv_MAL20, target_period_MAL20, abnormal_period_MAL20, target_seg_id_MAL20),
    (tss_csv_MAL26, target_period_MAL26, abnormal_period_MAL26, target_seg_id_MAL26),
    (tss_csv_MAL14, target_period_MAL14, abnormal_period_MAL14, target_seg_id_MAL14),
    (tss_csv_MAL16, target_period_MAL16, abnormal_period_MAL16, target_seg_id_MAL16),
    (tss_csv_MAL21, target_period_MAL21, abnormal_period_MAL21, target_seg_id_MAL21),
    (tss_csv_MAL07, target_period_MAL07, abnormal_period_MAL07, target_seg_id_MAL07),
    (tss_csv_MAL15, target_period_MAL15, abnormal_period_MAL15, target_seg_id_MAL15),
    (tss_csv_MAL17, target_period_MAL17, abnormal_period_MAL17, target_seg_id_MAL17),
    (tss_csv_MAL05, target_period_MAL05, abnormal_period_MAL05, target_seg_id_MAL05)
    ]
    data_preprocessing_period(examples=examples, sensor_group_id=sensor_group_id, output_dir=output_dir_ts, initi_index_dict=initi_index_dict)
