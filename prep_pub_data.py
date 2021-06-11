import ast
import os
import pathlib

from tqdm import tqdm

import numpy as np
import pandas as pd

SMAP_FILENAME = [
    'A-1.npy',
    'A-2.npy',
    'A-3.npy',
    'A-4.npy',
    'A-5.npy',
    'A-6.npy',
    'A-7.npy',
    'A-8.npy',
    'A-9.npy',
    'B-1.npy',
    'D-1.npy',
    'D-2.npy',
    'D-3.npy',
    'D-4.npy',
    'D-5.npy',
    'D-6.npy',
    'D-7.npy',
    'D-8.npy',
    'D-9.npy',
    'D-11.npy',
    'D-12.npy',
    'D-13.npy',
    'E-1.npy',
    'E-2.npy',
    'E-3.npy',
    'E-4.npy',
    'E-5.npy',
    'E-6.npy',
    'E-7.npy',
    'E-8.npy',
    'E-9.npy',
    'E-10.npy',
    'E-11.npy',
    'E-12.npy',
    'E-13.npy',
    'F-1.npy',
    'F-2.npy',
    'F-3.npy',
    'G-1.npy',
    'G-2.npy',
    'G-3.npy',
    'G-4.npy',
    'G-6.npy',
    'G-7.npy',
    'P-1.npy',
    'P-2.npy',
    'P-3.npy',
    'P-4.npy',
    'P-7.npy',
    'R-1.npy',
    'S-1.npy',
    'T-1.npy',
    'T-2.npy',
    'T-3.npy'
    ]
MSL_FILENAME = [
    'C-1.npy',
    'C-2.npy',
    'D-14.npy',
    'D-15.npy',
    'D-16.npy',
    'F-4.npy',
    'F-5.npy',
    'F-7.npy',
    'F-8.npy',
    'M-1.npy',
    'M-2.npy',
    'M-3.npy',
    'M-4.npy',
    'M-5.npy',
    'M-6.npy',
    'M-7.npy',
    'P-10.npy',
    'P-11.npy',
    'P-14.npy',
    'P-15.npy',
    'S-2.npy',
    'T-4.npy',
    'T-5.npy',
    'T-8.npy',
    'T-9.npy',
    'T-12.npy',
    'T-13.npy'
    ]
SMD_FILENAME = [
    'machine-1-1.txt',
    'machine-1-2.txt',
    'machine-1-3.txt',
    'machine-1-4.txt',
    'machine-1-5.txt',
    'machine-1-6.txt',
    'machine-1-7.txt',
    'machine-1-8.txt',
    'machine-2-1.txt',
    'machine-2-2.txt',
    'machine-2-3.txt',
    'machine-2-4.txt',
    'machine-2-5.txt',
    'machine-2-6.txt',
    'machine-2-7.txt',
    'machine-2-8.txt',
    'machine-2-9.txt',
    'machine-3-1.txt',
    'machine-3-2.txt',
    'machine-3-3.txt',
    'machine-3-4.txt',
    'machine-3-5.txt',
    'machine-3-6.txt',
    'machine-3-7.txt',
    'machine-3-8.txt',
    'machine-3-9.txt',
    'machine-3-10.txt',
    'machine-3-11.txt'
    ]

PAMAP2_COMP_ACTID = [1, 2, 3, 4, 12, 13, 16, 17]
PAMAP2_ALL_ACTID = [1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 16, 17, 18, 19, 20, 24]
ALL_SUBJECT_ID = [1, 2, 3, 4, 5, 6, 7, 8, 9]
TRAIN_SUBJECT_ID = [1, 2, 3, 4, 5, 6, 7]
TEST_SUBJECT_ID = [8]
SENSOR_GROUP_ID = [2, 3, 4, 5]
ACTIVITY_ID_IDX = 1
TIME_IDX = 0

def add_pseudo_time_stamps(arr):
    n_samples = arr.shape[0]
    pseudo_time_stamps = np.arange(n_samples).reshape((n_samples,1))
    return np.concatenate([pseudo_time_stamps, arr], axis=1)

def add_column_names(df_):
    _column_names = ['Time'] + [str(i) for i in range(len(df_.columns)-1)]
    df_.columns = _column_names
    return df_

def change_first_column_name(df_):
    _names = df_.columns.values
    _column_names = []
    _column_names.append('Time')
    for i in range(1, len(_names)):
        _column_names.append(_names[i])
    df_.columns = _column_names
    return df_

def prep_pamap2(output_dir_tr, output_dir_ts):
    import numpy as np
    import random
    seed = 123
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    os.makedirs(output_dir_tr, exist_ok=True)
    os.makedirs(output_dir_ts, exist_ok=True)

    sensor_info_path = os.path.join('test_data', 'PAMAP2', 'sensor_groups_pamap2.csv')
    sensors_info = pd.read_csv(sensor_info_path, header=0)
    sensor_groups = sensors_info.groupby('group_index')
    sensor_group_id = []
    for i in SENSOR_GROUP_ID:
        s_group_id = sensor_groups.get_group(i)['index'].values.tolist()
        sensor_group_id.append(s_group_id)

    available_act_info_path = os.path.join('test_data', 'PAMAP2', 'pamap2_available_act.csv')
    available_act_info = pd.read_csv(available_act_info_path, header=0)
    available_act_info.index = available_act_info['activity_id']

    # check number of segments
    dataset_dir = os.path.join('test_data', 'PAMAP2')

    tss_data = {}
    abnormal_activity = []
    # load all data
    print('loading data...')
    for i in tqdm(ALL_SUBJECT_ID):
        # load data
        subject_name = 'subject10'+str(i)
        f_name = subject_name +'.dat'
        f_path = os.path.join(dataset_dir, f_name)
        data_ = np.loadtxt(f_path)
        df_ = pd.DataFrame(data_)

        # replace NaN with a numerical value
        df_ = df_.fillna(method='ffill')
        df_ = df_.fillna(method='bfill')

        subject_op_name = subject_name+'_op'
        f_name_op = subject_op_name +'.dat'
        f_path_op = os.path.join(dataset_dir, f_name_op)

        if os.path.exists(f_path_op):
            data_op = np.loadtxt(f_path_op)
            df_op = pd.DataFrame(data_op)

            # replace NaN with a numerical value
            df_op = df_op.fillna(method='ffill')
            df_op = df_op.fillna(method='bfill')

            df_ = pd.concat([df_, df_op], ignore_index=True)

        tss_data[str(i)] = df_
        pd_sr = available_act_info[subject_name]
        for k, _row in zip(pd_sr.index, pd_sr):
            if k not in PAMAP2_COMP_ACTID:
                if _row == 1:
                    abnormal_activity.append((i, k))

    print('check the data')
    for i in ALL_SUBJECT_ID:
        df_ = tss_data[str(i)]
        for i_act_id, _act_id in enumerate(PAMAP2_ALL_ACTID):
            df_sub = df_[df_.iloc[:, ACTIVITY_ID_IDX] == _act_id]
            print('subj: {}, act: {}, n_samples: {}'.format(i, _act_id, df_sub.shape[0]))

    # generate training data
    print('generating training data ...')
    for i_act_id, _act_id in tqdm(enumerate(PAMAP2_COMP_ACTID), total=len(PAMAP2_COMP_ACTID)):
        cnt = 0
        for _subj_id in TRAIN_SUBJECT_ID:
            df_ = tss_data[str(_subj_id)]
            df_sub = df_[df_.iloc[:, ACTIVITY_ID_IDX] == _act_id]
            prev_idx = None
            n_seg = 1
            n_samples = 0
            _array = []
            seg_samples = []
            segmented_array = []
            for k, _row in df_sub.iterrows():
                if prev_idx == None:
                    prev_idx = k
                    n_samples += 1
                    _array.append(_row)
                    continue
                if k - 1 != prev_idx:
                    n_seg += 1
                    seg_samples.append(n_samples)
                    segmented_array.append(_array)
                    n_samples = 0
                    _array = []
                prev_idx = k
                n_samples += 1
                _array.append(_row)
            seg_samples.append(n_samples)
            segmented_array.append(_array)
            for _n_samples, _array in zip(seg_samples, segmented_array):
                assert _n_samples == np.array(_array).shape[0]

            for _array in segmented_array:
                _df_seg = pd.DataFrame(_array)
                _df_seg.columns = list(range(len(_df_seg.columns)))
                _df_seg = change_first_column_name(_df_seg)
                for i_sensor_g_id, sensor_g_id in enumerate(SENSOR_GROUP_ID):
                    selected_column_idx = [0]
                    selected_column_idx.extend(sensor_group_id[i_sensor_g_id])
                    selected_column_name = _df_seg.columns.values[selected_column_idx]
                    _out_dir_tr = os.path.join(output_dir_tr, str(i_sensor_g_id)+'___'+str(i_act_id))
                    os.makedirs(_out_dir_tr, exist_ok=True)
                    output_path = os.path.join(_out_dir_tr, str(cnt)+'.csv')
                    _df_seg.to_csv(output_path, columns=selected_column_name, index=False)
                cnt += 1
            # print('subj: {}, act: {}, n_seg: {}, n_samples: {}'.format(_subj_id, _act_id, n_seg, seg_samples))

    # generate testing data
    print('generating normal testing data ...')
    pointer_seg_dict = {}
    for i_act_id, _act_id in tqdm(enumerate(PAMAP2_COMP_ACTID), total=len(PAMAP2_COMP_ACTID)):
        cnt = 0
        for _subj_id in TEST_SUBJECT_ID:
            df_ = tss_data[str(_subj_id)]
            df_sub = df_[df_.iloc[:, ACTIVITY_ID_IDX] == _act_id]
            prev_idx = None
            n_seg = 1
            n_samples = 0
            _array = []
            seg_samples = []
            segmented_array = []
            for k, _row in df_sub.iterrows():
                if prev_idx == None:
                    prev_idx = k
                    n_samples += 1
                    _array.append(_row)
                    continue
                if k - 1 != prev_idx:
                    n_seg += 1
                    seg_samples.append(n_samples)
                    segmented_array.append(_array)
                    n_samples = 0
                    _array = []
                prev_idx = k
                n_samples += 1
                _array.append(_row)
            seg_samples.append(n_samples)
            segmented_array.append(_array)
            for _n_samples, _array in zip(seg_samples, segmented_array):
                assert _n_samples == np.array(_array).shape[0]

            for _array in segmented_array:
                _df_seg = pd.DataFrame(_array)
                _df_seg.columns = list(range(len(_df_seg.columns)))
                _df_seg = change_first_column_name(_df_seg)
                for i_sensor_g_id, sensor_g_id in enumerate(SENSOR_GROUP_ID):
                    selected_column_idx = [0]
                    selected_column_idx.extend(sensor_group_id[i_sensor_g_id])
                    selected_column_name = _df_seg.columns.values[selected_column_idx]
                    _out_dir_ts = os.path.join(output_dir_ts, str(i_sensor_g_id)+'___'+str(i_act_id))
                    os.makedirs(_out_dir_ts, exist_ok=True)
                    output_path = os.path.join(_out_dir_ts, str(cnt)+'.csv')
                    _df_seg.to_csv(output_path, columns=selected_column_name, index=False)

                    # for label
                    _out_path_label = os.path.join(_out_dir_ts, 'label_'+str(cnt)+'.txt')
                    _n_samples_ts = _df_seg.shape[0]
                    _label = np.zeros((_n_samples_ts, 1))
                    np.savetxt(_out_path_label, _label, delimiter=',')
                cnt += 1
        pointer_seg_dict[_act_id] = cnt

    print('generating abnormal testing data ...')
    n_segments = len(PAMAP2_COMP_ACTID)
    if n_segments > len(abnormal_activity):
        sampled_abnormal_activity = random.choices(abnormal_activity, n_segments)
    else:
        sampled_abnormal_activity = random.sample(abnormal_activity, n_segments)

    print('selected abnormal activity')
    print(sampled_abnormal_activity)
    _pointer = 0
    for i_act_id, _act_id in tqdm(enumerate(PAMAP2_COMP_ACTID), total=len(PAMAP2_COMP_ACTID)):
        cnt = pointer_seg_dict[_act_id]
        _ab_subj_id, _ab_act_id = sampled_abnormal_activity[_pointer]
        df_ = tss_data[str(_ab_subj_id)]
        df_sub = df_[df_.iloc[:, ACTIVITY_ID_IDX] == _ab_act_id]
        prev_idx = None
        n_seg = 1
        n_samples = 0
        _array = []
        seg_samples = []
        segmented_array = []
        for k, _row in df_sub.iterrows():
            if prev_idx == None:
                prev_idx = k
                n_samples += 1
                _array.append(_row)
                continue
            if k - 1 != prev_idx:
                n_seg += 1
                seg_samples.append(n_samples)
                segmented_array.append(_array)
                n_samples = 0
                _array = []
            prev_idx = k
            n_samples += 1
            _array.append(_row)
        seg_samples.append(n_samples)
        segmented_array.append(_array)
        for _n_samples, _array in zip(seg_samples, segmented_array):
            assert _n_samples == np.array(_array).shape[0]
        for i_sensor_g_id, sensor_g_id in enumerate(SENSOR_GROUP_ID):
            _array = segmented_array[0]  # use the first segment
            _df_seg = pd.DataFrame(_array)
            _df_seg.columns = list(range(len(_df_seg.columns)))
            _df_seg = change_first_column_name(_df_seg)
            for i_sensor_g_id, sensor_g_id in enumerate(SENSOR_GROUP_ID):
                selected_column_idx = [0]
                selected_column_idx.extend(sensor_group_id[i_sensor_g_id])
                selected_column_name = _df_seg.columns.values[selected_column_idx]
                _out_dir_ts = os.path.join(output_dir_ts, str(i_sensor_g_id)+'___'+str(i_act_id))
                os.makedirs(_out_dir_ts, exist_ok=True)
                output_path = os.path.join(_out_dir_ts, str(cnt)+'.csv')
                _df_seg.to_csv(output_path, columns=selected_column_name, index=False)

                # for label
                _out_path_label = os.path.join(_out_dir_ts, 'label_'+str(cnt)+'.txt')
                _n_samples_ts = _df_seg.shape[0]
                _label = np.ones((_n_samples_ts, 1))
                np.savetxt(_out_path_label, _label, delimiter=',')

                _ = pathlib.Path(os.path.join(_out_dir_ts, 'nm_subject10'+str(_subj_id)+'_act'+str(_act_id)))
                _.touch()
                _ = pathlib.Path(os.path.join(_out_dir_ts, 'ab_subject10'+str(_ab_subj_id)+'_act'+str(_ab_act_id)))
                _.touch()
        pointer_seg_dict[_act_id] += 1
        _pointer += 1


def prep_jpl(output_dir_tr, output_dir_ts, target_filenames):
    os.makedirs(output_dir_tr, exist_ok=True)
    os.makedirs(output_dir_ts, exist_ok=True)

    dataset_dir = os.path.join('test_data', 'JPL')
    label_path = os.path.join(dataset_dir, 'labeled_anomalies.csv')
    label_df = pd.read_csv(label_path)

    for i, _fname in tqdm(enumerate(target_filenames), total=len(target_filenames)):
        _chan_id = os.path.splitext(_fname)[0]

        # training data
        _out_dir_tr = os.path.join(output_dir_tr, '0___'+str(i))
        os.makedirs(_out_dir_tr, exist_ok=True)
        _out_path_tr = os.path.join(_out_dir_tr, '0.csv')
        _data_tr = np.load(os.path.join(dataset_dir, 'train', _fname))
        _data_tr = add_pseudo_time_stamps(_data_tr)
        tr_df = pd.DataFrame(_data_tr)
        tr_df = add_column_names(tr_df)
        tr_df.to_csv(_out_path_tr, index=False)
        _ = pathlib.Path(os.path.join(_out_dir_tr, _chan_id))
        _.touch()

        # test data
        _out_dir_ts = os.path.join(output_dir_ts, '0___'+str(i))
        os.makedirs(_out_dir_ts, exist_ok=True)
        _out_path_ts = os.path.join(_out_dir_ts, '0.csv')
        _data_ts = np.load(os.path.join(dataset_dir, 'test', _fname))
        _data_ts = add_pseudo_time_stamps(_data_ts)
        ts_df = pd.DataFrame(_data_ts)
        ts_df = add_column_names(ts_df)
        ts_df.to_csv(_out_path_ts, index=False)
        _ = pathlib.Path(os.path.join(_out_dir_ts, _chan_id))
        _.touch()

        # label
        _out_path_label = os.path.join(_out_dir_ts, 'label.txt')
        _n_samples_ts = _data_ts.shape[0]
        _label_df_targe = label_df[label_df['chan_id']==_chan_id]
        _label = np.zeros((_n_samples_ts, 1))
        for _, _row in _label_df_targe.iterrows():
            list_anomaly_sequences = ast.literal_eval(_row['anomaly_sequences'])
            for _indicies in list_anomaly_sequences:
                _label[_indicies[0]:_indicies[1]+1,0] = 1
        np.savetxt(_out_path_label, _label, delimiter=',')

def prep_smd(output_dir_tr, output_dir_ts):
    os.makedirs(output_dir_tr, exist_ok=True)
    os.makedirs(output_dir_ts, exist_ok=True)

    dataset_dir = os.path.join('test_data', 'ServerMachineDataset')

    for i, _fname in tqdm(enumerate(SMD_FILENAME), total=len(SMD_FILENAME)):
        _chan_id = os.path.splitext(_fname)[0]

        # training data
        _out_dir_tr = os.path.join(output_dir_tr, '0___'+str(i))
        os.makedirs(_out_dir_tr, exist_ok=True)
        _out_path_tr = os.path.join(_out_dir_tr, '0.csv')
        _data_tr = np.loadtxt(os.path.join(dataset_dir, 'train', _fname), dtype=np.float32, delimiter=',')
        _data_tr = add_pseudo_time_stamps(_data_tr)
        tr_df = pd.DataFrame(_data_tr)
        tr_df = add_column_names(tr_df)
        tr_df.to_csv(_out_path_tr, index=False)
        _ = pathlib.Path(os.path.join(_out_dir_tr, _chan_id))
        _.touch()

        # test data
        _out_dir_ts = os.path.join(output_dir_ts, '0___'+str(i))
        os.makedirs(_out_dir_ts, exist_ok=True)
        _out_path_ts = os.path.join(_out_dir_ts, '0.csv')
        _data_ts = np.loadtxt(os.path.join(dataset_dir, 'test', _fname), dtype=np.float32, delimiter=',')
        _data_ts = add_pseudo_time_stamps(_data_ts)
        ts_df = pd.DataFrame(_data_ts)
        ts_df = add_column_names(ts_df)
        ts_df.to_csv(_out_path_ts, index=False)
        _ = pathlib.Path(os.path.join(_out_dir_ts, _chan_id))
        _.touch()

        # label
        _out_path_label = os.path.join(_out_dir_ts, 'label.txt')
        _label = np.loadtxt(os.path.join(dataset_dir, 'test_label', _fname), dtype=np.float32, delimiter=',')
        np.savetxt(_out_path_label, _label, delimiter=',')

if __name__ == '__main__':
    output_dir_tr = './test_data/pamap2_tr'
    output_dir_ts = './test_data/pamap2_ts'
    prep_pamap2(output_dir_tr, output_dir_ts)

    output_dir_tr = './test_data/smap_tr'
    output_dir_ts = './test_data/smap_ts'
    prep_jpl(output_dir_tr, output_dir_ts, SMAP_FILENAME)

    output_dir_tr = './test_data/msl_tr'
    output_dir_ts = './test_data/msl_ts'
    prep_jpl(output_dir_tr, output_dir_ts, MSL_FILENAME)

    output_dir_tr = './test_data/smd_tr'
    output_dir_ts = './test_data/smd_ts'
    prep_smd(output_dir_tr, output_dir_ts)
