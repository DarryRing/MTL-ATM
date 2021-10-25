#!/usr/bin/env python
# encoding: utf-8


from __future__ import print_function
import argparse
import pickle

import numpy as np
import torch

import misc


def predict_subject(model, cat_seq, value_seq, time_seq, l, all_tp, all_tp_new,flag
):
    """
    Predict Alzheimer’s disease progression for a subject
    Args:
        model: trained pytorch model
        cat_seq: sequence of diagnosis [nb_input_timpoints, nb_classes]
        value_seq: sequence of other features [nb_input_timpoints, nb_features]
        time_seq: months from baseline [nb_output_timpoints, nb_features]
    nb_input_timpoints <= nb_output_timpoints
    Returns:
        out_cat: predicted diagnosis
        out_val: predicted features
    """
    in_val = np.full((len(time_seq)+1, ) + value_seq.shape[1:], np.nan)
    in_val[:len(value_seq)] = value_seq

    in_cat = np.full((len(time_seq)+1, ) + cat_seq.shape[1:], np.nan)
    in_cat[:len(cat_seq)] = cat_seq
    ocat = []
    oval = []
    true_length = len(all_tp)
    task_name =  flag
    for i in range(1, true_length):
        single_icat = np.array([in_cat[i - 1], in_cat[i]])
        single_ival = np.array([in_val[i - 1], in_val[i]])
        single_itime = np.array(time_seq.cpu().numpy()[i - 1]).reshape(1, 1, 1)
        single_itime = torch.tensor(single_itime, dtype=torch.float32).cuda(1)

        with torch.no_grad():
            ocatogory, ovalue = model(single_icat, single_ival, single_itime, task_name
)
        idx = np.isnan(in_cat[i])
        # val_seq[j][idx] = o_val.data.cpu().numpy()[idx]
        # val_seq[j][idx] = o_val.data.cpu().numpy()[:, -self.nb_measures:][idx]
        # in_cat[i][idx] = in_cat[i-1][idx]
        in_cat[i][idx] = ocatogory[-1].cpu().numpy()[idx]
        idx = np.isnan(in_val[i])
        # cat_seq[j][idx] = o_cat.data.cpu().numpy()[idx]
        # in_val[i][idx] = in_val[i-1][idx]
        in_val[i][idx] = ovalue[-1, :, 2:].cpu().numpy()[idx]
        # in_val[i] = ovalue[-1].cpu().numpy()[:, -20:]
        # in_cat[i] = ocatogory[-1].cpu().numpy()[:, -20:]
        # in_val[i] = ovalue[-1].cpu().numpy()[:, -20:]
        # in_cat[i] = ocatogory[-1].cpu().numpy()[:, -20:]
        ocat.append(ocatogory[-1].cpu().numpy())
        oval.append(ovalue[-1].cpu().numpy())
    delta_new = all_tp_new[true_length:] - all_tp_new[true_length-1]
    delta_new = delta_new / 100.
    delta_new = torch.tensor(np.array(delta_new).reshape(len(delta_new), 1, 1), dtype=torch.float32).cuda(1)
    for i in range(true_length, l):
        single_icat = np.array([in_cat[true_length - 1], in_cat[i]])
        single_ival = np.array([in_val[true_length - 1], in_val[i]])
        single_itime = np.array(delta_new.cpu().numpy()[i - true_length]).reshape(1, 1, 1)
        single_itime = torch.tensor(single_itime, dtype=torch.float32).cuda(1)

        with torch.no_grad():
            ocatogory, ovalue = model(single_icat, single_ival, single_itime,task_name)
        # in_val[1] = ovalue[-1].cpu().numpy()
        # in_cat[1] = ocatogory[-1].cpu().numpy()
        ocat.append(ocatogory[-1].cpu().numpy())
        oval.append(ovalue[-1].cpu().numpy())


    out_cat = np.array(ocat)
    out_val = np.array(oval)
    # with torch.no_grad():
    #     out_cat, out_val = model(in_cat, in_val, time_seq)
    # out_cat = out_cat.cpu().numpy()
    # out_val = out_val.cpu().numpy()

    assert out_cat.shape[1] == out_val.shape[1] == 1

    return out_cat, out_val


def predict(model, dataset, pred_start, duration, baseline, flag):
    """
    Predict Alzheimer’s disease progression using a trained model
    Args:
        model: trained pytorch model
        dataset: test data
        pred_start (dictionary): the date at which prediction begins
        duration (dictionary): how many months into the future to predict
        baseline (dictionary): the baseline date
    Returns:
        dictionary which contains the following key/value pairs:
            subjects: list of subject IDs
            DX: list of diagnosis prediction for each subject
            ADAS13: list of ADAS13 prediction for each subject
            Ventricles: list of ventricular volume prediction for each subject
    """
    model.eval()
    ret = {'subjects': dataset.subjects}
    ret['DX'] = []  # 1. likelihood of NL, MCI, and Dementia
    ret['ADAS13'] = []  # 2. (best guess, upper and lower bounds on 50% CI)
    ret['Ventricles'] = []  # 3. (best guess, upper and lower bounds on 50% CI)
    ret['dates'] = misc.make_date_col(
        [pred_start[s] for s in dataset.subjects], duration)

    col = ['ADAS13', 'Ventricles', 'ICV']
    # indices = misc.get_index(list(dataset.value_fields()), col)
    indices = misc.get_index(list(dataset.attributes), col)
    mean = model.mean[col].values.reshape(1, -1)
    stds = model.stds[col].values.reshape(1, -1)

    count = 0
    for data in dataset:
        count = count + 1
        # print("subj: ", count)
        rid = data['rid']
        all_tp = data['tp'].squeeze(axis=1)
        start = misc.month_between(pred_start[rid], baseline[rid])
        # assert np.all(all_tp == np.arange(len(all_tp)))
        l = len(all_tp)
        all_tp_new = list(all_tp)
        all_tp_new.extend([m for m in range(start, start+duration)])
        all_tp_new = np.array(all_tp_new)
        # delta = all_tp_new[1:] - all_tp_new[:-1]
        # delta = torch.tensor(np.array(delta).reshape(len(delta), 1, 1), dtype=torch.float32).cuda(1)
        # build delta
        delta = all_tp_new[1:len(all_tp_new)] - all_tp_new[0:len(all_tp_new)-1]
        delta = delta/100.
        delta = torch.tensor(np.array(delta).reshape(len(delta), 1, 1), dtype=torch.float32).cuda(1)
        # delta = torch.tensor(np.array(delta).reshape(len(delta), 1, 1), dtype=torch.float32).cuda(1)
        mask = all_tp < start
        # itime = np.arange(start + duration)
        itime = delta
        icat = np.asarray(
            [misc.to_categorical(c, 3) for c in data['cat'][mask]])
        ival = data['val'][:, None, :][mask]
        # delta = all_tp_new[1:] - all_tp_new[0]

        ocat, oval = predict_subject(model, icat, ival, itime, l+duration, all_tp, all_tp_new,flag)
        oval = oval[-duration:, 0, indices] * stds + mean

        ret['DX'].append(ocat[-duration:, 0, :])
        ret['ADAS13'].append(misc.add_ci_col(oval[:, 0], 1, 0, 85))
        ret['Ventricles'].append(
            misc.add_ci_col(oval[:, 1] / oval[:, 2], 5e-4, 0, 1))
    print("subj: ", count)
    return ret


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--data', required=True)
    parser.add_argument('--out', '-o', required=True)

    return parser.parse_args()


def main(args):
    """
    Predict Alzheimer’s disease progression using a trained model
    Save prediction as a csv file
    Args:
        args: includes model path, input/output paths
    Returns:
        None
    """
    device = torch.device(
        'cuda:1') if torch.cuda.is_available() else torch.device('cpu')
    model = torch.load(args.checkpoint)
    model.to(device)

    with open(args.data, 'rb') as fhandler:
        data = pickle.load(fhandler)

    prediction = predict(model, data['test'], data['pred_start'],
                         data['duration'], data['baseline'], args.flag)
    misc.build_pred_frame(prediction, args.out)


class Params():
    def __init__(self,model,val,prediction, flag):
        self.checkpoint = model
        self.data = val
        self.out = prediction
        self.flag = flag


# if __name__ == '__main__':
#     # main(get_args())
#     main(Params())
