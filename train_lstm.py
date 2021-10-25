from __future__ import print_function, division
import argparse
import json
import time
import pickle
import Sequence_Time_LSTM_1
import numpy as np
import torch
import misc
import warnings
import random
import config


def ent_loss(pred, true, mask):
    """
    Calculate cross-entropy loss
    Args:
        pred: predicted probability distribution,
              [nb_timpoints, nb_subjects, nb_classes]
        true: true class, [nb_timpoints, nb_subjects, 1]
        mask: timepoints to evaluate, [nb_timpoints, nb_subjects, 1]
    Returns:
        cross-entropy loss
    """
    assert isinstance(pred, torch.Tensor)
    assert isinstance(true, np.ndarray) and isinstance(mask, np.ndarray)
    nb_subjects = true.shape[1]

    pred = pred.reshape(pred.size(0) * pred.size(1), -1)
    mask = mask.reshape(-1, 1)

    o_true = pred.new_tensor(true.reshape(-1, 1)[mask], dtype=torch.long)
    o_pred = pred[pred.new_tensor(
        mask.squeeze(1).astype(np.uint8), dtype=torch.uint8).bool()]

    return torch.nn.functional.cross_entropy(
        o_pred, o_true, reduction='sum') / nb_subjects

def mae_loss(pred, true, mask):
    """
    Calculate mean absolute error (MAE)
    Args:
        pred: predicted values, [nb_timpoints, nb_subjects, nb_features]
        true: true values, [nb_timpoints, nb_subjects, nb_features]
        mask: values to evaluate, [nb_timpoints, nb_subjects, nb_features]
    Returns:
        MAE loss
    """
    assert isinstance(pred, torch.Tensor)
    assert isinstance(true, np.ndarray) and isinstance(mask, np.ndarray)
    nb_subjects = true.shape[1]

    invalid = ~mask
    true[invalid] = 0
    indices = pred.new_tensor(invalid.astype(np.uint8), dtype=torch.uint8).bool()
    assert pred.shape == indices.shape
    loss_temporal_smooth = 0
    loss_temporal_smooth = sum(sum(sum(abs(pred[1:,:,:] - pred[:-1,:,:]))))
    pred[indices] = 0
    return torch.nn.functional.l1_loss(
        pred, pred.new(true), reduction='sum') / nb_subjects + loss_temporal_smooth/nb_subjects


def to_cat_seq(labels):
    """
    Return one-hot representation of a sequence of class labels
    Args:
        labels: [nb_subjects, nb_timpoints]
    Returns:
        [nb_subjects, nb_timpoints, nb_classes]
    """
    return np.asarray([misc.to_categorical(c, 3) for c in labels])

def cat_all_null(a, b):
    """
    judge whether the categories are all null
    :param a:
    :param b:
    :return:
    """
    a = a.squeeze()
    b = b.squeeze()
    c = a[b]
    res = len(c) > 0
    return res
def train_1epoch(args, model, dataset, dataset2, dataset3, optimizer):
    """
    Train an recurrent model for 1 epoch
    Args:
        args: include training hyperparametres and input/output paths
        model: pytorch model object
        dataset: training data
        optimizer: optimizer
    Returns:
        cross-entropy loss of epoch
        mean absolute error (MAE) loss of epoch
    """
    model.train()
    total_ent = total_mae = 0
    batch_dataset = []
    for batch in dataset:
        batch["task_name"] = 0
        batch_dataset.append(batch)
    for batch in dataset2:
        batch["task_name"] = 1
        batch_dataset.append(batch)
    for batch in dataset3:
        batch["task_name"] = 2
        batch_dataset.append(batch)

    random.shuffle(batch_dataset)
    for batch in batch_dataset:
        if len(batch['tp']) == 1:
            continue
        lengths = batch["lengths"]

        mask = [l == 5 for l in lengths]
        for key, v in batch.items():
            # print(type(v))
            if not isinstance(v, list) and not isinstance(v, int):
                batch[key] = v[:, mask, :]
        batch["lengths"] = len(mask)
        optimizer.zero_grad()
        batch_ent = torch.tensor(np.array(0.), requires_grad=True).cuda(config.CUDA)
        batch_mae = torch.tensor(np.array(0.), requires_grad=True).cuda(config.CUDA)
        batch_cat = to_cat_seq(batch["cat"])
        batch_val = batch["val"]
        batch_delta = torch.Tensor(batch["delta_batch"]).cuda(config.CUDA)
        batch_size = batch_val.shape[1]
        batch_delta = batch_delta / 100.
        pred_cat, pred_val = model(batch_cat, batch_val, batch_delta, batch["task_name"])
        mask_cat = batch['cat_msk'][1:]
        if cat_all_null(batch['true_cat'][1:][0:batch_cat.shape[0], :, :], mask_cat) is False:
            batch_size = batch_size - 1
            continue
        ent = ent_loss(pred_cat, batch['true_cat'][1:][0:batch_cat.shape[0], :, :], mask_cat)
        mae = mae_loss(pred_val, batch['true_val'][1:][0:batch_cat.shape[0], :, :],
                       batch['val_msk'][1:][0:batch_cat.shape[0], :, :])
        batch_ent = batch_ent + ent
        batch_mae = batch_mae + mae
        total_loss = batch_mae/22 + args.w_ent * batch_ent
        total_loss.backward()
        optimizer.step()
        total_ent = total_ent + batch_ent.item() * batch_size
        total_mae = total_mae + batch_mae.item() * batch_size
    return total_ent / len(dataset.subjects), total_mae / len(dataset.subjects)/22


def save_config(args, config_path):
    """
    Save training configuration as json file
    Args:
        args: include training hyperparametres and input/output paths
        config_path: path of output json file
    Returns:
        None
    """
    with open(config_path, 'w') as fhandler:
        print(json.dumps(vars(args), sort_keys=True), file=fhandler)
class Params():
    def __init__(self, data, data2, data3, batch_size, epochs, lr, out, nb_layers, hize):
        # self.data = '../../output/val.pkl'
        self.data = data
        self.data2 = data2
        self.data3 = data3
        self.i_drop = 0.1
        self.h_drop = 0.1
        self.batch_size = batch_size
        self.validation = "false"
        self.epochs = epochs
        self.lr = lr
        self.model = "time_LSTM"
        self.weight_decay = 5e-7
        self.out = out
        self.w_ent = 1.
        self.nb_layers = nb_layers
        self.h_size = hize
        self.seed = 0
        self.verbose = "True"
        self.TIME_EMBED_SIZE = 1

def train(args):

    log = print if args.verbose else lambda *x, **i: None

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    with open(args.data, 'rb') as fhandler:
        data = pickle.load(fhandler)

    with open(args.data2, 'rb') as fhandler:
        data2 = pickle.load(fhandler)

    with open(args.data3, 'rb') as fhandler:
        data3 = pickle.load(fhandler)

    nb_measures = len(data['train'].value_fields())
    model = Sequence_Time_LSTM_1.Sequence_Time_LSTM_1(
        nb_classes=3,
        nb_measures=nb_measures,
        nb_layers=args.nb_layers,
        h_size=args.h_size,
        h_drop=args.h_drop,
        i_drop=args.i_drop,
        # input_length = args.input_length,
        TIME_EMBED_SIZE=args.TIME_EMBED_SIZE
        )
    setattr(model, 'mean', data['mean'])
    setattr(model, 'stds', data['stds'])

    device = torch.device(
        'cuda:1') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)
    log(model)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    start = time.time()
    try:
        for i in range(args.epochs):
            loss = train_1epoch(args, model, data['train'], data2['train'], data3['train'], optimizer)
            log_info = (i + 1, args.epochs, misc.time_from(start)) + loss
            log('%d/%d %s ENT %.3f, MAE %.3f' % log_info)
    except KeyboardInterrupt:
        print('Early exit')

    torch.save(model, args.out)
    save_config(args, '%s.json' % args.out)


if __name__ == '__main__':
    train(Params())