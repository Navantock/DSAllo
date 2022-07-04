import torch
from torch import nn as nn

from modules.Predictor import DS_GCNPredictor, DS_CycleGCNPRedictor, SimpleLayerGCNPredictor

import os
from tqdm import tqdm

from dataset import End2EndPredictorDataset
from configs import args_train
from utils import set_seed_global, cal_correct_pred, cal_accuracy, draw_train_curve


def train_predictor(predictor, run_device, save_model: bool = False,
                    model_name: str = "MyPredictor", draw_curve: bool = True):
    ds_dataset = End2EndPredictorDataset(root="./datasets")
    test_dataset = End2EndPredictorDataset(root="./test_datasets")

    optimizer = torch.optim.Adam(predictor.parameters(),
                                 lr=args_train.lr,
                                 weight_decay=args_train.wd)
    loss_P = nn.CrossEntropyLoss()

    best_epoch = 0
    best_val_acc = 0
    best_val_test_acc = 0
    epoch_train_acc = []
    epoch_val_acc = []
    epoch_test_acc = []

    with tqdm(range(args_train.epoch)) as tq:
        for epoch in tq:
            epoch_loss = []
            predictor.train()
            optimizer.zero_grad()
            for step, (pro_data, labels) in enumerate(ds_dataset):
                train_mask = labels["train_mask"].to(run_device)
                if train_mask.sum() == 0:
                    continue
                y = labels["allomask"].to(run_device)
                output = predictor(pro_data)
                loss = loss_P(output[train_mask], y[train_mask])
                epoch_loss.append(float(loss))
                loss.backward()
            optimizer.step()
            epoch_loss = sum(epoch_loss) / len(epoch_loss)

            train_acc_cnt, train_all_cnt = 0, 0
            valid_acc_cnt, valid_all_cnt = 0, 0
            test_acc_cnt, test_all_cnt = 0, 0
            with torch.no_grad():
                predictor.eval()
                for step, (pro_data, labels) in enumerate(ds_dataset):
                    train_mask = labels["train_mask"].to(run_device)
                    if train_mask.sum() == 0:
                        continue
                    valid_mask = labels["valid_mask"].to(run_device)
                    y = labels["allomask"].to(run_device)
                    output = predictor(pro_data)
                    pred = output.argmax(dim=1)

                    train_all_cnt += int(train_mask.sum())
                    train_acc_cnt += cal_correct_pred(pred, y, train_mask)
                    valid_all_cnt += int(valid_mask.sum())
                    valid_acc_cnt += cal_correct_pred(pred, y, valid_mask)

                for step, (pro_data, labels) in enumerate(test_dataset):
                    test_mask = labels["test_mask"].to(run_device)
                    if test_mask.sum() == 0:
                        continue
                    y = labels["allomask"].to(run_device)
                    output = predictor(pro_data)
                    pred = output.argmax(dim=1)

                    test_all_cnt += int(test_mask.sum())
                    test_acc_cnt += cal_correct_pred(pred, y, test_mask)

            train_acc = cal_accuracy(train_acc_cnt, train_all_cnt)
            valid_acc = cal_accuracy(valid_acc_cnt, valid_all_cnt)
            test_acc = cal_accuracy(test_acc_cnt, test_all_cnt)
            # Train infos
            infos = {
                'Epoch': epoch,
                'EpochAvgLoss': '{:.3f}'.format(epoch_loss),
                'TrainAcc': '{:.3f}'.format(train_acc),
                'ValidAcc': '{:.3f}'.format(valid_acc),
                'TestAcc': '{:.3f}'.format(test_acc)
            }

            tq.set_postfix(infos)
            epoch_train_acc.append(train_acc)
            epoch_val_acc.append(valid_acc)
            epoch_test_acc.append(test_acc)

    if save_model:
        torch.save(model.state_dict(), "./models/{}_params.pt".format(model_name))
    if draw_curve:
        draw_train_curve(epoch_train_acc, epoch_val_acc, epoch_test_acc, model_name)


if __name__ == "__main__":
    print(args_train)
    set_seed_global(args_train.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # model = SimpleLayerGCNPredictor(feature_dim=20).to(device)
    # train_predictor(model, device, model_name="SimpleGCN")
    model = DS_CycleGCNPRedictor(surf_num=3,
                                 embedding_dim=args_train.embedding_dim,
                                 dp=args_train.dp).to(device)
    train_predictor(model, device, save_model=True, model_name="DS_CycleGCNPRedictor")
    # model = DS_GCNPredictor(surf_num=3).to(device)
    # train_predictor(model, device, save_model=True, model_name="DS_GCNPRedictor")


