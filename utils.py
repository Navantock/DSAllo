import torch
from torch.nn.functional import softmax

from Bio.PDB.PDBList import PDBList

import numpy as np
import matplotlib.pyplot as plt

import os

from dataset import get_protein_data


def set_seed_global(seed: int, force_deter=False):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if force_deter:
        torch.use_deterministic_algorithms(True)
        import os
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'


def cal_correct_pred(pred, label, mask):
    return int((pred[mask] == label[mask]).sum())


def cal_accuracy(correct_pred, total_pred):
    return correct_pred / total_pred


def draw_train_curve(train_acc_list, val_acc_list, test_acc_list, model_name, save_dir: str = "./results"
                                                                                              "/learning_curves"):
    assert len(train_acc_list) == len(val_acc_list) == len(test_acc_list), \
        RuntimeError("Learning Curve Data Sheet With Unequal Length")
    epochs = [i for i in range(len(train_acc_list))]
    best_epoch = val_acc_list.index(max(val_acc_list))
    best_test_epoch = test_acc_list.index(max(test_acc_list))
    plt.figure()
    plt.xlabel("epoch")
    plt.ylabel("acc")
    plt.rcParams["xtick.direction"] = "in"
    plt.rcParams["ytick.direction"] = "in"
    plt.plot(epochs, train_acc_list, color="green", label="{}:train acc".format(model_name))
    plt.plot(epochs, val_acc_list, color="blue", label="{}:val acc".format(model_name))
    plt.plot(epochs, test_acc_list, color="orange", label="{}:test acc".format(model_name))
    plt.axvline(best_epoch, linestyle='--', color="red")
    plt.axvline(best_test_epoch, linestyle='--', color="pink")

    plt.legend(loc=1)
    plt.savefig(os.path.join(save_dir, model_name + ".png"))


def draw_roc_curve():
    # TODO: Implement
    pass


def interpolateColor(alpha):
    colorA = (1, 0, 0)
    colorB = (0, 0, 1)
    return tuple(np.array(colorA)*(1-alpha) + np.array(colorB)*(alpha))


def PredictAlloRes(Predictor_model, Predictor_params_file: str,
                   mode: str = "probability" or "binary", visualize: bool = True, **kwargs):
    pdb_file_path = None
    pdb_file_name = None
    if "PDB_ID" in kwargs:
        Downloader = PDBList()
        if "save_dir" in kwargs:
            pdb_file_name = Downloader.retrieve_pdb_file(kwargs["PDB_ID"], pdir=kwargs["save_dir"], file_format='pdb')
            pdb_file_path = os.path.join(kwargs["save_dir"], pdb_file_name)
        else:
            Warning("No save directory required. Save pdb_file in current directory.")
            pdb_file_name = Downloader.retrieve_pdb_file(kwargs["PDB_ID"], pdir=".", file_format='pdb')
            pdb_file_path = "./" + pdb_file_name
    elif "pdb_file_path" in kwargs:
        pdb_file_path = kwargs["pdb_file_path"]
        pdb_file_name = pdb_file_path[-8:-4]

    assert pdb_file_path is not None, RuntimeError("No 'PDB_ID' or 'pdb_file_path' found.")
    assert os.path.exists(pdb_file_path), RuntimeError("PDB file path not found.")
    assert os.path.exists(pdb_file_path), RuntimeError("Predictor params file not found.")

    Predictor_model.load_state_dict(torch.load(Predictor_params_file))
    ds_list, surface_res = get_protein_data(pdb_file_path)
    print("Read data successfully.")
    result = Predictor_model(ds_list)
    print("Calculate successfully.")
    result_binary = [float(_.argmax(dim=0)) for _ in result]
    result_p = [float(_[0]/sum(_)) for _ in result]
    if visualize:
        x = [res['CA'].get_coord()[0] for res in surface_res]
        y = [res['CA'].get_coord()[1] for res in surface_res]
        z = [res['CA'].get_coord()[2] for res in surface_res]

        plt.figure()
        plt.hot()
        ax = plt.axes(projection='3d')
        ax.set_xlabel('X', fontsize=10, color='black')
        ax.set_ylabel('Y', fontsize=10, color='black')
        ax.set_zlabel('Z', fontsize=10, color='black')
        ax.scatter3D(x, y, z, s=30, c=result_binary, cmap="viridis")
        plt.show()
        plt.savefig("./results/hot_visualize/{}_hot_binary.png".format(pdb_file_name))

        plt.figure()
        plt.hot()
        ax = plt.axes(projection='3d')
        ax.set_xlabel('X', fontsize=10, color='black')
        ax.set_ylabel('Y', fontsize=10, color='black')
        ax.set_zlabel('Z', fontsize=10, color='black')
        ax.scatter3D(x, y, z, s=30, c=result_p, cmap="viridis")
        plt.show()
        plt.savefig("./results/hot_visualize/{}_hot_probability.png".format(pdb_file_name))

    out = {}

    return out
