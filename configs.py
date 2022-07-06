import argparse


def usr_arg_train():
    arg_parser_train = argparse.ArgumentParser()
    # Random seed
    arg_parser_train.add_argument('--seed', type=int, default=42)
    # Hidden dimension of neural networks
    arg_parser_train.add_argument('--embedding_dim', type=int, default=32)
    # Epochs to train
    arg_parser_train.add_argument('--epoch', type=int, default=200)
    # Learning rate
    arg_parser_train.add_argument('--lr', type=float, default=1e-2)
    # Weight decay
    arg_parser_train.add_argument('--wd', type=float, default=1e-5)
    # Dropout probability
    arg_parser_train.add_argument('--dp', type=float, default=0.5)
    # Experiment: 0 None; 1 Test ACM in all params; 2 Compare ACM-GCN with GCN
    arg_parser_train.add_argument('--experiment', type=int, default=0)

    return arg_parser_train.parse_args()


def use_arg_main():
    arg_parser_main = argparse.ArgumentParser()
    # PDB file path
    arg_parser_main.add_argument('--f', type=str, default="./AsBench_Core_Set/allosteric_proteins/AS001000501_3UO9.pdb")
    # pdb id
    arg_parser_main.add_argument('--pdb_id', type=str, default="")
    # predictor name
    arg_parser_main.add_argument('--p', type=str, default="DS_CycleGCN")
    # predictor params file path
    arg_parser_main.add_argument('--pf', type=str, default="./models/DS_CycleGCNPRedictor_params.pt")
    # if visualize
    arg_parser_main.add_argument('--v', type=int, default=1)
    # predictor output mode
    arg_parser_main.add_argument('--vis_mode', type=str, default="show")

    return arg_parser_main.parse_args()
