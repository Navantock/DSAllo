import torch.cuda

from modules.Predictor import DS_GCNPredictor, DS_CycleGCNPRedictor, SimpleLayerGCNPredictor
from utils import PredictAlloRes
from configs import args_main

if __name__ == "__main__":
    print(args_main)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    PModel = None
    if args_main.p == "DS_GCN":
        PModel = DS_GCNPredictor(surf_num=3).to(device)
    elif args_main.p == "DS_CycleGCN":
        PModel = DS_CycleGCNPRedictor(surf_num=3).to(device)
    elif args_main.p == "SimpleLayerGCN":
        PModel = SimpleLayerGCNPredictor().to(device)
    else:
        raise ValueError("No such predict model.")

    if args_main.pdb_id:
        PredictAlloRes(Predictor_model=PModel, Predictor_params_file=args_main.pf,
                       mode=args_main.mode, PDB_ID=args_main.pdb_id)
    else:
        PredictAlloRes(Predictor_model=PModel, Predictor_params_file=args_main.pf,
                       mode=args_main.mode, pdb_file_path=args_main.f)
