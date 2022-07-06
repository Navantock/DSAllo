import torch.cuda

from modules.Predictor import DS_GCNPredictor, DS_CycleGCNPRedictor, SimpleLayerGCNPredictor
from utils import PredictAlloRes
from configs import use_arg_main

if __name__ == "__main__":
    args_main = use_arg_main()
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

    out_file = None
    if args_main.pdb_id:
        out_file = PredictAlloRes(Predictor_model=PModel, Predictor_params_file=args_main.pf, PDB_ID=args_main.pdb_id,
                                  visualize=bool(args_main.v), vis_mode=args_main.vis_mode)
    else:
        out_file = PredictAlloRes(Predictor_model=PModel, Predictor_params_file=args_main.pf, pdb_file_path=args_main.f,
                                  visualize=bool(args_main.v), vis_mode=args_main.vis_mode)

    print("Predicted info of allosteric sites has been writen.")
    print("Please check file path {}.".format(out_file))
