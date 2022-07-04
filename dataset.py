import torch
from torch.utils.data import Dataset
from torch_geometric.data import InMemoryDataset as GDataset
from torch_geometric.data import Data as GData

from Bio.PDB.PDBParser import PDBParser
from Bio.PDB.NeighborSearch import NeighborSearch
from Bio.PDB.Polypeptide import is_aa
from Bio.PDB.ResidueDepth import residue_depth, get_surface

from typing import Union, List, Tuple
from math import ceil
import random
import os

RES_CODE = {"GLY": 0, "ALA": 1, "VAL": 2, "LEU": 3, "ILE": 4, "PRO": 5,
            "PHE": 6, "TYR": 7, "TRP": 8, "SER": 9, "THR": 10, "CYS": 11,
            "MET": 12, "ASN": 13, "GLN": 14, "ASP": 15, "GLU": 16, "LYS": 17,
            "ARG": 18, "HIS": 19, "OTHER": 20}
PDB_PARSER = PDBParser(PERMISSIVE=1)


def generate_ds_list_complex(model, pro_ds_surface, deep_surf_num: int = 2, allo_cutoff: float = 3):
    pro_ds_surface_id = [res.get_full_id()[2:] for res in pro_ds_surface]
    deepSurfaceList = []
    all_atom, modulator = [], []
    surface = get_surface(model)
    for chain in model:
        for residue in chain:
            all_atom.extend(residue.get_list())
            if residue.get_id()[0] != " ":
                if residue.get_id()[0][:2] == "H_":
                    modulator.extend(residue.get_list())
                continue
            if residue.get_full_id()[2:] not in pro_ds_surface_id:
                rd = residue_depth(residue, surface)
                deepSurfaceList.append([residue, rd])

    max_depth = max(deepSurfaceList, key=lambda x: x[1])[1]
    min_depth = min(deepSurfaceList, key=lambda x: x[1])[1]
    stride = (max_depth - min_depth) / deep_surf_num
    deepSurface = [[] for _ in range(deep_surf_num)]
    for res in deepSurfaceList:
        if res[1] == min_depth:
            deepSurface[deep_surf_num - 1].append(res[0])
        elif res[1] == max_depth:
            deepSurface[0].append(res[0])
        else:
            deepSurface[-ceil((res[1] - min_depth) / stride)].append(res[0])

    deepSurface.append([])
    for res in pro_ds_surface:
        cur_id = res.get_full_id()[2:]
        deepSurface[-1].append(model[cur_id[0]][cur_id[1]])

    neighbor = NeighborSearch(all_atom)
    allo_sites = []
    for a in modulator:
        for n in neighbor.search(a.get_coord(), radius=allo_cutoff, level="R"):
            cur_id = n.get_full_id()[-2:]
            if cur_id[1][0] == " " and cur_id not in allo_sites:
                allo_sites.append(cur_id)

    return deepSurface, allo_sites


def generate_ds_list_pro(model, deep_surf_num: int = 2, surface_thickness: float = 6.0):
    deepSurfaceList = []
    surface = get_surface(model)
    for chain in model:
        for residue in chain:
            if residue.get_id()[0] != " ":
                continue
            rd = residue_depth(residue, surface)
            deepSurfaceList.append([residue, rd])

    max_depth = max(deepSurfaceList, key=lambda x: x[1])[1]
    ds_max_depth = max_depth - surface_thickness
    stride = ds_max_depth / deep_surf_num

    deepSurface = [[] for _ in range(deep_surf_num + 1)]

    for res in deepSurfaceList:
        if res[1] < surface_thickness:
            deepSurface[-1].append(res[0])
        elif res[1] >= max_depth:
            deepSurface[0].append(res[0])
        else:
            deepSurface[-1 - ceil((res[1] - surface_thickness) / stride)].append(res[0])

    return deepSurface


def generate_layer_graph(ds, _cutoff: float):
    atom_list = []
    x = []
    edge_index = [[], []]
    res_index = {}
    cnt = 0
    for res in ds:
        x.append([1 if i == RES_CODE[res.get_resname()] else 0 for i in range(20)])
        res_index[res] = cnt
        cnt += 1
        atom_list.extend(res.get_list())
    neighbor = NeighborSearch(atom_list)
    for pair in neighbor.search_all(radius=_cutoff, level="R"):
        r1, r2 = res_index[pair[0]], res_index[pair[1]]
        edge_index[0].extend([r1, r2])
        edge_index[1].extend([r2, r1])
    return x, edge_index


def generate_layer2layer_graph(ds_outside, ds_inside, _layerCutoff):
    layer_edge_index = [[], []]
    degrees = [1 for _ in range(len(ds_outside))]
    for i in range(len(ds_outside)):
        for j in range(len(ds_inside)):
            if ds_outside[i]['CA'] - ds_inside[j]['CA'] <= _layerCutoff:
                degrees[i] += 1
                layer_edge_index[0].append(i)
                layer_edge_index[1].append(j)
    return layer_edge_index, degrees


def graph_from_ds_list_pro(deepSurface, allo_sites, root, pdb_file_name, cutoff: float = 3.0,
                           layer_cutoff: float = 8.0):
    data_list = []
    layer_cnt = 0
    for ds in deepSurface:
        x, edge_index = generate_layer_graph(ds, _cutoff=cutoff)
        layer_edge_index, degrees = ([[], []], []) if layer_cnt == 0 else \
            generate_layer2layer_graph(ds, deepSurface[layer_cnt - 1], _layerCutoff=layer_cutoff)
        data_list.append(GData(x=torch.tensor(x), edge_index=torch.tensor(edge_index),
                               layer_edge_index=torch.tensor(layer_edge_index), degrees=torch.tensor(degrees)))
        # Surface allo mask
        if layer_cnt == len(deepSurface) - 1:
            allo_mask = []
            for res in ds:
                for allo_res_id in allo_sites:
                    if res.get_full_id()[-2:] == allo_res_id:
                        allo_mask.append(1)
                        break
                else:
                    allo_mask.append(0)
            if not os.path.exists(os.path.join(root, "allomask")):
                os.mkdir(os.path.join(root, "allomask"))
            torch.save(torch.tensor(allo_mask), os.path.join(root, "allomask/{}_mask.pt").format(pdb_file_name[:-4]))

        layer_cnt += 1

    if not os.path.exists(os.path.join(root, "protein")):
        os.mkdir(os.path.join(root, "protein"))
    torch.save(data_list, os.path.join(root, "protein/{}_pro.pt").format(pdb_file_name[:-4]))


def graph_from_ds_list_complex(deepSurface, root, pdb_file_name, cutoff: float = 3.0,
                               layer_cutoff: float = 8.0):
    data_list = []
    layer_cnt = 0
    for ds in deepSurface:
        x, edge_index = generate_layer_graph(ds, _cutoff=cutoff)
        layer_edge_index, degrees = ([], []) if layer_cnt == 0 else \
            generate_layer2layer_graph(ds, deepSurface[layer_cnt - 1], _layerCutoff=layer_cutoff)
        data_list.append(GData(x=torch.tensor(x), edge_index=torch.tensor(edge_index),
                               layer_edge_index=torch.tensor(layer_edge_index), degrees=torch.tensor(degrees)))
        layer_cnt += 1

    if not os.path.exists(os.path.join(root, "complex")):
        os.mkdir(os.path.join(root, "complex"))
    torch.save(data_list, os.path.join(root, "complex/{}_complex.pt").format(pdb_file_name[:-4]))


def split_train_valid(root, allomask_dir="allomask", positive_rate: float = 0.5, valid_rate: float = 0.2):
    for mask_file in os.listdir(os.path.join(root, allomask_dir)):
        path = os.path.join(root, allomask_dir, mask_file)
        allo = torch.load(path)
        all_positive = sum(allo)
        valid_positive = ceil(all_positive * valid_rate)
        train_positive = all_positive - valid_positive
        positive_idx = []
        negative_idx = []
        for i in range(len(allo)):
            if allo[i] == 1:
                positive_idx.append(i)
            else:
                negative_idx.append(i)

        valid_n = random.sample(negative_idx, ceil(valid_positive / positive_rate - valid_positive))
        del_list = []
        for i in range(len(negative_idx)):
            if negative_idx[i] in valid_n:
                del_list.append(i)
        negative_idx = [i for idx, i in enumerate(negative_idx) if idx not in del_list]

        del_list = []
        valid_p = random.sample(positive_idx, valid_positive)
        for i in range(len(positive_idx)):
            if positive_idx[i] in valid_p:
                del_list.append(i)
        positive_idx = [i for idx, i in enumerate(positive_idx) if idx not in del_list]

        train_idx = random.sample(negative_idx, ceil(train_positive / positive_rate - train_positive)) \
                    + random.sample(positive_idx, train_positive)
        valid_idx = valid_n + valid_p

        train_mask = torch.BoolTensor([(_ in train_idx) for _ in range(len(allo))])
        valid_mask = torch.BoolTensor([(_ in valid_idx) for _ in range(len(allo))])
        torch.save({"allomask": allo, "train_mask": train_mask, "valid_mask": valid_mask}, path)


def split_test(root, allomask_dir="allomask", positive_rate: float = 0.2):
    for mask_file in os.listdir(os.path.join(root, allomask_dir)):
        path = os.path.join(root, allomask_dir, mask_file)
        allo = torch.load(path)
        positive_idx = []
        negative_idx = []
        for i in range(len(allo)):
            if allo[i] == 1:
                positive_idx.append(i)
            else:
                negative_idx.append(i)

        negative_cnt = min(ceil(len(positive_idx) / positive_rate - len(positive_idx)), len(negative_idx))
        test_idx = random.sample(negative_idx, negative_cnt) + positive_idx
        test_mask = torch.BoolTensor([(_ in test_idx) for _ in range(len(allo))])
        torch.save({"allomask": allo, "test_mask": test_mask}, path)


def generate_train_dataset(pro_dir="./AsBench_Core_Set/allosteric_proteins",
                           complex_dir="./AsBench_Core_Set/protein-modulator_complexes",
                           root="./datasets", surface_thickness: float = 3.0,
                           deep_surf_num: int = 2, allo_cutoff: float = 3.0, cutoff: float = 3.0,
                           length: int = 10):
    pro_file_names = os.listdir(pro_dir)
    complex_file_names = os.listdir(complex_dir)
    for i in range(length):
        pro_path = os.path.join(pro_dir, pro_file_names[i])
        complex_path = os.path.join(complex_dir, complex_file_names[i])
        pro_serial = pro_path[-8:-4]
        complex_serial = pro_serial + "_complex"
        pro_structure = PDB_PARSER.get_structure(pro_serial, pro_path)
        complex_structure = PDB_PARSER.get_structure(complex_serial, complex_path)
        if len(pro_structure) > 1 or len(complex_structure) > 1:
            print("Too many models at {}.".format(pro_path))
            continue
        for pro_model, complex_model in zip(pro_structure, complex_structure):
            pro_ds_list = generate_ds_list_pro(pro_model, deep_surf_num=deep_surf_num,
                                               surface_thickness=surface_thickness)
            complex_ds_list, allo_sites = generate_ds_list_complex(complex_model, pro_ds_list[-1],
                                                                   allo_cutoff=allo_cutoff)

            # save allo sites id
            if not os.path.exists(os.path.join(root, "allosites")):
                os.mkdir(os.path.join(root, "allosites"))
            torch.save(allo_sites, os.path.join(root, "allosites/{}_sites.pt".format(pro_file_names[i][:-4])))

            print(allo_sites)
            graph_from_ds_list_pro(pro_ds_list, allo_sites=allo_sites, root=root, pdb_file_name=pro_file_names[i], cutoff=cutoff)
            graph_from_ds_list_complex(complex_ds_list, root=root, pdb_file_name=pro_file_names[i], cutoff=cutoff)
    print("Done Successfully!")


def get_protein_data(pro_path="./AsBench_Core_Set/allosteric_proteins",
                     surface_thickness: float = 3.0,
                     deep_surf_num: int = 2, cutoff: float = 3.0,
                     layer_cutoff: float = 8.0):
    print("Generating Protein Data...")
    pro_serial = pro_path[:-4]
    pro_structure = PDB_PARSER.get_structure(pro_serial, pro_path)
    if len(pro_structure) > 1:
        raise RuntimeError("Too many models at {}.".format(pro_path) + " Please config the model you want to predict.")
    for pro_model in pro_structure:
        deepSurface = generate_ds_list_pro(pro_model, deep_surf_num=deep_surf_num,
                                           surface_thickness=surface_thickness)
        surface_res = deepSurface[-1]
        data_list = []
        layer_cnt = 0
        for ds in deepSurface:
            x, edge_index = generate_layer_graph(ds, _cutoff=cutoff)
            layer_edge_index, degrees = ([[], []], []) if layer_cnt == 0 else \
                generate_layer2layer_graph(ds, deepSurface[layer_cnt - 1], _layerCutoff=layer_cutoff)
            data_list.append(GData(x=torch.tensor(x), edge_index=torch.tensor(edge_index),
                                   layer_edge_index=torch.tensor(layer_edge_index), degrees=torch.tensor(degrees)))
            layer_cnt += 1
        return data_list, surface_res


class End2EndPredictorDataset(Dataset):
    def __init__(self, root, pro_dir="protein", mask_dir="allomask"):
        super(End2EndPredictorDataset, self).__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.root = root
        self.pro_dir_path = os.path.join(root, pro_dir)
        self.mask_dir_path = os.path.join(root, mask_dir)

    @property
    def pro_ds_file_names(self):
        return os.listdir(self.pro_dir_path)

    @property
    def mask_ds_file_names(self):
        return os.listdir(self.mask_dir_path)

    def __len__(self):
        return min(len(self.pro_ds_file_names), len(self.complex_ds_file_names))

    def __getitem__(self, item):
        pro_path = os.path.join(self.pro_dir_path, self.pro_ds_file_names[item])
        mask_path = os.path.join(self.mask_dir_path, self.mask_ds_file_names[item])
        pro_gdata = torch.load(pro_path)
        labels = torch.load(mask_path)
        return pro_gdata, labels


if __name__ == "__main__":
    random.seed(514)
    generate_train_dataset(
        pro_dir="./AsBench_Core_Set/allosteric_proteins_test",
        complex_dir="./AsBench_Core_Set/protein-modulator_complexes_test",
        root="./test_datasets", length=10)
    split_test(root="./test_datasets")
