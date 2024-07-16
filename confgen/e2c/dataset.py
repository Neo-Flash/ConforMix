from genericpath import exists
import numpy as np
import random
import os
import json
from tqdm import tqdm
from random import sample
import pickle
from rdkit import Chem
from rdkit.Chem.rdchem import Mol, HybridizationType
import torch
from confgen import utils
from torch_geometric.data import InMemoryDataset, Data
from torch_sparse import SparseTensor
import re
import confgen
from ..molecule.graph import rdk2graph, rdk2graphedge
import copy
from rdkit.Chem.rdmolops import RemoveHs
from confgen.molecule.gt import isomorphic_core, isomorphic_core_from_graph
from confgen.model.gnn import one_hot_atoms, one_hot_bonds
from torch_geometric.data import Data
from torch_geometric.transforms import LineGraph
import numpy as np
import warnings 
import time  
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from concurrent.futures import ProcessPoolExecutor, TimeoutError
def calculate_neighborhood(edge_index, num_nodes):
    nei_src_index = []
    nei_tgt_index = [[] for _ in range(num_nodes)]
    for i, j in zip(edge_index[0], edge_index[1]):
        nei_src_index.append(i)
        nei_tgt_index[i].append(j)
    nei_src_index = np.unique(nei_src_index)
    max_nei = max(len(nei) for nei in nei_tgt_index)
    nei_tgt_mask = np.ones((max_nei, num_nodes), dtype=bool)
    for i, neis in enumerate(nei_tgt_index):
        nei_tgt_mask[:len(neis), i] = False
        nei_tgt_index[i].extend([-1] * (max_nei - len(neis)))  
    nei_tgt_index = np.array(nei_tgt_index)
    return nei_src_index, nei_tgt_index, nei_tgt_mask
def convert_to_line_graph(x, edge_index, edge_attr):
    padded_edge_attr = torch.cat([edge_attr, torch.zeros(edge_attr.size(0), 173 - 15)], dim=1)
    data = Data(edge_index=edge_index, edge_attr=edge_attr)
    line_graph_transform = LineGraph(force_directed=True)
    line_graph_data = line_graph_transform(data)
    line_graph_node_features = []
    for idx in range(edge_index.shape[1]):
        node_a, node_b = edge_index[:, idx]
        edge_feature = padded_edge_attr[idx]
        new_node_feature = (x[node_a] + x[node_b]) / 2 + edge_feature
        line_graph_node_features.append(new_node_feature)
    line_graph_node_features = torch.stack(line_graph_node_features)
    line_graph_data.x = line_graph_node_features
    line_graph_edge_attr = []
    for i in range(line_graph_data.edge_index.shape[1]):
        edge_a, edge_b = line_graph_data.edge_index[:, i]
        new_edge_feature = (edge_attr[edge_a] + edge_attr[edge_b]) / 2
        line_graph_edge_attr.append(new_edge_feature)
    line_graph_edge_attr = torch.stack(line_graph_edge_attr)
    line_graph_data.edge_attr = line_graph_edge_attr
    return line_graph_data
class PygGeomDataset(InMemoryDataset):
    def __init__(
        self,
        root="dataset",
        rdk2graph=rdk2graph,
        transform=None,
        pre_transform=None,
        dataset="qm9",
        base_path="/Users/flash/Desktop/BaseLine/DMCG/dataset",
        seed=None,
        extend_edge=False,
        data_split="cgcf",
        remove_hs=False,
    ):
        self.original_root = root
        self.rdk2graph = rdk2graph
        if seed == None:
            self.seed = 2021
        else:
            self.seed = seed
        assert dataset in ["qm9", "drugs", "iso17", "pdb", "csd"]
        self.folder = os.path.join(root, f"geom_{dataset}_{data_split}")
        if extend_edge:
            self.rdk2graph = rdk2graphedge
            self.folder = os.path.join(root, f"geom_{dataset}_{data_split}_ee")
        if remove_hs:
            self.folder = os.path.join(root, f"geom_{dataset}_{data_split}_rh_ext_gt")
        else:
            self.folder = os.path.join(root, f"geom_{dataset}_{data_split}_ext_gt")
        self.base_path = base_path
        self.dataset_name = dataset
        self.data_split = data_split
        self.remove_hs = remove_hs
        super().__init__(self.folder, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])
    @property
    def raw_file_names(self):
        return "data.csv.gz"
    @property
    def processed_file_names(self):
        return "geometric_data_processed.pt"
    def download(self):
        if os.path.exists(self.processed_paths[0]):
            return
        else:
            assert os.path.exists(self.base_path)
    def process(self):
        assert self.dataset_name in ["qm9", "drugs", "iso17", "pdb", "csd"]
        if self.data_split == "pdb":
            self.process_pdb()
            return
        if self.data_split == "csd":
            self.process_csd()
            return
        if self.data_split == "confgf":
            self.process_confgf()
            return
        summary_path = os.path.join(self.base_path, f"summary_{self.dataset_name}.json")
        with open(summary_path, "r") as src:
            summ = json.load(src)
        pickle_path_list = []
        for smiles, meta_mol in tqdm(summ.items()):
            u_conf = meta_mol.get("uniqueconfs")
            if u_conf is None:
                continue
            pickle_path = meta_mol.get("pickle_path")
            if pickle_path is None:
                continue
            if "." in smiles:
                continue
            pickle_path_list.append(pickle_path)
        data_list = []
        num_mols = 0
        num_confs = 0
        bad_case = 0
        random.seed(19970327)
        random.shuffle(pickle_path_list)
        train_size = int(len(pickle_path_list) * 0.8)
        valid_size = int(len(pickle_path_list) * 0.9)
        train_idx = []
        valid_idx = []
        test_idx = []
        for i, pickle_path in enumerate(tqdm(pickle_path_list)):
            if self.dataset_name in ["drugs"]:
                if i < train_size:
                    if len(train_idx) >= 2000000:
                        continue
                elif i < valid_size:
                    if len(valid_idx) >= 100000:
                        continue
                else:
                    if len(test_idx) >= 100000:
                        continue
            with open(os.path.join(self.base_path, pickle_path), "rb") as src:
                mol = pickle.load(src)
            if mol.get("uniqueconfs") != len(mol.get("conformers")):
                bad_case += 1
                continue
            if mol.get("uniqueconfs") <= 0:
                bad_case += 1
                continue
            if mol.get("conformers")[0]["rd_mol"].GetNumBonds() < 1:
                bad_case += 1
                continue
            if "." in Chem.MolToSmiles(mol.get("conformers")[0]["rd_mol"]):
                bad_case += 1
                continue
            num_mols += 1
            for conf_meta in mol.get("conformers"):
                if self.remove_hs:
                    try:
                        new_mol = RemoveHs(conf_meta["rd_mol"])
                    except Exception:
                        continue
                else:
                    new_mol = conf_meta["rd_mol"]
                graph = self.rdk2graph(new_mol)
                assert len(graph["edge_attr"]) == graph["edge_index"].shape[1]
                assert len(graph["node_feat"]) == graph["num_nodes"]
                data = Data()
                data.edge_index = torch.from_numpy(graph["edge_index"]).to(torch.int64)
                data.edge_attr = torch.from_numpy(graph["edge_attr"]).to(torch.int64)
                data.x = torch.from_numpy(graph["node_feat"]).to(torch.int64)
                data.n_nodes = graph["n_nodes"]
                data.n_edges = graph["n_edges"]
                data.pos = torch.from_numpy(new_mol.GetConformer(0).GetPositions()).to(torch.float)
                data.lowestenergy = torch.as_tensor([mol.get("lowestenergy")]).to(torch.float)
                data.energy = torch.as_tensor([conf_meta["totalenergy"]]).to(torch.float)
                data.rd_mol = copy.deepcopy(new_mol)
                data.isomorphisms = isomorphic_core(new_mol)
                data.nei_src_index = torch.from_numpy(graph["nei_src_index"]).to(torch.int64)
                data.nei_tgt_index = torch.from_numpy(graph["nei_tgt_index"]).to(torch.int64)
                data.nei_tgt_mask = torch.from_numpy(graph["nei_tgt_mask"]).to(torch.bool)
                if i < train_size:
                    train_idx.append(len(data_list))
                elif i < valid_size:
                    valid_idx.append(len(data_list))
                else:
                    test_idx.append(len(data_list))
                data_list.append(data)
                num_confs += 1
        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]
        data, slices = self.collate(data_list)
        print("Saving...")
        print(f"num mols {num_mols} num confs {num_confs} num bad cases {bad_case}")
        torch.save((data, slices), self.processed_paths[0])
        os.makedirs(os.path.join(self.root, "split"), exist_ok=True)
        torch.save(
            {
                "train": torch.tensor(train_idx, dtype=torch.long),
                "valid": torch.tensor(valid_idx, dtype=torch.long),
                "test": torch.tensor(test_idx, dtype=torch.long),
            },
            os.path.join(self.root, "split", "split_dict.pt"),
        )
    def get_idx_split(self):
        path = os.path.join(self.root, "split")
        if os.path.isfile(os.path.join(path, "split_dict.pt")):
            return torch.load(os.path.join(path, "split_dict.pt"))
    def process_confgf(self):
        valid_conformation = 0
        train_idx = []
        valid_idx = []
        test_idx = []
        data_list = []
        bad_case = 0
        file_name = ["train_data_40k", "val_data_5k", "test_data_200"]
        if self.dataset_name == "drugs":
            file_name[0] = "train_data_39k"
        print("Converting pickle files into graphs...")
        for subset in file_name:
            pkl_fn = os.path.join(self.base_path, f"{subset}.pkl")
            with open(pkl_fn, "rb") as src:
                mol_list = pickle.load(src)
            # random.seed(42) 
            # mol_list = sample(mol_list, int(len(mol_list) * 0.001))
            mol_list = [x.rdmol for x in mol_list]
            for mol in tqdm(mol_list):
                if self.remove_hs:
                    try:
                        mol = RemoveHs(mol)
                    except Exception:
                        continue
                if "." in Chem.MolToSmiles(mol):
                    bad_case += 1
                    continue
                if mol.GetNumBonds() < 1:
                    bad_case += 1
                    continue
                graph = self.rdk2graph(mol)
                assert len(graph["edge_attr"]) == graph["edge_index"].shape[1]
                assert len(graph["node_feat"]) == graph["num_nodes"]
                data = CustomData()
                data.edge_index = torch.from_numpy(graph["edge_index"]).to(torch.int64)
                data.edge_attr = one_hot_bonds(torch.from_numpy(graph["edge_attr"])).to(torch.float32)
                data.x = one_hot_atoms(torch.from_numpy(graph["node_feat"])).to(torch.float32)
                data.n_nodes = graph["n_nodes"]
                data.n_edges = graph["n_edges"]
                data.pos = torch.from_numpy(mol.GetConformer(0).GetPositions()).to(torch.float)
                data.rd_mol = copy.deepcopy(mol)
                data.isomorphisms = isomorphic_core(mol)
                data.nei_src_index = torch.from_numpy(graph["nei_src_index"]).to(torch.int64)
                data.nei_tgt_index = torch.from_numpy(graph["nei_tgt_index"]).to(torch.int64)
                data.nei_tgt_mask = torch.from_numpy(graph["nei_tgt_mask"]).to(torch.bool)
                # line graph
                line_graph = convert_to_line_graph(data.x, data.edge_index, data.edge_attr)
                data.line_graph_n_nodes = len(line_graph.x)
                data.line_graph_pos  = torch.tensor((data.pos[data.edge_index[0]] + data.pos[data.edge_index[1]]) / 2).to(torch.float) 
                data.line_graph_isomorphisms = isomorphic_core_from_graph(torch.tensor(line_graph.x).to(torch.float32), torch.tensor(line_graph.edge_index).to(torch.int64), torch.tensor(line_graph.edge_attr).to(torch.float32))
                if "train" in subset:
                    train_idx.append(valid_conformation)
                elif "val" in subset:
                    valid_idx.append(valid_conformation)
                else:
                    test_idx.append(valid_conformation)
                valid_conformation += 1
                data_list.append(data)
        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]
        data, slices = self.collate(data_list)
        print("Saving...")
        print(f"num confs {valid_conformation} num bad cases {bad_case}")
        torch.save((data, slices), self.processed_paths[0])
        os.makedirs(os.path.join(self.root, "split"), exist_ok=True)
        if len(valid_idx) == 0:
            valid_idx = train_idx[:6400]
        torch.save(
            {
                "train": torch.tensor(train_idx, dtype=torch.long),
                "valid": torch.tensor(valid_idx, dtype=torch.long),
                "test": torch.tensor(test_idx, dtype=torch.long),
            },
            os.path.join(self.root, "split", "split_dict.pt"),
        )
    def process_pdb(self):
        print("Converting sdf files into graphs...")
        mol_list = [mol for mol in Chem.SDMolSupplier("dataset/CSD-2.sdf") if mol is not None]
        valid_conformation = 0
        data_list = []
        bad_case = 0
        random.shuffle(mol_list)
        total_mols = len(mol_list)
        train_size = int(total_mols * 0.8)
        valid_size = int(total_mols * 0.9)
        train_idx = []
        valid_idx = []
        test_idx = []
        for idx, mol in enumerate(tqdm(mol_list)):
            if self.remove_hs:
                try:
                    mol = Chem.RemoveHs(mol)
                except Exception:
                    continue
            if "." in Chem.MolToSmiles(mol):
                bad_case += 1
                continue
            if mol.GetNumBonds() < 1:
                bad_case += 1
                continue
            try:
                graph = self.rdk2graph(mol)
            except Exception as e:
                print(f"Skipping molecule due to an error: {e}")
                bad_case += 1
                continue
            assert len(graph["edge_attr"]) == graph["edge_index"].shape[1]
            assert len(graph["node_feat"]) == graph["num_nodes"]
            data = CustomData()
            data.edge_index = torch.from_numpy(graph["edge_index"]).to(torch.int64)
            data.edge_attr = one_hot_bonds(torch.from_numpy(graph["edge_attr"])).to(torch.float32)
            data.x = one_hot_atoms(torch.from_numpy(graph["node_feat"])).to(torch.float32)
            data.n_nodes = graph["n_nodes"]
            data.n_edges = graph["n_edges"]
            data.pos = torch.from_numpy(mol.GetConformer(0).GetPositions()).to(torch.float)
            data.rd_mol = copy.deepcopy(mol)
            data.isomorphisms = isomorphic_core(mol)
            data.nei_src_index = torch.from_numpy(graph["nei_src_index"]).to(torch.int64)
            data.nei_tgt_index = torch.from_numpy(graph["nei_tgt_index"]).to(torch.int64)
            data.nei_tgt_mask = torch.from_numpy(graph["nei_tgt_mask"]).to(torch.bool)
            line_graph = convert_to_line_graph(data.x, data.edge_index, data.edge_attr)
            data.line_graph_n_nodes = len(line_graph.x)
            data.line_graph_pos  = torch.tensor((data.pos[data.edge_index[0]] + data.pos[data.edge_index[1]]) / 2).to(torch.float) 
            data.line_graph_isomorphisms = isomorphic_core_from_graph(torch.tensor(line_graph.x).to(torch.float32), torch.tensor(line_graph.edge_index).to(torch.int64), torch.tensor(line_graph.edge_attr).to(torch.float32))
            if idx < train_size:
                train_idx.append(len(data_list))
            elif idx < valid_size:
                valid_idx.append(len(data_list))
            else:
                test_idx.append(len(data_list))
            data_list.append(data)
            valid_conformation += 1
        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]
        data, slices = self.collate(data_list)
        print("Saving...")
        print(f"num confs {valid_conformation} num bad cases {bad_case}")
        torch.save((data, slices), self.processed_paths[0])
        os.makedirs(os.path.join(self.root, "split"), exist_ok=True)
        torch.save(
            {
                "train": torch.tensor(train_idx, dtype=torch.long),
                "valid": torch.tensor(valid_idx, dtype=torch.long),
                "test": torch.tensor(test_idx, dtype=torch.long),
            },
            os.path.join(self.root, "split", "split_dict.pt"),
        )
    def remove_h(self,mol):
        return Chem.RemoveHs(mol)
    def process_csd(self):
        print("Converting sdf files into graphs...")
        mol_list = [mol for mol in Chem.SDMolSupplier("dataset/CSD.sdf") if mol is not None]
        valid_conformation = 0
        data_list = []
        bad_case = 0
        print("random.shuffle...")
        random.shuffle(mol_list)
        total_mols = len(mol_list)
        train_size = int(total_mols * 0.8)
        valid_size = int(total_mols * 0.9)
        train_idx = []
        valid_idx = []
        test_idx = []
        print("processing...")
        for idx, mol in enumerate(tqdm(mol_list)):
            try:
                Chem.SanitizeMol(mol)
                Chem.rdmolops.AssignAtomChiralTagsFromStructure(mol)
                Chem.rdmolops.AssignStereochemistry(mol)
            except:
                print("1...")
                bad_case += 1
                continue
            if mol.GetNumConformers() > 0:  
                coords = mol.GetConformer().GetPositions()
                if all(z == 0.0 for x, y, z in coords):
                    print("2...")
                    bad_case += 1
                    continue  
            if any(len(atom.GetNeighbors()) == 0 for atom in mol.GetAtoms()):
                print("3...")
                bad_case += 1
                continue
            heavy_atoms = [atom for atom in mol.GetAtoms() if atom.GetAtomicNum() > 1]
            if len(heavy_atoms) < 2 or len(heavy_atoms) > 25:
                print("4...")
                bad_case += 1
                continue
            if "." in Chem.MolToSmiles(mol):
                print("5...")
                bad_case += 1
                continue
            if mol.GetNumBonds() < 1:
                print("6..")
                bad_case += 1
                continue
            if self.remove_hs:
                print("去除H...")
                try:
                    mol = Chem.RemoveHs(mol)
                except Exception:
                    print("发现无法去除H的分子...")
                    bad_case += 1
                    continue
            try:
                graph = self.rdk2graph(mol)
            except ValueError as e:
                print(f"Skipping molecule due to an error: {e}")
                bad_case += 1
                continue
            assert len(graph["edge_attr"]) == graph["edge_index"].shape[1]
            assert len(graph["node_feat"]) == graph["num_nodes"]
            data = CustomData()
            data.edge_index = torch.from_numpy(graph["edge_index"]).to(torch.int64)
            data.edge_attr = one_hot_bonds(torch.from_numpy(graph["edge_attr"])).to(torch.float32)
            data.x = one_hot_atoms(torch.from_numpy(graph["node_feat"])).to(torch.float32)
            data.n_nodes = graph["n_nodes"]
            data.n_edges = graph["n_edges"]
            data.pos = torch.from_numpy(mol.GetConformer(0).GetPositions()).to(torch.float)
            data.rd_mol = copy.deepcopy(mol)
            data.isomorphisms = isomorphic_core(mol)
            data.nei_src_index = torch.from_numpy(graph["nei_src_index"]).to(torch.int64)
            data.nei_tgt_index = torch.from_numpy(graph["nei_tgt_index"]).to(torch.int64)
            data.nei_tgt_mask = torch.from_numpy(graph["nei_tgt_mask"]).to(torch.bool)
            line_graph = convert_to_line_graph(data.x, data.edge_index, data.edge_attr)
            data.line_graph_n_nodes = len(line_graph.x)
            data.line_graph_pos  = torch.tensor((data.pos[data.edge_index[0]] + data.pos[data.edge_index[1]]) / 2).to(torch.float) 
            data.line_graph_isomorphisms = isomorphic_core_from_graph(torch.tensor(line_graph.x).to(torch.float32), torch.tensor(line_graph.edge_index).to(torch.int64), torch.tensor(line_graph.edge_attr).to(torch.float32))
            if idx < train_size:
                train_idx.append(len(data_list))
            elif idx < valid_size:
                valid_idx.append(len(data_list))
            else:
                test_idx.append(len(data_list))
            data_list.append(data)
            valid_conformation += 1
        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]
        data, slices = self.collate(data_list)
        print("Saving...")
        print(f"num confs {valid_conformation} num bad cases {bad_case}")
        torch.save((data, slices), self.processed_paths[0])
        os.makedirs(os.path.join(self.root, "split"), exist_ok=True)
        torch.save(
            {
                "train": torch.tensor(train_idx, dtype=torch.long),
                "valid": torch.tensor(valid_idx, dtype=torch.long),
                "test": torch.tensor(test_idx, dtype=torch.long),
            },
            os.path.join(self.root, "split", "split_dict.pt"),
        )
class CustomData(Data):
    def __cat_dim__(self, key, value, *args):
        if isinstance(value, SparseTensor):
            return (0, 1)
        elif bool(re.search("(index|face|nei_tgt_mask)", key)):
            return -1
        return 0
