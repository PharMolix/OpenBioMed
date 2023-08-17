import logging
logger = logging.getLogger(__name__)

from abc import ABC, abstractmethod
import os
import json
import pandas as pd
import numpy as np
import pickle
from tqdm import tqdm
import random
import torch

from rdkit import Chem

from utils.cell_utils import load_hugo2ncbi

class KG(object):
    def  __init__(self):
        super(KG, self).__init__()

    @abstractmethod
    def __str__(self):
        raise NotImplementedError

    @abstractmethod
    def link(self, dataset):
        raise NotImplementedError

class BMKG(KG):
    def __init__(self, path):
        super(BMKG, self).__init__()
        self.drugs = json.load(open(os.path.join(path, "drug.json"), "r"))
        self.smi2drugid = {}
        for key in self.drugs:
            mol = Chem.MolFromSmiles(self.drugs[key]["SMILES"])
            if mol is not None:
                smi = Chem.MolToSmiles(mol, isomericSmiles=True)
                self.smi2drugid[smi] = key

        self.proteins = json.load(open(os.path.join(path, "protein.json"), "r"))
        self.seq2proteinid = {}
        for key in self.proteins:
            self.seq2proteinid[self.proteins[key]["sequence"]] = key

        self.edges = pd.read_csv(os.path.join(path, "links.csv"), dtype=str).values.tolist()

    def link(self, dataset):
        link_drug, link_protein = 0, 0
        drug2kg, drug2text = {}, {}
        for smi in dataset.smiles:
            iso_smi = Chem.MolToSmiles(Chem.MolFromSmiles(smi), isomericSmiles=True)
            if iso_smi in self.smi2drugid:
                link_drug += 1
                drug2kg[smi] = self.smi2drugid[iso_smi]
                drug2text[smi] = self.drugs[self.smi2drugid[iso_smi]]["text"].lower()
            else:
                drug2kg[smi] = None
                drug2text[smi] = ""
        protein2kg, protein2text = {}, {}
        for seq in dataset.proteins:
            if seq in self.seq2proteinid:
                link_protein += 1
                protein2kg[seq] = self.seq2proteinid[seq]
                protein2text[seq] = self.proteins[self.seq2proteinid[seq]]["text"].lower()
            else:
                protein2kg[seq] = None
                protein2text[seq] = ""
        logger.info("Linked drug %d/%d" % (link_drug, len(dataset.smiles)))
        logger.info("Linked protein %d/%d" % (link_protein, len(dataset.proteins)))
        return drug2kg, drug2text, protein2kg, protein2text
 
class BMKGv2(KG):
    def __init__(self, path):
        super(BMKGv2, self).__init__()
        self.kg = pickle.load(open(path, "rb"))
        self.adj = {}
        for triplet in self.kg["triplets"]:
            if triplet[0] not in self.adj:
                self.adj[triplet[0]] = [triplet]
            else:
                self.adj[triplet[0]].append(triplet)
            if triplet[2] not in self.adj:
                self.adj[triplet[2]] = [triplet]
            else:
                self.adj[triplet[2]].append(triplet)

class STRING(KG):
    def __init__(self, path, thresh=0.95):
        super(STRING, self).__init__()
        self.thresh = thresh
        _, self.hugo2ncbi = load_hugo2ncbi()
        self._load_proteins(path)
        self._load_edges(path)

    def _load_proteins(self, path):
        # self.proteins: Dict
        # Key: ensp id
        # Value: kg_id    - index in the knowledge graph
        #        name     - preferred name in HUGO
        #        sequence - amino acid sequence
        #        text     - description
        self.proteins = {}
        self.ncbi2ensp = {}
        df = pd.read_csv(os.path.join(path, "9606.protein.info.v11.0.txt"), sep='\t')
        for index, protein in df.iterrows():
            self.proteins[protein['protein_external_id']] = {
                "kg_id": index,
                "name": protein['preferred_name'],
                "text": protein['annotation']
            }
            self.ncbi2ensp[protein['preferred_name']] = protein['protein_external_id']
        # protein sequence
        with open(os.path.join(path, "9606.protein.sequences.v11.0.fa"), 'r') as f:
            id, buf = None, ''
            for line in f.readlines():
                if line.startswith('>'):
                    if id is not None:
                        self.proteins[id]["sequence"] = buf
                    id = line.lstrip('>').rstrip("\n")
                    buf = ''
                else:
                    buf = buf + line.rstrip("\n")
            
    def _load_edges(self, path):
        edges = pd.read_csv(os.path.join(path, "9606.protein.links.v11.0.txt"), sep=' ')
        selected_edges = edges['combined_score'] > (self.thresh * 1000)
        self.edges = edges[selected_edges][["protein1", "protein2"]].values.tolist()
        for i in range(len(self.edges)):
            self.edges[i][0] = self.proteins[self.edges[i][0]]["kg_id"]
            self.edges[i][1] = self.proteins[self.edges[i][1]]["kg_id"]

    def node_subgraph(self, node_idx, format="hugo"):
        if format == "hugo":
            node_idx = [self.hugo2ncbi[x] for x in node_idx]
        node_idx = [self.ncbi2ensp[x] if x in self.ncbi2ensp else x for x in node_idx]
        ensp2subgraphid = dict(zip(node_idx, range(len(node_idx))))
        names_ensp = list(self.proteins.keys())
        edge_index = []
        for i in self.edges:
            p0, p1 = names_ensp[i[0]], names_ensp[i[1]]
            if p0 in node_idx and p1 in node_idx:
                edge_index.append((ensp2subgraphid[p0], ensp2subgraphid[p1]))
                edge_index.append((ensp2subgraphid[p1], ensp2subgraphid[p0]))
        edge_index = list(set(edge_index))
        return np.array(edge_index, dtype=np.int64).T

    def __str__(self):
        return "Collected from string v11.0 database, totally %d proteins and %d edges" % (len(self.proteins), len(self.edges))

SUPPORTED_KG = {"BMKG": BMKG, "STRING": STRING}

def subgraph_sample(num_nodes, edge_index, strategy, num_samples, directed=False):
    ### Inputs:
    # edge_index: edge index
    # strategy: sampling strategy, e.g. bfs
    ### Output:
    # indexes of sampled edges
    adj = []
    for i in range(num_nodes):
        adj.append([])
    for i, edge in enumerate(edge_index):
        adj[edge[0]].append(i)
    node_queue = []
    visited = [0] * num_nodes
    selected_edges = []
    
    random_node = random.randint(0, num_nodes - 1)
    while len(adj[random_node]) > 5:
        random_node = random.randint(0, num_nodes - 1)
    node_queue.append(random_node)
    visited[random_node] = 1

    def dfs(u):
        visited[u] = 1
        for i in adj[u]:
            if i not in selected_edges:
                selected_edges.append(i)
                selected_edges.append(i ^ 1)
            if len(selected_edges) >= num_samples:
                return
        for i in adj[u]:
            v = edge_index[i][1]
            if visited[v]:
                continue
            dfs(v)
            if len(selected_edges) >= num_samples:
                return

    if strategy == 'dfs':
        dfs(random_node)
    else:
        while len(selected_edges) < num_samples:
            u = node_queue.pop(0)
            for i in adj[u]:
                v = edge_index[i][1]
                if i not in selected_edges:
                    selected_edges.append(i)
                    selected_edges.append(i ^ 1)
                if not visited[v]:
                    visited[v] = 1
                    node_queue.append(v)

    return selected_edges

def embed(graph, model='ProNE', filter_out={}, dim=256, save=True, save_path=''):
    ### Inputs:
    # G: object of KG
    # model: network embedding model, e.g. ProNE
    ### Outputs:
    # emb: numpy array, |G| * dim
    if save and os.path.exists(save_path):
        logger.info("Load KGE from saved file.")
        return pickle.load(open(save_path, "rb"))

    from cogdl.data import Adjacency
    from cogdl.models.emb.prone import ProNE
    name2id = {}
    cnt = 0
    filtered = 0
    row = []
    col = []
    for h, t, r in graph.edges:
        if (h, t) in filter_out:
            filtered += 1
            continue
        if h not in name2id:
            cnt += 1
            name2id[h] = cnt
        if t not in name2id:
            cnt += 1
            name2id[t] = cnt
        row.append(name2id[h])
        col.append(name2id[t])
    logger.info("Filtered out %d edges in val/test set" % (filtered))
        
    row = torch.tensor(row)
    col = torch.tensor(col)
    graph = Adjacency(row, col)
    emb_model = ProNE(dim, 5, 0.2, 0.5)
    logger.info("Generating KGE...")
    emb = emb_model(graph)
    
    kg_emb = {}
    for key in name2id:
        kg_emb[key] = emb[name2id[key]]

    if save:
        pickle.dump(kg_emb, open(save_path, "wb"))

    return kg_emb

def bfs(graph, node_id, max_depth):
    ### Inputs:
    # G: object of KG
    # node_id: the id of the starting node
    # max_depth: the max number of steps to go
    ### Outputs:
    # dist: a list, dist[i] is the list of i-hop neighbors
    pass