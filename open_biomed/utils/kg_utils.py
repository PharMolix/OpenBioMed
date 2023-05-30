import logging
logger = logging.getLogger(__name__)

from abc import ABC, abstractmethod
import os
import csv
import json
import pandas as pd
import numpy as np
import pickle
from tqdm import tqdm
import torch

from rdkit import Chem, DataStructs
import json
from tqdm import tqdm
import networkx as nx

from utils.cell_utils import load_hugo2ncbi
from utils.mol_utils import valid_smiles

from mhfp.encoder import MHFPEncoder
from mhfp.lsh_forest import LSHForestHelper



class KG(object):
    def  __init__(self):
        super(KG, self).__init__()
        self.drugs = None
        self.proteins = None
        self.edges = None
        self.drugs_dict = {}
        self.proteins_dict = {}
        self.G = nx.Graph()
        self.kg_embedding = None

    @abstractmethod
    def __str__(self):
        raise NotImplementedError

    @abstractmethod
    def link(self, dataset):
        raise NotImplementedError


class BMKG(KG):
    def __init__(self, path):
        super(BMKG, self).__init__()
        self.drugs = json.load(open(os.path.join(path, "bmkg-dp_drug.json"), "r"))
        self.smi2drugid = {}
        for key in self.drugs:
            # TODO: allchem
            mol = Chem.MolFromSmiles(self.drugs[key]["SMILES"])
            if mol is not None:
                smi = Chem.MolToSmiles(mol, isomericSmiles=True)
                self.smi2drugid[smi] = key

        self.proteins = json.load(open(os.path.join(path, "bmkg-dp_protein.json"), "r"))
        self.seq2proteinid = {}
        for key in self.proteins:
            self.seq2proteinid[self.proteins[key]["sequence"]] = key

        self.edges = pd.read_csv(os.path.join(path, "kg_data.csv"), dtype=str).values.tolist()

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
        if hasattr(dataset, "proteins"):
            for seq in dataset.proteins:
                if seq in self.seq2proteinid:
                    link_protein += 1
                    protein2kg[seq] = self.seq2proteinid[seq]
                    protein2text[seq] = self.proteins[self.seq2proteinid[seq]]["text"].lower()
                else:
                    protein2kg[seq] = None
                    protein2text[seq] = "No description for the protein is available."
            logger.info("Linked proteien %d/%d" % (link_protein, len(dataset.proteins)))
        logger.info("Linked drug %d/%d" % (link_drug, len(dataset.smiles)))

        return drug2kg, drug2text, protein2kg, protein2text


class BMKGv2(KG):
    
    def __init__(self, path):
        super(BMKGv2, self).__init__()
        # triplets, ent_dict, rel_dict, n_ent=49111, n_rel=16
        self.kg = pickle.load(open(os.path.join(path, "kg/kg.pkl"), "rb"))
        self.adj = {}
        # TODO: 这里smiles没做标准化
        for triplet in self.kg["triplets"]:
            if triplet[0] not in self.adj:
                self.adj[triplet[0]] = [triplet]
            else:
                self.adj[triplet[0]].append(triplet)
            if triplet[2] not in self.adj:
                self.adj[triplet[2]] = [triplet]
            else:
                self.adj[triplet[2]].append(triplet)
        self.smi2drugid = {}
        # eg ['24492', 'B(=O)O'] kgid and smiles
        with open(os.path.join(path, "pair.txt"), encoding="utf-8") as f:
            while True:
                tmp_list = f.readline().replace('\n','').split("\t")
                if len(tmp_list) == 2:
                    kg_key, smiles = tmp_list[0], tmp_list[1]
                    iso_smi = Chem.MolToSmiles(Chem.MolFromSmiles(smiles), isomericSmiles=True)
                    if iso_smi and iso_smi not in self.smi2drugid.keys():
                        # get the kg index of smiles
                        # TODO: we
                        try:
                            self.smi2drugid[iso_smi] = self.kg["ent_dict"][kg_key]
                        except Exception as e:
                            self.smi2drugid[iso_smi] = -1
                else:
                    break
        
        # iso_smiles
        self.kg_smiles = list(self.smi2drugid.keys())
        
        # load chebi20 text
        self.chebi20_smiles = []
        self.chebi20_texts = []
        # TODO: hard code
        self.chebi20_path = "../datasets/molcap/chebi-20"
        split_list = ["train", "validation", "test"]
        for split in split_list:
            with open(os.path.join(self.chebi20_path, split + ".txt")) as f:
                reader = csv.DictReader(f, delimiter='\t', quoting=csv.QUOTE_NONE)
                for line in reader:
                    # TODO:
                    iso_smi = Chem.MolToSmiles(Chem.MolFromSmiles(line["SMILES"]), isomericSmiles=True)
                    if iso_smi:
                        self.chebi20_smiles.append(iso_smi)
                        self.chebi20_texts.append(line["description"])
        self.smi2text = dict(zip(self.chebi20_smiles, self.chebi20_texts))
        
        # build_Query
        self.mhfp_encoder = MHFPEncoder()
        self.lsh_forest_helper = LSHForestHelper()
        self.fps = []
        for i, smi in enumerate(self.kg_smiles):
            fp = self.mhfp_encoder.encode(smi)
            self.lsh_forest_helper.add(i, fp)
            self.fps.append(fp)
        self.lsh_forest_helper.index()
        
    def link(self, dataset):
        link_drug = 0
        drug2kg, drug2text = {}, {}
        # TODO: 这里是不是应该用dataset.drugs
        for smi in dataset.smiles:
            # TODO:
            iso_smi = Chem.MolToSmiles(Chem.MolFromSmiles(smi), isomericSmiles=True)
            if iso_smi in self.smi2drugid:
                link_drug += 1
                drug2kg[smi] = self.smi2drugid[iso_smi]
                # TODO: add text later
                # drug2text[smi] = self.drugs[self.smi2drugid[iso_smi]]["text"]
                drug2text[smi] = "No description for the drug is available."
            else:
                drug2kg[smi] = None
                drug2text[smi] = "No description for the drug is available."
        logger.info("Linked drug %d/%d" % (link_drug, len(dataset.smiles)))
        return drug2kg, drug2text, None, None

    def link_chebi20(self, dataset, neighbour=5):
        len_drugs = len(dataset.smiles)
        drug2kg, drug2text = {}, {}
        kg_linked = 0
        neighbour_linked = 0
        text_linked = 0
        for i, raw_smi in enumerate(dataset.smiles):
            # add kg index
            # TODO: 这里要确定一下dataset里的smiles是否已经经过了iso化
            # iso_smi = Chem.MolToSmiles(Chem.MolFromSmiles(smi), isomericSmiles=True)
            drug2kg[raw_smi] = []
            smi = Chem.MolToSmiles(Chem.MolFromSmiles(raw_smi), isomericSmiles=True)
            if smi in self.kg_smiles:
                drug2kg[raw_smi].append(self.smi2drugid[smi])
                kg_linked += 1
            else:
                # 这里只对最近的5个邻居做索引
                fp = self.mhfp_encoder.encode(smi)
                knn = self.lsh_forest_helper.query(fp, neighbour, self.fps)[0:]
                fp1 = Chem.RDKFingerprint(Chem.MolFromSmiles(smi), fpSize=2048)
                for j in knn:
                    # TODO: 
                    fp2 = Chem.RDKFingerprint(Chem.MolFromSmiles(self.kg_smiles[j]), fpSize=2048)
                    sim = DataStructs.FingerprintSimilarity(fp1, fp2)
                    # TODO: 阈值可调
                    if sim > 0.6:
                        drug2kg[raw_smi].append(self.smi2drugid[self.kg_smiles[j]])
                if len(drug2kg[raw_smi]) != 0:
                    neighbour_linked += 1
            if len(drug2kg[raw_smi]) == 0:
                drug2kg[raw_smi] = [-1]

            # add text
            if smi in self.chebi20_smiles:
                drug2text[raw_smi] = self.smi2text[smi]
                text_linked += 1
            else:
                drug2text[raw_smi] = "No description for the drug is available."
        kg_link_ratio = round(kg_linked/len_drugs, 2)
        neighbour_link_ratio = round(neighbour_linked/len_drugs, 2)
        text_link_ratio = round(text_linked/len_drugs, 2)
        print(f"total {len_drugs} drugs and {kg_link_ratio} linked directly, {neighbour_link_ratio} linked by neighbours, text linked {text_link_ratio}")
        return drug2kg, drug2text, None, None
    
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


class BMKGV3(KG):
    def __init__(self, path):
        super(BMKG, self).__init__()
        """
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
        """
        #self.load_drugs(osp.join(path, "bmkg-dp_drug.json"))
        #self.load_proteins(osp.join(path, "bmkg-dp_protein.json"))
        #self.load_edges(osp.join(path, "edge.csv"))
        self.kg_embedding = pickle.load(open(path + "/" + "kg_embed_ace2.pickle", "rb"))
        self.load_drugs(osp.join(path, "bmkg-dp_drug.json"), save=True, save_path="")
        self.load_proteins(osp.join(path, "bmkg-dp_protein.json"), save=True, save_path="")
        # self.load_edges(osp.join(path, "kg_data.csv"))

    # TODO: 现在默认json里边存的是个dict，key是id，value是一个dict，里面包含各种信息
    def load_node(self, path):
        with open(path, "r") as f:
            node_dict = json.load(f)
        return node_dict

    def load_drugs(self, path, save=False, save_path=None):
        self.drugs = self.load_node(path)
        for key, value in self.drugs.items():
            smile = value["SMILES"]            
            self.drugs_dict[smile] = {"bmkg_id": str(key), "text": value["text"], "fingerprint": value["fingerprint"]}           
        if save:
            if not save_path:
                save_path = osp.join(osp.dirname(path), "SMILES_dict.json")
            with open(save_path, 'w') as f:
                json.dump(self.drugs_dict, f)

    def load_proteins(self, path, save=False, save_path=None):
        self.proteins = self.load_node(path)
        for key, value in self.proteins.items():
            seq = value["sequence"]
            self.proteins_dict[seq] = {"bmkg_id": str(key), "text": value["text"], "descriptor": value["descriptor"]}
        if save:
            if not save_path:
                save_path = osp.join(osp.dirname(path), "seqs_dict.json")
            with open(save_path, 'w') as f:
                json.dump(self.proteins_dict, f)
                
    # TODO: 发现有些id不在self.drugs和self.proteins里
    def get_node_info(self, id):
        node, text, feature = None, None, None
        try:
            if isinstance(id, str) and id.startswith("DB"):
                node = self.drugs[id]["SMILES"]
                text = self.drugs[id]["text"]
                feature = self.drugs[id]["fingerprint"]
            else:
                node = self.proteins[id]["sequence"]
                text = self.proteins[id]["text"]
                feature = self.proteins[id]["descriptor"]
            return node, text, feature
        except Exception as e:
            # print(e)
            return node, text, feature
          
    def load_edges(self, path):
        edge_data = pd.read_csv(path, delimiter=',')
        print(f"The shape of KG is {edge_data.shape}")
        # edges = edge_data.values.tolist()
        for index, edge in tqdm(edge_data.iterrows()):
            head, tail = str(edge['x_id']), str(edge['end_id'])
            self.G.add_edge(head, tail)
            
    def save_graph(self, path):
        pass

    def get_drug(self, smi, radius=2):
        try:
            drug = self.drugs_dict[smi]
            drug_id = drug["bmkg_id"]
            drug_embedding = self.kg_embedding[drug_id]
            # drug_graph = nx.ego_graph(self.G, drug_id, radius)
            drug_graph = None
            return (drug, drug_graph, drug_embedding)
        except Exception as e:
            # print(e)
            return (None, None, None)
            
    def get_finger(self, smile):
        mols =Chem.MolFromSmiles(smile)
        fp = AllChem.GetMorganFingerprintAsBitVect(mols,2,nBits=1024,)
        fp = list(fp)
        return fp
        
    def get_cos_similar(self, v1, v2):
        num = float(np.dot(v1, v2))  # 向量点乘
        denom = np.linalg.norm(v1) * np.linalg.norm(v2)  # 求模长的乘积
        return 0.5 + 0.5 * (num / denom) if denom != 0 else 0
        
    def get_drug_wfin(self, smi, radius=2):
        fin = self.get_finger(smi)
        drug_id = ''
        max_equ = 0.9
        for kg_smi,drug in self.drugs_dict.items():
            kg_fin = drug['fingerprint']
            equ_val = self.get_cos_similar(fin,kg_fin)
            if equ_val>max_equ and drug['bmkg_id'] in self.kg_embedding.keys():
                max_equ = equ_val
                drug_id = drug['bmkg_id']
                drug_use = drug
        if drug_id != '':
            drug_embedding = self.kg_embedding[drug_id]
            # drug_graph = nx.ego_graph(self.G, drug_id, radius)
            drug_graph = None
            return (drug_use, drug_graph, drug_embedding)
        else:
            return (None, None, None)
        
    
    def string_similar(self, s1, s2):
        return difflib.SequenceMatcher(None, s1, s2).quick_ratio()
        
    def get_drug_wseqsim(self, smi, radius=2):
        drug_id = ''
        max_equ = 0.9
        for kg_smi,drug in self.drugs_dict.items():
            equ_val = self.string_similar(smi,kg_smi)
            if equ_val>max_equ and drug['bmkg_id'] in self.kg_embedding.keys():
                max_equ = equ_val
                drug_id = drug['bmkg_id']
                drug_use = drug
        if drug_id != '':
            drug_embedding = self.kg_embedding[drug_id]
            # drug_graph = nx.ego_graph(self.G, drug_id, radius)
            drug_graph = None
            return (drug_use, drug_graph, drug_embedding)
        else:
            return (None, None, None)
    
    # TODO:这里还没有进行对比seq是否相同的逻辑
    def get_protein(self, seq, radius=2):
        try:
            protein = self.proteins_dict[seq]
            protein_id = protein["bmkg_id"]
            protein_embedding = self.kg_embedding[protein_id]
            # protein_graph = nx.ego_graph(self.G, protein, radius)
            protein_graph = None
            return (protein, protein_graph, protein_embedding)
        except Exception as e:
            # print(e) 
            return (None, None, None)

    def model_train():
        pass

    def __str__(self):
        pass


        self.edges = pd.read_csv(os.path.join(path, "links.csv"), dtype=str).values.tolist()

    def link(self, dataset):
        link_drug, link_protein = 0, 0
        drug2kg, drug2text = {}, {}
        for smi in dataset.smiles:
            # TODO: 后续换成标准的smiles2drug API
            iso_smi = Chem.MolToSmiles(Chem.MolFromSmiles(smi), isomericSmiles=True)
            if iso_smi in self.smi2drugid:
                link_drug += 1
                drug2kg[smi] = self.smi2drugid[iso_smi]
                drug2text[smi] = self.drugs[self.smi2drugid[iso_smi]]["text"]
            else:
                drug2kg[smi] = None
                drug2text[smi] = "No description for the drug is available."
        protein2kg, protein2text = {}, {}
        if hasattr(dataset, "proteins"):
            for seq in dataset.proteins:
                if seq in self.seq2proteinid:
                    link_protein += 1
                    protein2kg[seq] = self.seq2proteinid[seq]
                    protein2text[seq] = self.proteins[self.seq2proteinid[seq]]["text"]
                else:
                    protein2kg[seq] = None
                    protein2text[seq] = "No description for the protein is available."
        logger.info("Linked drug %d/%d" % (link_drug, len(dataset.smiles)))
        logger.info("Linked proteien %d/%d" % (link_protein, len(dataset.proteins)))
        return drug2kg, drug2text, protein2kg, protein2text

SUPPORTED_KG = {"BMKG": BMKG, "BMKGV2": BMKGv2, "STRING": STRING}

def sample(graph, node_id, sampler):
    ### Inputs:
    # G: object of KG
    # node_id: the id of the center node
    # sampler: sampling strategy, e.g. ego-net
    ### Outputs:
    # G': graph in pyg Data(x, y, edge_index)
    pass

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