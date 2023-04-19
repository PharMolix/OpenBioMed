import logging
logger = logging.getLogger(__name__)

import os
import pickle
import torch
import numpy as np

from torch_geometric.data import Data, Batch
from torch_geometric.nn import graclus, max_pool

from feat.base_featurizer import BaseFeaturizer
from utils.kg_utils import STRING

class CellTGSAFeaturizer(BaseFeaturizer):
    def __init__(self, config):
        super(CellTGSAFeaturizer, self).__init__()
        save_path = "../assets/drp/gene_graph.pkl"
        if not os.path.exists(save_path):
            logger.info("Generating gene graph...")
            self.selected_index_hugo = []
            with open("../assets/drp/selected_genes.txt", "r") as f:
                line = f.readline().strip("\n").split(",")
                for index in line:
                    self.selected_index_hugo.append(index.lstrip('(').rstrip(')'))
            self.ppi_graph = STRING("../assets/kg/STRING", config["edge_threshold"]).node_subgraph(self.selected_index_hugo)
            self.predefined_cluster = self._gen_predefined_cluster()
            pickle.dump({
                "graph": self.ppi_graph,
                "predefined_cluster": self.predefined_cluster
            }, open(save_path, "wb"))
        else:
            logger.info("Loading gene graph from cache...")
            data = pickle.load(open(save_path, "rb"))
            self.ppi_graph = data["graph"]
            self.predefined_cluster = data["predefined_cluster"]

    def _gen_predefined_cluster(self):
        g = Data(edge_index=torch.tensor(self.ppi_graph, dtype=torch.long), x=torch.zeros(len(self.selected_index_hugo), 1))
        g = Batch.from_data_list([g])
        cluster_predefine = {}
        for i in range(5):
            cluster = graclus(g.edge_index, None, g.x.size(0))
            logger.info("%d nodes at cluster level #%d" % (len(cluster.unique()), i))
            g = max_pool(cluster, g, transform=None)
            cluster_predefine[i] = cluster
        return cluster_predefine

    def __call__(self, data):
        feat = {}
        for cell_name in data:
            feat[cell_name] = Data(
                x=torch.tensor(data[cell_name], dtype=torch.float32),
                edge_index=torch.tensor(self.ppi_graph)
            )
        return feat

class CellBarFeaturizer(BaseFeaturizer):
    def __init__(self, config):
        super(CellBarFeaturizer, self).__init__()
        self.n_bars = config["n_bars"] + 2

    def __call__(self, data):
        data = data.toarray()[0]
        data[data > self.n_bars - 2] = self.n_bars - 2
        data = torch.from_numpy(data).long()
        return torch.cat((data, torch.tensor([0])))
    
class CellFullseqFeaturizer(BaseFeaturizer):
    def __init__(self, config):
        super(CellFullseqFeaturizer, self).__init__()

    def __call__(self, data):
        data = data.toarray()[0]
        data = torch.from_numpy(data)
        return data

SUPPORTED_CELL_FEATURIZER = {
    "Bar": CellBarFeaturizer,
    "Fullseq": CellFullseqFeaturizer,
    "TGSA": CellTGSAFeaturizer
}
    