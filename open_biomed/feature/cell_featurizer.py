import logging
logger = logging.getLogger(__name__)

import os
import pickle
import torch
import numpy as np

from torch_geometric.data import Data, Batch
from torch_geometric.nn import graclus, max_pool

from open_biomed.feature.base_featurizer import BaseFeaturizer
from open_biomed.utils.kg_utils import STRING

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
        
class CellTensorDictFeaturizer(BaseFeaturizer):
    def __init__(self, config):
        super(CellTensorDictFeaturizer, self).__init__()

    def __call__(self, data):
        for k in data:
            data[k] = torch.from_numpy(data[k])
        return data

class CellBarDictFeaturizer(BaseFeaturizer):
    def __init__(self, config):
        super(CellBarDictFeaturizer, self).__init__()
        self.n_bars = config["n_bars"] + 2

    def __call__(self, data):
        for k in data:
            d = data[k]
            d[d > self.n_bars - 2] = self.n_bars - 2
            d = torch.from_numpy(d).long()
            data[k] = torch.cat((d, torch.tensor([0])))
        return data

class CellGeneFormerFeaturizer(BaseFeaturizer):
    def __init__(self, config):
        super(CellGeneFormerFeaturizer, self).__init__()

    def __call__(self, data):
        data.obs['n_counts'] = data.X.sum(axis=1)
        data.obs['filter_pass'] = True
        data.var['ensembl_id'] = data.var_names

        from geneformer import TranscriptomeTokenizer
        from geneformer.tokenizer import tokenize_cell

        class MyTranscriptomeTokenizer(TranscriptomeTokenizer):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)

            def tokenize_anndata(self, data):
                if self.custom_attr_name_dict is not None:
                    file_cell_metadata = {
                        attr_key: [] for attr_key in self.custom_attr_name_dict.keys()
                    }

                # with lp.connect(str(loom_file_path)) as data:
                    # define coordinates of detected protein-coding or miRNA genes and vector of their normalization factors
                coding_miRNA_loc = np.where(
                    [self.genelist_dict.get(i, False) for i in data.var["ensembl_id"]]
                )[0]
                norm_factor_vector = np.array(
                    [
                        self.gene_median_dict[i]
                        for i in data.var["ensembl_id"][coding_miRNA_loc]
                    ]
                )
                coding_miRNA_ids = data.var["ensembl_id"][coding_miRNA_loc]
                coding_miRNA_tokens = np.array(
                    [self.gene_token_dict[i] for i in coding_miRNA_ids]
                )

                # define coordinates of cells passing filters for inclusion (e.g. QC)
                try:
                    data.obs["filter_pass"]
                except AttributeError:
                    var_exists = False
                else:
                    var_exists = True

                if var_exists is True:
                    filter_pass_loc = np.where(
                        [True if i == 1 else False for i in data.obs["filter_pass"]]
                    )[0]
                elif var_exists is False:
                    print(
                        f"data has no column attribute 'filter_pass'; tokenizing all cells."
                    )
                    filter_pass_loc = np.array([i for i in range(data.shape[0])])

                # scan through .loom files and tokenize cells
                tokenized_cells = []

                # for (_ix, _selection, view) in data.scan(items=filter_pass_loc, axis=1):
                #     # select subview with protein-coding and miRNA genes
                #     subview = view.view[coding_miRNA_loc, :]
                subview = data[filter_pass_loc, coding_miRNA_loc]

                # normalize by total counts per cell and multiply by 10,000 to allocate bits to precision
                # and normalize by gene normalization factors
                subview_norm_array = (
                    subview.X.toarray().T
                    / subview.obs.n_counts.to_numpy()
                    * 10_000
                    / norm_factor_vector[:, None]
                )
                # tokenize subview gene vectors
                tokenized_cells += [
                    tokenize_cell(subview_norm_array[:, i], coding_miRNA_tokens)
                    for i in range(subview_norm_array.shape[1])
                ]

                # add custom attributes for subview to dict
                if self.custom_attr_name_dict is not None:
                    for k in file_cell_metadata.keys():
                        file_cell_metadata[k] += subview.obs[k].tolist()
                else:
                    file_cell_metadata = None

                return tokenized_cells, file_cell_metadata

        tokenizer = MyTranscriptomeTokenizer(dict([(k, k) for k in {"n_counts", "filter_pass"}]), nproc=4)
        tokenized_cells, cell_metadata = tokenizer.tokenize_anndata(data)
        tokenized_cells = [cell[:2048] for cell in tokenized_cells]
        #tokenized_dataset = tokenizer.create_dataset(tokenized_cells, cell_metadata)
        #return tokenized_dataset
        return tokenized_cells, cell_metadata

SUPPORTED_CELL_FEATURIZER = {
    "Bar": CellBarFeaturizer,
    "Fullseq": CellFullseqFeaturizer,
    "TGSA": CellTGSAFeaturizer,
    "TensorDict": CellTensorDictFeaturizer,
    "BarDict": CellBarDictFeaturizer,
    "GeneFormer": CellGeneFormerFeaturizer,
}
    