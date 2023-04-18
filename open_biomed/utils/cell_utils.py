import logging
logger = logging.getLogger(__name__)

from abc import ABC, abstractmethod
import numpy as np

def load_hugo2ncbi():
    ncbi2hugo = {}
    hugo2ncbi = {}
    try:
        with open("../assets/drp/enterez_NCBI_to_hugo_gene_symbol_march_2019.txt", "r") as f:
            for line in f.readlines():
                line = line.strip("\n").split("\t")
                if len(line) > 1:
                    ncbi2hugo[line[0]] = line[1]
                    hugo2ncbi[line[1]] = line[0]
    except:
        logger.warn("NCBI2hugo gene not found")
    return ncbi2hugo, hugo2ncbi

class GeneSelector(ABC):
    def __init__(self):
        super(GeneSelector, self).__init__()

    @abstractmethod
    def __call__(self, genes, format="NCBI"):
        raise NotImplementedError

class TGSAGeneSelector(GeneSelector):
    def __init__(self):
        super(TGSAGeneSelector, self).__init__()
        self.ncbi2hugo, self.hugo2ncbi = load_hugo2ncbi()
        self.selected_index_hugo = []
        with open("../assets/drp/selected_genes.txt", "r") as f:
            line = f.readline().strip("\n").split(",")
            for index in line:
                self.selected_index_hugo.append(index.lstrip('(').rstrip(')'))

    def __call__(self, genes, format="NCBI"):
        if format == "NCBI":
            genename2id = dict(zip(genes, range(len(genes))))
            return [genename2id[self.hugo2ncbi[x]] for x in self.selected_index_hugo]

SUPPORTED_GENE_SELECTOR = {
    "TGSA": TGSAGeneSelector,
}