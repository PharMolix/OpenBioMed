import numpy as np

import torch
import torch.nn as nn

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from open_biomed.models.base_models import MolEncoder, TextEncoder

def msra_initialization(m):
    """
    MSRA initialization of the weights of a :class:`torch.nn.Module` (a layer),
    that is, the weights of the layer are :math:`\\mathbf{W} \\sim N(0, 2 / D)`,
    where :math:`D` is the incoming dimension. For more details see
    `He's paper`_.

    .. _`He's paper`: https://arxiv.org/abs/1502.01852

    Parameters
    ----------
    m: :class:`torch.nn.Module`
        Module (layer) whose weights should be normalized.
    """
    nn.init.normal_(m.weight, mean=0., std=np.sqrt(2. / m.in_features))
    nn.init.zeros_(m.bias)

class MultilayerPerceptron(nn.Module):
    """
    Feed-forward neural network with `feature_size` input units, `num_targets`
    output units, and hidden layers given by the list `hidden_layer_sizes`.
    The input layer and all hidden layers share the following generic structure

    .. math::

        \\text{dropout} \\Big( f \\big( \\text{norm}(W x + b) \\big) \\Big) \\text{,}

    where

    - :math:`x` is the input to the layer,
    - :math:`W` and :math:`b` are learnable weights,
    - :math:`\\text{norm}` is a placeholder for a normalization layer (leave
      empty for no normalization),
    - :math:`f` is a placeholder for an activation function (leave empty for no
      non-linearity),
    - :math:`\\text{dropout}` is a placeholder for a dropout layer (leave empty
      for no dropout).

    The output layer is not followed by normalization, non-linearity (this will
    be included in the loss function), nor dropout.
    """

    def __init__(self, feature_size, hidden_layer_sizes, num_targets, dropout_input=.0, dropout_hidden=.0, nonlinearity='Identity'):
        super().__init__()

        # linear layers
        self.linear_input = nn.Linear(feature_size, hidden_layer_sizes[0])
        self.linear_hidden_l = nn.ModuleList(
            [nn.Linear(s, spp) for s, spp in zip(hidden_layer_sizes[:-1], hidden_layer_sizes[1:])]
        )
        self.linear_output = nn.Linear(hidden_layer_sizes[-1], num_targets)

        # normalization layers (placeholders)
        self.normalization_input = nn.Identity()
        self.normalization_hidden_l = nn.ModuleList(
            [nn.Identity() for _ in hidden_layer_sizes[1:]]
        )
        assert len(self.linear_hidden_l) == len(self.normalization_hidden_l), 'Something went wrong initializing the hidden layers.'

        # non-linearity and dropout (placeholders)
        self.nonlinearity = getattr(nn, nonlinearity)()
        self.dropout_input = nn.Dropout(p=dropout_input)
        self.dropout_hidden = nn.Dropout(p=dropout_hidden)
        self.num_weight_matrices = len(hidden_layer_sizes) + 1

    def forward(self, x):
        x = self.linear_input(x)
        x = self.normalization_input(x)
        x = self.nonlinearity(x)
        x = self.dropout_input(x)
        if len(self.linear_hidden_l) > 0:
            for linear_hidden, normalization_hidden in zip(self.linear_hidden_l, self.normalization_hidden_l):
                x = linear_hidden(x)
                x = normalization_hidden(x)
                x = self.nonlinearity(x)
                x = self.dropout_hidden(x)
        x = self.linear_output(x)
        return x

    def initialize_weights(self, init):
        """
        Initialize all the weights using the method `init`.
        """
        init(self.linear_input)
        if len(self.linear_hidden_l) > 0:
            for i, _ in enumerate(self.linear_hidden_l):
                init(self.linear_hidden_l[i])
        init(self.linear_output)

class NetworkLayerNorm(MultilayerPerceptron):
    """
    Child class of :class:`MultilayerPerceptron` where

    - normalization layers are set to :class:`~torch.nn.LayerNorm`,
    - non-linearity is set to :class:`~torch.nn.__` which can be set by the argument nonlinearity,
    - dropout layers are set to :class:`~torch.nn.Dropout`,

    and the weights are initialized using :meth:`msra_initialization`.
    """
    def __init__(self, feature_size, hidden_layer_sizes, num_targets, dropout_input, dropout_hidden, 
    nonlinearity='ReLU'):
        super().__init__(feature_size, hidden_layer_sizes, num_targets)
        self.normalization_input = nn.LayerNorm(
            normalized_shape=self.linear_input.out_features,
            elementwise_affine=False
        )
        for i, linear_hidden in enumerate(self.linear_hidden_l):
            self.normalization_hidden_l[i] = nn.LayerNorm(
                normalized_shape=linear_hidden.out_features,
                elementwise_affine=False
            )
        self.nonlinearity = getattr(nn, nonlinearity if nonlinearity else 'ReLU')()
        self.dropout_input = nn.Dropout(p=dropout_input)
        self.dropout_hidden = nn.Dropout(p=dropout_hidden)
        self.initialize_weights(init=msra_initialization)

class CLAMP(MolEncoder, TextEncoder):
    def __init__(self, config):
        super(CLAMP, self).__init__()
        self.compound_features_size = config["compound_features_size"]
        self.assay_features_size = config["assay_features_size"]
        self.embedding_size = config["embedding_size"]

        self.compound_encoder = NetworkLayerNorm(
            feature_size=self.compound_features_size,
            hidden_layer_sizes=config["compound_layer_sizes"],
            num_targets=self.embedding_size,
            dropout_input=config["dropout_input"],
            dropout_hidden=config["dropout_hidden"],
            nonlinearity=config["nonlinearity"]
        )
        self.assay_encoder = NetworkLayerNorm(
            feature_size=self.assay_features_size,
            hidden_layer_sizes=config["assay_layer_sizes"],
            num_targets=self.embedding_size,
            dropout_input=config["dropout_input"],
            dropout_hidden=config["dropout_hidden"],
            nonlinearity=config["nonlinearity"]
        )
        import clip
        self.text_encoder, _ = clip.load("ViT-B/32", device="cpu", download_root=config["clip_path"])
        self.norm = False

    def encode_mol(self, mol):
        return self.compound_encoder(mol)

    def encode_text(self, text):
        h = self.text_encoder.encode_text(text).float()
        return self.assay_encoder(h)

if __name__ == "__main__":
    import json
    from utils.data_utils import DataProcessorFast
    molecules = [
        'CCOP(=O)(Nc1cccc(Cl)c1)OCC', #inactive
        'O=C(O)c1ccccc1O', #inactive
        'NNP(=S)(NN)c1ccccc1', #active
        'CC(=O)OC1=CC=CC=C1C(=O)O', # Aspirin
    ]
    assay_descriptions = [
        'HIV: Experimentally measured abilities to inhibit HIV replication.',
    ]
    config = json.load(open("./configs/mtr/clamp.json", "r"))
    processor_mol = DataProcessorFast(entity_type="molecule", config=config["data"]["mol"])
    processor_text = DataProcessorFast(entity_type="text", config=config["data"]["text"])
    molecules = processor_mol(molecules)
    texts = processor_text(assay_descriptions)
    model = CLAMP(config["network"])
    state_dict = torch.load("./ckpts/fusion_ckpts/clamp/checkpoint.pt", map_location="cpu")["model_state_dict"]
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    print(molecules.sum(dim=1), texts)

    with torch.no_grad():
        mol_feats = model.encode_mol(molecules)
        text_feats = model.encode_text(texts)
        mol_feats = mol_feats / (torch.norm(mol_feats, dim=1, keepdim=True) + 1e-13)
        text_feats = text_feats / (torch.norm(text_feats, dim=1, keepdim=True) + 1e-13)
        sim = mol_feats @ text_feats.T
        print(sim.softmax(dim=0))