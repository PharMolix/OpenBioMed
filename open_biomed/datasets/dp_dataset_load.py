import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors


def _load_bbbp_dataset(input_path):
    input_df = pd.read_csv(input_path, sep=',')
    smiles_list = input_df['smiles']
    rdkit_mol_objs_list = [AllChem.MolFromSmiles(s) for s in smiles_list]

    preprocessed_rdkit_mol_objs_list = [m if m is not None else None
                                        for m in rdkit_mol_objs_list]
    preprocessed_smiles_list = [AllChem.MolToSmiles(m) if m is not None else None
                                for m in preprocessed_rdkit_mol_objs_list]
    labels = input_df['p_np']
    # convert 0 to -1
    labels = labels.replace(0, -1)
    # there are no nans
    assert len(smiles_list) == len(preprocessed_rdkit_mol_objs_list)
    assert len(smiles_list) == len(preprocessed_smiles_list)
    assert len(smiles_list) == len(labels)
    return smiles_list, \
           preprocessed_rdkit_mol_objs_list, labels.values


def _load_clintox_dataset(input_path):
    input_df = pd.read_csv(input_path, sep=',')
    smiles_list = input_df['smiles']
    rdkit_mol_objs_list = [AllChem.MolFromSmiles(s) for s in smiles_list]

    preprocessed_rdkit_mol_objs_list = [m if m is not None else None
                                        for m in rdkit_mol_objs_list]
    preprocessed_smiles_list = [AllChem.MolToSmiles(m) if m is not None else None
                                for m in preprocessed_rdkit_mol_objs_list]
    tasks = ['FDA_APPROVED', 'CT_TOX']
    labels = input_df[tasks]
    # convert 0 to -1
    labels = labels.replace(0, -1)
    # there are no nans
    assert len(smiles_list) == len(preprocessed_rdkit_mol_objs_list)
    assert len(smiles_list) == len(preprocessed_smiles_list)
    assert len(smiles_list) == len(labels)
    return smiles_list, \
           preprocessed_rdkit_mol_objs_list, labels.values


# input_path = 'dataset/clintox/raw/clintox.csv'
# smiles_list, rdkit_mol_objs_list, labels = _load_clintox_dataset(input_path)

def _load_esol_dataset(input_path):
    # NB: some examples have multiple species
    input_df = pd.read_csv(input_path, sep=',')
    smiles_list = input_df['smiles']
    rdkit_mol_objs_list = [AllChem.MolFromSmiles(s) for s in smiles_list]
    labels = input_df['measured log solubility in mols per litre']
    assert len(smiles_list) == len(rdkit_mol_objs_list)
    assert len(smiles_list) == len(labels)
    return smiles_list, rdkit_mol_objs_list, labels.values


# input_path = 'dataset/esol/raw/delaney-processed.csv'
# smiles_list, rdkit_mol_objs_list, labels = _load_esol_dataset(input_path)

def _load_freesolv_dataset(input_path):

    input_df = pd.read_csv(input_path, sep=',')
    smiles_list = input_df['smiles']
    rdkit_mol_objs_list = [AllChem.MolFromSmiles(s) for s in smiles_list]
    labels = input_df['expt']
    assert len(smiles_list) == len(rdkit_mol_objs_list)
    assert len(smiles_list) == len(labels)
    return smiles_list, rdkit_mol_objs_list, labels.values


def _load_lipophilicity_dataset(input_path):

    input_df = pd.read_csv(input_path, sep=',')
    smiles_list = input_df['smiles']
    rdkit_mol_objs_list = [AllChem.MolFromSmiles(s) for s in smiles_list]
    labels = input_df['exp']
    assert len(smiles_list) == len(rdkit_mol_objs_list)
    assert len(smiles_list) == len(labels)
    return smiles_list, rdkit_mol_objs_list, labels.values


def _load_malaria_dataset(input_path):

    input_df = pd.read_csv(input_path, sep=',')
    smiles_list = input_df['smiles']
    rdkit_mol_objs_list = [AllChem.MolFromSmiles(s) for s in smiles_list]
    labels = input_df['activity']
    assert len(smiles_list) == len(rdkit_mol_objs_list)
    assert len(smiles_list) == len(labels)
    return smiles_list, rdkit_mol_objs_list, labels.values


def _load_cep_dataset(input_path):

    input_df = pd.read_csv(input_path, sep=',')
    smiles_list = input_df['smiles']
    rdkit_mol_objs_list = [AllChem.MolFromSmiles(s) for s in smiles_list]
    labels = input_df['PCE']
    assert len(smiles_list) == len(rdkit_mol_objs_list)
    assert len(smiles_list) == len(labels)
    return smiles_list, rdkit_mol_objs_list, labels.values


def _load_muv_dataset(input_path):

    input_df = pd.read_csv(input_path, sep=',')
    smiles_list = input_df['smiles']
    rdkit_mol_objs_list = [AllChem.MolFromSmiles(s) for s in smiles_list]
    tasks = ['MUV-466', 'MUV-548', 'MUV-600', 'MUV-644', 'MUV-652', 'MUV-689',
             'MUV-692', 'MUV-712', 'MUV-713', 'MUV-733', 'MUV-737', 'MUV-810',
             'MUV-832', 'MUV-846', 'MUV-852', 'MUV-858', 'MUV-859']
    labels = input_df[tasks]
    # convert 0 to -1
    labels = labels.replace(0, -1)
    # convert nan to 0
    labels = labels.fillna(0)
    assert len(smiles_list) == len(rdkit_mol_objs_list)
    assert len(smiles_list) == len(labels)
    return smiles_list, rdkit_mol_objs_list, labels.values


def check_columns(df, tasks, N):
    bad_tasks = []
    total_missing_count = 0
    for task in tasks:
        value_list = df[task]
        pos_count = sum(value_list == 1)
        neg_count = sum(value_list == -1)
        missing_count = sum(value_list == 0)
        total_missing_count += missing_count
        pos_ratio = 100. * pos_count / (pos_count + neg_count)
        missing_ratio = 100. * missing_count / N
        assert pos_count + neg_count + missing_count == N
        if missing_ratio >= 50:
            bad_tasks.append(task)
        print('task {}\t\tpos_ratio: {:.5f}\tmissing ratio: {:.5f}'.format(task, pos_ratio, missing_ratio))
    print('total missing ratio: {:.5f}'.format(100. * total_missing_count / len(tasks) / N))
    return bad_tasks


def check_rows(labels, N):
    from collections import defaultdict
    p, n, m = defaultdict(int), defaultdict(int), defaultdict(int)
    bad_count = 0
    for i in range(N):
        value_list = labels[i]
        pos_count = sum(value_list == 1)
        neg_count = sum(value_list == -1)
        missing_count = sum(value_list == 0)
        p[pos_count] += 1
        n[neg_count] += 1
        m[missing_count] += 1
        if pos_count + neg_count == 0:
            bad_count += 1
    print('bad_count\t', bad_count)
    
    print('pos\t', p)
    print('neg\t', n)
    print('missing\t', m)
    return


def _load_pcba_dataset(input_path):
    input_df = pd.read_csv(input_path, sep=',')
    tasks = list(input_df.columns)[:-2]

    N = input_df.shape[0]
    temp_df = input_df[tasks]
    temp_df = temp_df.replace(0, -1)
    temp_df = temp_df.fillna(0)

    bad_tasks = check_columns(temp_df, tasks, N)
    for task in bad_tasks:
        tasks.remove(task)
    print('good tasks\t', len(tasks))

    labels = input_df[tasks]
    labels = labels.replace(0, -1)
    labels = labels.fillna(0)
    labels = labels.values
    print(labels.shape)  # 439863, 92
    check_rows(labels, N)

    input_df.dropna(subset=tasks, how='all', inplace=True)
    # convert 0 to -1
    input_df = input_df.replace(0, -1)
    # convert nan to 0
    input_df = input_df.fillna(0)
    labels = input_df[tasks].values
    print(input_df.shape)  # 435685, 92
    N = input_df.shape[0]
    check_rows(labels, N)

    smiles_list = input_df['smiles'].tolist()
    rdkit_mol_objs_list = [AllChem.MolFromSmiles(s) for s in smiles_list]

    assert len(smiles_list) == len(rdkit_mol_objs_list)
    assert len(smiles_list) == len(labels)
    return smiles_list, rdkit_mol_objs_list, labels


def _load_sider_dataset(input_path):

    input_df = pd.read_csv(input_path, sep=',')
    smiles_list = input_df['smiles']
    rdkit_mol_objs_list = [AllChem.MolFromSmiles(s) for s in smiles_list]
    tasks = ['Hepatobiliary disorders',
             'Metabolism and nutrition disorders', 'Product issues', 'Eye disorders',
             'Investigations', 'Musculoskeletal and connective tissue disorders',
             'Gastrointestinal disorders', 'Social circumstances',
             'Immune system disorders', 'Reproductive system and breast disorders',
             'Neoplasms benign, malignant and unspecified (incl cysts and polyps)',
             'General disorders and administration site conditions',
             'Endocrine disorders', 'Surgical and medical procedures',
             'Vascular disorders', 'Blood and lymphatic system disorders',
             'Skin and subcutaneous tissue disorders',
             'Congenital, familial and genetic disorders',
             'Infections and infestations',
             'Respiratory, thoracic and mediastinal disorders',
             'Psychiatric disorders', 'Renal and urinary disorders',
             'Pregnancy, puerperium and perinatal conditions',
             'Ear and labyrinth disorders', 'Cardiac disorders',
             'Nervous system disorders',
             'Injury, poisoning and procedural complications']
    labels = input_df[tasks]
    # convert 0 to -1
    labels = labels.replace(0, -1)
    assert len(smiles_list) == len(rdkit_mol_objs_list)
    assert len(smiles_list) == len(labels)
    return smiles_list, rdkit_mol_objs_list, labels.values


def _load_toxcast_dataset(input_path):

    # NB: some examples have multiple species, some example smiles are invalid
    input_df = pd.read_csv(input_path, sep=',')
    smiles_list = input_df['smiles']
    rdkit_mol_objs_list = [AllChem.MolFromSmiles(s) for s in smiles_list]
    # Some smiles could not be successfully converted
    # to rdkit mol object so them to None
    preprocessed_rdkit_mol_objs_list = [m if m is not None else None
                                        for m in rdkit_mol_objs_list]
    preprocessed_smiles_list = [AllChem.MolToSmiles(m) if m is not None else None
                                for m in preprocessed_rdkit_mol_objs_list]
    tasks = list(input_df.columns)[1:]
    labels = input_df[tasks]
    # convert 0 to -1
    labels = labels.replace(0, -1)
    # convert nan to 0
    labels = labels.fillna(0)
    assert len(smiles_list) == len(preprocessed_rdkit_mol_objs_list)
    assert len(smiles_list) == len(preprocessed_smiles_list)
    assert len(smiles_list) == len(labels)
    return smiles_list, \
           preprocessed_rdkit_mol_objs_list, labels.values


def _load_chembl_with_labels_dataset(root_path):
    # 1. load folds and labels
    f = open(os.path.join(root_path, 'folds0.pckl'), 'rb')
    folds = pickle.load(f)
    f.close()

    f = open(os.path.join(root_path, 'labelsHard.pckl'), 'rb')
    targetMat = pickle.load(f)
    sampleAnnInd = pickle.load(f)
    targetAnnInd = pickle.load(f)
    f.close()

    targetMat = targetMat
    targetMat = targetMat.copy().tocsr()
    targetMat.sort_indices()
    targetAnnInd = targetAnnInd
    targetAnnInd = targetAnnInd - targetAnnInd.min()

    folds = [np.intersect1d(fold, sampleAnnInd.index.values).tolist() for fold in folds]
    targetMatTransposed = targetMat[sampleAnnInd[list(chain(*folds))]].T.tocsr()
    targetMatTransposed.sort_indices()
    # # num positive examples in each of the 1310 targets
    trainPosOverall = np.array([np.sum(targetMatTransposed[x].data > 0.5)
                                for x in range(targetMatTransposed.shape[0])])
    # # num negative examples in each of the 1310 targets
    trainNegOverall = np.array([np.sum(targetMatTransposed[x].data < -0.5)
                                for x in range(targetMatTransposed.shape[0])])
    # dense array containing the labels for the 456331 molecules and 1310 targets
    denseOutputData = targetMat.A  # possible values are {-1, 0, 1}

    # 2. load structures
    f = open(os.path.join(root_path, 'chembl20LSTM.pckl'), 'rb')
    rdkitArr = pickle.load(f)
    f.close()

    assert len(rdkitArr) == denseOutputData.shape[0]
    assert len(rdkitArr) == len(folds[0]) + len(folds[1]) + len(folds[2])

    preprocessed_rdkitArr = []
    print('preprocessing')
    for i in range(len(rdkitArr)):
        print(i)
        m = rdkitArr[i]
        if m is None:
            preprocessed_rdkitArr.append(None)
        else:
            mol_species_list = split_rdkit_mol_obj(m)
            if len(mol_species_list) == 0:
                preprocessed_rdkitArr.append(None)
            else:
                largest_mol = get_largest_mol(mol_species_list)
                if len(largest_mol.GetAtoms()) <= 2:
                    preprocessed_rdkitArr.append(None)
                else:
                    preprocessed_rdkitArr.append(largest_mol)

    assert len(preprocessed_rdkitArr) == denseOutputData.shape[0]

    smiles_list = [AllChem.MolToSmiles(m) if m is not None else None
                   for m in preprocessed_rdkitArr]
    # bc some empty mol in the rdkitArr zzz...

    assert len(preprocessed_rdkitArr) == len(smiles_list)

    return smiles_list, preprocessed_rdkitArr, folds, denseOutputData


# root_path = 'dataset/chembl_with_labels'
def check_smiles_validity(smiles):
    try:
        m = Chem.MolFromSmiles(smiles)
        if m:
            return True
        else:
            return False
    except:
        return False


def split_rdkit_mol_obj(mol):
    """
    Split rdkit mol object containing multiple species or one species into a
    list of mol objects or a list containing a single object respectively """

    smiles = AllChem.MolToSmiles(mol, isomericSmiles=True)
    smiles_list = smiles.split('.')
    mol_species_list = []
    for s in smiles_list:
        if check_smiles_validity(s):
            mol_species_list.append(AllChem.MolFromSmiles(s))
    return mol_species_list

def get_largest_mol(mol_list):
    """
    Given a list of rdkit mol objects, returns mol object containing the
    largest num of atoms. If multiple containing largest num of atoms,
    picks the first one """

    num_atoms_list = [len(m.GetAtoms()) for m in mol_list]
    largest_mol_idx = num_atoms_list.index(max(num_atoms_list))
    return mol_list[largest_mol_idx]

def _load_tox21_dataset(input_path):
    input_df = pd.read_csv(input_path, sep=',')
    smiles_list = input_df['smiles']
    rdkit_mol_objs_list = [AllChem.MolFromSmiles(s) for s in smiles_list]
    tasks = ['NR-AR', 'NR-AR-LBD', 'NR-AhR', 'NR-Aromatase', 'NR-ER', 'NR-ER-LBD',
             'NR-PPAR-gamma', 'SR-ARE', 'SR-ATAD5', 'SR-HSE', 'SR-MMP', 'SR-p53']
    labels = input_df[tasks]
    # convert 0 to -1
    labels = labels.replace(0, -1)
    # convert nan to 0
    labels = labels.fillna(0)
    assert len(smiles_list) == len(rdkit_mol_objs_list)
    assert len(smiles_list) == len(labels)
    return smiles_list, rdkit_mol_objs_list, labels.values


def _load_hiv_dataset(input_path):
    input_df = pd.read_csv(input_path, sep=',')
    smiles_list = input_df['smiles']
    rdkit_mol_objs_list = [AllChem.MolFromSmiles(s) for s in smiles_list]
    labels = input_df['HIV_active']
    # convert 0 to -1
    labels = labels.replace(0, -1)
    # there are no nans
    assert len(smiles_list) == len(rdkit_mol_objs_list)
    assert len(smiles_list) == len(labels)
    return smiles_list, rdkit_mol_objs_list, labels.values

def _load_bace_dataset(input_path):
    input_df = pd.read_csv(input_path, sep=',')
    smiles_list = input_df['mol']
    rdkit_mol_objs_list = [AllChem.MolFromSmiles(s) for s in smiles_list]
    labels = input_df['Class']
    # convert 0 to -1
    labels = labels.replace(0, -1)
    # there are no nans
    folds = input_df['Model']
    folds = folds.replace('Train', 0)  # 0 -> train
    folds = folds.replace('Valid', 1)  # 1 -> valid
    folds = folds.replace('Test', 2)  # 2 -> test
    assert len(smiles_list) == len(rdkit_mol_objs_list)
    assert len(smiles_list) == len(labels)
    assert len(smiles_list) == len(folds)
    # return smiles_list, rdkit_mol_objs_list, folds.values, labels.values
    return smiles_list, rdkit_mol_objs_list, labels.values