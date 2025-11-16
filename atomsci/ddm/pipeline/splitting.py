"""Encapsulates everything that depends on how datasets are split: the splitting itself, training, validation,
testing, generation of predicted values and performance metrics.
"""

import time
import os
import shutil
import warnings
import logging
import copy
import deepchem as dc
import numpy as np
import pandas as pd
from deepchem.data import NumpyDataset, DiskDataset
from atomsci.ddm.pipeline.ave_splitter import AVEMinSplitter
from atomsci.ddm.pipeline.temporal_splitter import TemporalSplitter
from atomsci.ddm.pipeline.MultitaskScaffoldSplit import MultitaskScaffoldSplitter
from atomsci.ddm.utils.many_to_one import many_to_one_df
from atomsci.ddm.pipeline.utils import get_memory_usage
import collections
try:
    import mlflow
    MLFLOW_LOADED = True
except:
    MLFLOW_LOADED = False

logging.basicConfig(format='%(asctime)-15s %(message)s')
log = logging.getLogger('ATOM')

# List of splitter types that require SMILES strings as compound IDs
smiles_splits = ['scaffold', 'multitaskscaffold', 'butina', 'fingerprint']

# List of model parameters related to splitting. Make sure to update this list if we add more parameters.
split_params = ['splitter', 'split_strategy', 'split_valid_frac', 'split_test_frac', 'butina_cutoff',
                'num_folds', 'base_splitter', 'cutoff_date', 'date_col', 'previously_split',
                'mtss_num_super_scaffolds', 'mtss_num_generations', 'mtss_train_test_dist_weight', 
                'mtss_train_valid_dist_weight', 'mtss_split_fraction_weight', 'mtss_num_pop',
                'mtss_response_distr_weight']

def create_splitting(params, random_state=None, seed=None):
    """Factory function to create appropriate type of Splitting object, based on dataset parameters
    
    Args:
        params (Namespace object): contains all parameter information.
    
    Returns:
        (Splitting object): Splitting subtype (TrainValidTestSplitting or KFoldSplitting)
        determined by params.split_strategy
        
    Raises:
        Exception: If params.split_strategy not in ['train_valid_test','k_fold_cv']. Unsupported split strategy
        
    """
    if MLFLOW_LOADED:
        mlflow.log_params({
            "split_strategy": params.split_strategy,
            "production": params.production,
        })

    if params.production:
        log.debug("Creating ProductionSplitting object...")
        return ProductionSplitting(params, random_state=random_state, seed=seed)
    elif params.split_strategy == 'train_valid_test':
        log.debug("Creating TrainValidTestSplitting object...")
        return TrainValidTestSplitting(params, random_state=random_state, seed=seed)
    elif params.split_strategy == 'simple_train_valid_test':
        log.debug("Creating SimpleTrainValidTestSplitting object...")
        return SimpleTrainValidTestSplitting(params, random_state=random_state, seed=seed)
    elif params.split_strategy == 'k_fold_cv':
        log.debug("Creating KFoldSplitting object...")
        return KFoldSplitting(params, random_state=random_state, seed=seed)
    else:
        raise Exception("Unknown split strategy %s" % params.split_strategy)

# ****************************************************************************************
def select_dset_by_attr_ids(dataset, attr_df):
    """Returns a subset of the given dc.data.Dataset object selected by matching compound IDs in the index of attr_df
    against the ids in the dataset.

    Args:
        dataset (Dataset): The deepchem dataset, should have matching ids with ids in attr_df

        attr_df (DataFrame): Contains the compound ids to subset the dataset. Ids should match with dataset

    Returns:
        subset (Dataset): A subset of the deepchem dataset as determined by the ids in attr_df

    """
    id_df = pd.DataFrame({'indices' : np.arange(len(dataset.ids), dtype=np.int32)}, index=[str(e) for e in dataset.ids])
    match_df = id_df.join(attr_df, how='inner')
    subset = dataset.select(match_df.indices.values)
    return subset

# ****************************************************************************************
def select_dset_by_id_list(dataset, id_list):
    """Returns a subset of the given dc.data.Dataset object selected by matching compound IDs in the given list
    against the ids in the dataset.

    Args:
        dataset (Dataset): The deepchem dataset, should have matching ids with ids in id_list

        id_list (list): Contains a list of compound ids to subset the dataset. Ids should match with dataset

    Returns:
        subset (Dataset): A subset of the deepchem dataset as determined by the ids in id_list

    """
    #TODO: Need to test
    id_df = pd.DataFrame({'indices' : np.arange(len(dataset.ids), dtype=np.int32)}, 
        index=[str(e) for e in dataset.ids])
    match_df = id_df.loc[id_df.index.isin(id_list)]
    subset = dataset.select(match_df.indices.values)
    return subset

# ****************************************************************************************
def select_attrs_by_dset_ids(dataset, attr_df):
    """Returns a subset of the data frame attr_df selected by matching compound IDs in the index of attr_df
    against the ids in the dc.data.Dataset object dataset.

    Args:
        dataset (Dataset): The deepchem dataset, should have matching ids with ids in attr_df

        attr_df (DataFrame): Contains the compound ids. Ids should match with dataset

    Returns:
        subattr_df (DataFrame): A subset of attr_df as determined by the ids in dataset

    """
    #TODO: Need to test
    id_df = pd.DataFrame(index=[str(e) for e in set(dataset.ids)])
    subattr_df = id_df.join(attr_df, how='inner')
    return subattr_df

# ****************************************************************************************
def select_attrs_by_dset_smiles(dataset, attr_df, smiles_col):
    """Returns a subset of the data frame attr_df selected by matching SMILES strings in attr_df
    against the ids in the dc.data.Dataset object dataset.

    Args:
        dataset (Dataset): The deepchem dataset, should have matching ids with ids in attr_df

        attr_df (DataFrame): Contains the compound ids. Ids should match with dataset. Should contain a column
        of SMILES strings under the smiles_col

        smiles_col (str): Name of the column containing smiles strings

    Returns:
        subattr_df (DataFrame): A subset of attr_df as determined by the ids in dataset. Selected by matching SMILES
        strings in attr_df to the ids in the dataset

    """
    id_df = pd.DataFrame(index=[str(e) for e in dataset.ids])
    subattr_df = id_df.merge(attr_df.drop_duplicates(subset=smiles_col), how='inner', left_index=True, right_on=smiles_col)
    return subattr_df

# ****************************************************************************************
def check_if_dupe_smiles_dataset(dataset,attr_df, smiles_col):
    """Returns a boolean. True if there are duplication within the deepchem Dataset dataset.ids or if there are
        duplicates in the smiles_col of the attr_df

    Args:
       dataset (deepchem Dataset): full featurized dataset

       attr_df (Pandas DataFrame): dataframe containing SMILES strings indexed by compound IDs,

       smiles_col (string): name of SMILES column (hack for now until deepchem fixes scaffold and butina splitters)

    Returns:
       (bool): True if there duplicates in the ids of the dataset or if there are duplicates in the smiles_col
       of the attr_df. False if there are no duplicates

    """
    dupes = [(item, count) for item, count in collections.Counter([str(e) for e in dataset.ids]).items() if count > 1]
    dupe_attr = [(item, count) for item, count in collections.Counter(attr_df[smiles_col].values.tolist()).items() if count > 1]

    if not len(dupes)==0 or not len(dupe_attr)==0:
        return True
    else:
        return False
# ****************************************************************************************

class Splitting(object):
    """Base class for train/validation/test and k-fold dataset splitting. Wrapper for DeepChem Splitter
    classes that handle the specific splitting methods (e.g. random, scaffold, etc.).

    Attributes:
        Set in __init__:
            params (Namespace object): contains all parameter information

            split (str): Type of splitter in ['index','random','scaffold','butina','ave_min','stratified']

            splitter (Deepchem split object): A splitting object of the subtype specified by split

    """

    def __init__(self, params, random_state=None, seed=None):
        """Constructor, also serves as a factory method for creating the associated DeepChem splitter object

        Args:
            params (Namespace object): contains all parameter information.

        Raises:
            Exception: if splitter is not in ['index','random','scaffold','butina','ave_min','stratified'],
            it is not supported by this class

        Side effects:

            Sets the following Splitting object attributes

                params (Namespace object): contains all parameter information

                split (str): Type of splitter in ['index','random','scaffold','butina','ave_min','stratified']

                splitter (Deepchem split object): A splitting object of the subtype specified by split

        """
        self.random_state = random_state
        self.seed = seed

        self.params = params
        self.split = params.splitter
        log.debug(f"Split initialiser: params.splitter = {params.splitter}")
        if params.production:
            self.splitter = ProductionSplitter()
        elif params.splitter == 'index':
            self.splitter = dc.splits.IndexSplitter()
        elif params.splitter == 'random':
            self.splitter = dc.splits.RandomSplitter()
        elif params.splitter == 'scaffold':
            self.splitter = dc.splits.ScaffoldSplitter()
        elif params.splitter == 'multitaskscaffold':
            self.splitter = MultitaskScaffoldSplitter()
        elif params.splitter == 'stratified':
            self.splitter = dc.splits.RandomStratifiedSplitter()
        elif params.splitter == 'butina':
            self.splitter = dc.splits.ButinaSplitter(cutoff=params.butina_cutoff)
        elif params.splitter == 'fingerprint':
            self.splitter = dc.splits.FingerprintSplitter()
        elif params.splitter == 'ave_min':
            if params.featurizer == 'ecfp':
                self.splitter = AVEMinSplitter(metric='jaccard')
            else:
                self.splitter = AVEMinSplitter(metric='euclidean')
        elif params.splitter == 'temporal':
            if params.base_splitter == 'ave_min':
                if params.featurizer == 'ecfp':
                    metric = 'jaccard'
                else:
                    metric = 'euclidean'
            else:
                metric = None
            if params.base_splitter in smiles_splits:
                _id_col = params.smiles_col
            else:
                _id_col = params.id_col
            self.splitter = TemporalSplitter(cutoff_date=params.cutoff_date,
                    date_col=params.date_col,
                    base_splitter=params.base_splitter, metric=metric)
        else:
            raise Exception("Unknown splitting method %s" % params.splitter)

    # ****************************************************************************************
    def get_split_prefix(self, parent=''):
        """Must be implemented by subclasses

        Raises:
            NotImplementedError: The method is implemented by subclasses

        """
        raise NotImplementedError

    # ****************************************************************************************
    def split_dataset(self, dataset, attr_df, smiles_col):
        """Must be implemented by subclasses

        Raises:
            NotImplementedError: The method is implemented by subclasses

        """
        raise NotImplementedError

    # ****************************************************************************************
    def needs_smiles(self):
        """Returns True if underlying DeepChem splitter requires compound IDs to be SMILES strings

        Returns:
            (bool): True if Deepchem splitter requires SMILES strings as compound IDs, currently only true if
            using scaffold or butina splits

        """
        return (self.split in smiles_splits) or ((self.split == 'temporal') and (self.params.base_splitter in smiles_splits))


# ****************************************************************************************
class KFoldSplitting(Splitting):
    """Subclass to deal with everything related to k-fold cross-validation splits

    Attributes:
        Set in __init__:
            params (Namespace object): contains all parameter information

            split (str): Type of splitter in ['index','random','scaffold','butina','ave_min','stratified']

            splitter (Deepchem split object): A splitting object of the subtype specified by split

            num_folds (int): The number of k-fold splits to perform

    """

    def __init__(self, params, random_state=None, seed=None):
        """Initialization method for KFoldSplitting.

                Sets the following attributes for KFoldSplitting:
           params (Namespace object): contains all parameter information

           split (str): Type of splitter in ['index','random','scaffold','butina','ave_min','stratified']

           splitter (Deepchem split object): A splitting object of the subtype specified by split

           num_folds (int): The number of k-fold splits to perform

        """
        super().__init__(params, random_state, seed)
        self.num_folds = params.num_folds


    # ****************************************************************************************

    def get_split_prefix(self, parent=''):
        """Returns a string identifying the split strategy (TVT or k-fold) and the splitting method (index, scaffold,
        etc.) for use in filenames, dataset keys, etc.

        Args:
            parent (str): Default to empty string. Sets the parent directory for the output string

        Returns:
            (str): A string that identifies the split strategy and the splitting method. Appends a parent directory in
            front of the fold description

        """
        if parent != '':
            parent = "%s/" % parent
        return "%s%d_fold_cv_%s" % (parent, self.num_folds, self.split)

    # ****************************************************************************************
    def split_dataset(self, dataset, attr_df, smiles_col):
        #smiles_col is a hack for now until deepchem fixes their scaffold and butina splitters
        """Splits dataset into training, testing and validation sets.

        Args:
            dataset (deepchem Dataset): full featurized dataset

            attr_df (Pandas DataFrame): dataframe containing SMILES strings indexed by compound IDs,

            smiles_col (string): name of SMILES column (hack for now until deepchem fixes scaffold and butina splitters)

        Returns:
            [(train, valid)], test, [(train_attr, valid_attr)], test_attr:

            train (deepchem Dataset): training dataset.

            valid (deepchem Dataset): validation dataset.

            test (deepchem Dataset): testing dataset.

            train_attr (Pandas DataFrame): dataframe of SMILES strings indexed by compound IDs for training set.

            valid_attr (Pandas DataFrame): dataframe of SMILES strings indexed by compound IDs for validation set.

            test_attr (Pandas DataFrame): dataframe of SMILES strings indexed by compound IDs for test set.

        Raises:
            Exception if there are duplicate ids or smiles strings in the dataset or the attr_df

        """

        # Duplicate SMILES and compound_ids are merged into single compounds
        # in DatasetManager. The first instance of each is kept. Assumes many to one 
        # mapping of compound_ids and SMILES. dataset.ids is either compound_id or
        # SMILES depending on the call to self.needs_smiles(). Later expand_selection
        # will expect SMILES or compound_ids in dataset.ids depending on needs_smiles
        # passed into the constructor
        dm = DatasetManager(dataset=dataset, attr_df=attr_df, smiles_col=smiles_col,
            needs_smiles=self.needs_smiles())
        dataset = dm.compact_dataset()

        # Under k-fold CV, the training/validation splits are determined by num_folds; only the test set fraction
        # is directly specified through command line parameters. If we use Butina splitting, we can't control
        # the test set size either.
        train_frac = 1.0 - self.params.split_test_frac

        # Use DeepChem train_test_split() to select held-out test set; then use k_fold_split on the
        # training set to split it into training/validation folds.
        if self.split == 'butina':
            train_cv, test, _ = self.splitter.train_valid_test_split(dataset, seed=self.seed)
            self.splitter = dc.splits.ScaffoldSplitter()
            train_cv_pairs = self.splitter.k_fold_split(train_cv, self.num_folds, seed=self.seed)
        else:
            # TODO: Add special handling for AVE splitter
            train_cv, test = self.splitter.train_test_split(dataset, frac_train=train_frac, seed=self.seed)
            train_cv_pairs = self.splitter.k_fold_split(train_cv, self.num_folds, seed=self.seed)

        train_valid_dsets = []
        train_valid_attr = []

        for train, valid in train_cv_pairs:
            exp_train, exp_train_attr = dm.expand_selection(train.ids)
            exp_valid, exp_valid_attr = dm.expand_selection(valid.ids)

            train_valid_dsets.append((exp_train, exp_valid))
            train_valid_attr.append((exp_train_attr, exp_valid_attr))
        
        test, test_attr = dm.expand_selection(test.ids)

        return train_valid_dsets, test, train_valid_attr, test_attr

# ****************************************************************************************

class TrainValidTestSplitting(Splitting):
    """Subclass to deal with everything related to standard train/validation/test splits

    Attributes:
        Set in __init__:
            params (Namespace object): contains all parameter information

            split (str): Type of splitter in ['index','random','scaffold','butina','ave_min','temporal','stratified']

            splitter (Deepchem split object): A splitting object of the subtype specified by split

            num_folds (int): The number of k-fold splits to perform

    """

    def __init__(self, params, random_state=None, seed=None):
        """Initialization method for TrainValidTestSplitting.

                Sets the following attributes for TrainValidTestSplitting:
           params (Namespace object): contains all parameter information

           split (str): Type of splitter in ['index','random','scaffold','butina','ave_min','temporal','stratified']

           splitter (Deepchem split object): A splitting object of the subtype specified by split

           num_folds (int): The number of k-fold splits to perform. In this case, it is always set to 1

        """
        super().__init__(params, random_state=random_state, seed=seed)
        self.num_folds = 1

    # ****************************************************************************************
    def get_split_prefix(self, parent=''):
        """Returns a string identifying the split strategy (TVT or k-fold) and the splitting method
        (index, scaffold, etc.) for use in filenames, dataset keys, etc.

        Args:
            parent (str): Default to empty string. Sets the parent directory for the output string

        Returns:
            (str): A string that identifies the split strategy and the splitting method.
            Appends a parent directory in front of the fold description

        """
        if parent != '':
            parent = "%s/" % parent
        return "%strain_valid_test_%s" % (parent, self.split)

    # ****************************************************************************************

    def split_dataset(self, dataset, attr_df, smiles_col):
        #smiles_col is a hack for now until deepchem fixes their scaffold and butina splitters
        """Splits dataset into training, testing and validation sets.

        For ave_min, random, scaffold, index splits
            self.params.split_valid_frac & self.params.split_test_frac should be defined and
            train_frac = 1.0 - self.params.split_valid_frac - self.params.split_test_frac

        For butina split, test size is not user defined, and depends on available clusters that qualify for placement in the test set
            train_frac = 1.0 - self.params.split_valid_frac

        For temporal split, test size is also not user defined, and depends on number of compounds with dates after cutoff date.
            train_frac = 1.0 - self.params.split_valid_frac
        Args:
            dataset (deepchem Dataset): full featurized dataset

            attr_df (Pandas DataFrame): dataframe containing SMILES strings indexed by compound IDs,

            smiles_col (string): name of SMILES column (hack for now until deepchem fixes scaffold and butina splitters)

        Returns:
            [(train, valid)], test, [(train_attr, valid_attr)], test_attr:
            train (deepchem Dataset): training dataset.

            valid (deepchem Dataset): validation dataset.

            test (deepchem Dataset): testing dataset.

            train_attr (Pandas DataFrame): dataframe of SMILES strings indexed by compound IDs for training set.

            valid_attr (Pandas DataFrame): dataframe of SMILES strings indexed by compound IDs for validation set.

            test_attr (Pandas DataFrame): dataframe of SMILES strings indexed by compound IDs for test set.

        Raises:
            Exception if there are duplicate ids or smiles strings in the dataset or the attr_df

        """
        log.info("Splitting data by %s" % self.params.splitter)
        log.debug(f"Needs smiles: {self.needs_smiles()}")
        log.debug(f"Memory usage before creating DatasetManager: {get_memory_usage():.3f} GiB")
        if self.params.use_disk_dataset:
            orig_dataset_dir = dataset.data_dir

        # Duplicate SMILES and compound_ids are merged into single compounds
        # in DatasetManager. The first instance of each is kept. Assumes many to one 
        # mapping of compound_ids and SMILES. dataset.ids is either compound_id or
        # SMILES depending on the call to self.needs_smiles(). Later expand_selection
        # will expect SMILES or compound_ids in dataset.ids depending on needs_smiles
        # passed into the constructor
        tick = time.time()
        dm = DatasetManager(dataset=dataset, attr_df=attr_df, smiles_col=smiles_col,
            needs_smiles=self.needs_smiles())
        log.debug(f"Time to create DatasetManager: {time.time() - tick} s")
        log.debug(f"Memory usage after creating DatasetManager: {get_memory_usage():.3f} GiB")
        tick = time.time()
        dataset = dm.compact_dataset()
        log.debug(f"Time to compact dataset: {time.time() - tick} s")
        log.debug(f"Memory usage after calling dm.compact_dataset: {get_memory_usage():.3f} GiB")

        if self.split == 'butina':
            # Can't use train_test_split with Butina because Butina splits into train and valid sets only.
            train_valid, test, _ = self.splitter.train_valid_test_split(dataset, seed=self.seed)
            self.splitter = dc.splits.ScaffoldSplitter()
            # With Butina splitting, we don't have control over the size of the test set
            train_frac = 1.0 - self.params.split_valid_frac
            train, valid = self.splitter.train_test_split(train_valid, frac_train=train_frac, seed=self.seed)
        elif self.split == 'ave_min':
            # AVEMinSplitter also only does train-valid splits, but at least nested splits seem to work.
            # TODO: Change this if we modify AVE splitter to do 3-way splits internally.
            train_valid_frac = 1.0 - self.params.split_test_frac
            train_frac = train_valid_frac - self.params.split_valid_frac
            log.info("Performing split for test set")
            train_valid, test, _ = self.splitter.train_valid_test_split(dataset, frac_train=train_valid_frac, 
                                                                        frac_valid=self.params.split_test_frac,
                                                                        frac_test=0.0, seed=self.seed)
            log.info("Performing split of training and validation sets")
            train, valid, _ = self.splitter.train_valid_test_split(train_valid, frac_train=train_frac/train_valid_frac, 
                                                                   frac_valid=self.params.split_valid_frac/train_valid_frac,
                                                                   frac_test=0.0, seed=self.seed)
            log.info("Results of 3-way split: %d training, %d validation, %d test compounds" % (
                     train.X.shape[0], valid.X.shape[0], test.X.shape[0]))
        elif self.split == 'temporal':
            # TemporalSplitter requires that we pass attr_df so it can get the dates for each compound
            train_frac = 1.0 - self.params.split_valid_frac
            train, valid, test = self.splitter.train_valid_test_split(dataset, attr_df=attr_df,
                                frac_train=train_frac, frac_valid=self.params.split_valid_frac)
        elif self.split == 'multitaskscaffold':
            # perform multitask scaffold split
            train_frac = 1.0 - self.params.split_valid_frac - self.params.split_test_frac
            train, valid, test = self.splitter.train_valid_test_split(
                dataset, 
                frac_train=train_frac, 
                frac_valid=self.params.split_valid_frac, 
                frac_test=self.params.split_test_frac, 
                diff_fitness_weight_tvt=self.params.mtss_train_test_dist_weight,
                diff_fitness_weight_tvv=self.params.mtss_train_valid_dist_weight,
                ratio_fitness_weight=self.params.mtss_split_fraction_weight,
                response_distr_fitness_weight=self.params.mtss_response_distr_weight,
                num_super_scaffolds=self.params.mtss_num_super_scaffolds,
                num_pop=self.params.mtss_num_pop,
                num_generations=self.params.mtss_num_generations,
                seed=self.seed)
        else:
            log.debug("Doing train_valid_test_split...")
            log.debug(f"Memory usage before calling splitter.train_valid_test_split: {get_memory_usage():.3f} GiB")
            train_frac = 1.0 - self.params.split_valid_frac - self.params.split_test_frac
            tick = time.time()
            train, valid, test = self.splitter.train_valid_test_split(dataset, 
                frac_train=train_frac, frac_valid=self.params.split_valid_frac, frac_test=self.params.split_test_frac, seed=self.seed)
            log.debug(f"Time to call splitter.train_valid_test_split: {time.time() - tick} s")
            log.debug(f"Memory usage after calling splitter.train_valid_test_split: {get_memory_usage():.3f} GiB")

            log.debug(f"train {train}; valid {valid}; test {test}")
            if isinstance(dataset, DiskDataset):
                log.debug(f"train dir: {train.data_dir}")
                log.debug(f"valid dir: {valid.data_dir}")
                log.debug(f"test dir: {test.data_dir}")

        # After splitting unique compound_ids or SMILES are expanded 
        tick = time.time()
        train, train_attr = dm.expand_selection(train.ids)
        valid, valid_attr = dm.expand_selection(valid.ids)
        test, test_attr = dm.expand_selection(test.ids)
        log.debug(f"Time to call expand selection: {time.time() - tick} s")
        log.debug(f"Memory usage after calling dm.expand_selection: {get_memory_usage():.3f} GiB")

        if isinstance(dataset, DiskDataset):
            train.move(orig_dataset_dir + "-train")
            valid.move(orig_dataset_dir + "-valid")
            test.move(orig_dataset_dir + "-test")
            log.debug(f"train dir after expand: {train.data_dir}")
            log.debug(f"valid dir after expand: {valid.data_dir}")
            log.debug(f"test dir after expand: {test.data_dir}")

        # Note grouping of train/valid return values as tuple lists, to match format of 
        # KFoldSplitting.split_dataset().
        return [(train, valid)], test, [(train_attr, valid_attr)], test_attr

# ****************************************************************************************

class SimpleTrainValidTestSplitting(TrainValidTestSplitting):
    def split_dataset(self, dataset, attr_df, smiles_col):
        """
        Very basic splitter to improve performance on very large disk datasets.

        Assumes the rows in the input CSV file that the dataset was created from were
        randomly shuffled, e.g. using the `shuf` command.

        Just takes the first train_fraction for train dataset, next valid_fraction for
        valid dataset, etc.

        """
        if not isinstance(dataset, DiskDataset):
            raise ValueError("Shouldn't be using SimpleTrainValidTestSplitting with NumpyDatset -> use TrainValidTestSplitting instead")
        log.debug("Start of simple splitter")
        log.debug(f"self.split = {self.split}")

        # TODO: may still need dataset manager and compact/expand???
        # TODO: may want to reshard if needed to ensure even split

        # create three new datasets
        train_frac = 1.0 - self.params.split_valid_frac - self.params.split_test_frac
        log.debug(f"Fractions: train {train_frac}; valid {self.params.split_valid_frac}; test {self.params.split_test_frac}")

        total_len = len(dataset)
        num_shards = dataset.get_number_shards()
        log.debug(f"Length of full dataset: {total_len}")
        log.debug(f"Num shards: {dataset.get_number_shards()}")
        log.debug(f"Shard size: {dataset.get_shard_size()}")

        test_num_shards = int(num_shards * self.params.split_test_frac)
        valid_num_shards = int(num_shards * self.params.split_valid_frac)
        train_num_shards = num_shards - test_num_shards - valid_num_shards
        log.debug(f"Num shards split: train {train_num_shards}; valid {valid_num_shards}; test {test_num_shards}")

        # create the train dataset
        log.debug(f"Creating train dataset...")
        tick = time.time()
        shard_start = 0
        shard_end = train_num_shards
        log.debug(f"Shards: {range(shard_start, shard_end)}")
        train = dataset.subset(
            range(shard_start, shard_end),
            subset_dir=dataset.data_dir + "-train",
        )
        attr_begin = 0
        attr_end = len(train)
        log.debug(f"attr range: {attr_begin}:{attr_end}")
        train_attr = attr_df.iloc[attr_begin:attr_end, :]
        log.debug(f"Created train dataset in {time.time() - tick} s")

        # create the valid dataset
        log.debug(f"Creating valid dataset...")
        tick = time.time()
        shard_start = shard_end
        shard_end = shard_start + valid_num_shards
        log.debug(f"Shards: {range(shard_start, shard_end)}")
        valid = dataset.subset(
            range(shard_start, shard_end),
            subset_dir=dataset.data_dir + "-valid",
        )
        attr_begin = attr_end
        attr_end = attr_begin + len(valid)
        log.debug(f"attr range: {attr_begin}:{attr_end}")
        valid_attr = attr_df.iloc[attr_begin:attr_end, :]
        log.debug(f"Created valid dataset in {time.time() - tick} s")

        # create the test dataset
        log.debug(f"Creating test dataset...")
        tick = time.time()
        shard_start = shard_end
        shard_end = shard_start + test_num_shards
        log.debug(f"Shards: {range(shard_start, shard_end)}")
        test = dataset.subset(
            range(shard_start, shard_end),
            subset_dir=dataset.data_dir + "-test",
        )
        attr_begin = attr_end
        attr_end = attr_begin + len(test)
        log.debug(f"attr range: {attr_begin}:{attr_end}")
        test_attr = attr_df.iloc[attr_begin:attr_end, :]
        log.debug(f"Created test dataset in {time.time() - tick} s")

        log.debug("End of simple splitter")

        return [(train, valid)], test, [(train_attr, valid_attr)], test_attr

# ****************************************************************************************

class ProductionSplitter(dc.splits.Splitter):
    def split(
            self, dataset, frac_train=1, frac_valid=1, frac_test=1, seed=None, log_every_n = None
    ):
        """We implement a production run as having a split that contains all samples in every subset"""
        num_datapoints = len(dataset)
        return (list(range(num_datapoints)), list(range(num_datapoints)), list(range(num_datapoints)))


# ****************************************************************************************

class ProductionSplitting(Splitting):
    def __init__(self, params, random_state=None, seed=None):
        """This Splitting only does one thing and ignores all splitter parameters"""
        super().__init__(params, random_state=random_state, seed=seed)
        self.split = 'production'
        self.num_folds = 1

    # ****************************************************************************************
    def get_split_prefix(self, parent=''):
        """Returns a string identifying the split strategy (production) and the splitting method
        (index, scaffold, etc.) for use in filenames, dataset keys, etc.

        Args:
            parent (str): Default to empty string. Sets the parent directory for the output string

        Returns:
            (str): A string that identifies the split strategy and the splitting method.
            Appends a parent directory in front of the fold description

        """
        if parent != '':
            parent = "%s/" % parent
        return "%sproduction_%s" % (parent, self.split)

    # ****************************************************************************************
    def split_dataset(self, dataset, attr_df, smiles_col):
        """Splits dataset into training, testing and validation sets.
        This should contain the entire dataset in each subset

        Args:
            dataset (deepchem Dataset): full featurized dataset

            attr_df (Pandas DataFrame): dataframe containing SMILES strings indexed by compound IDs,

            smiles_col (string): name of SMILES column (hack for now until deepchem fixes scaffold and butina splitters)

        Returns:
            [(train, valid)], test, [(train_attr, valid_attr)], test_attr:
            train (deepchem Dataset): training dataset.

            valid (deepchem Dataset): validation dataset.

            test (deepchem Dataset): testing dataset.

            train_attr (Pandas DataFrame): dataframe of SMILES strings indexed by compound IDs for training set.

            valid_attr (Pandas DataFrame): dataframe of SMILES strings indexed by compound IDs for validation set.

            test_attr (Pandas DataFrame): dataframe of SMILES strings indexed by compound IDs for test set.

        Raises:
            Exception if there are duplicate ids or smiles strings in the dataset or the attr_df

        """
        log.info("Splitting data by Production")

        # Duplicate SMILES and compound_ids are merged into single compounds
        # in DatasetManager. The first instance of each is kept. Assumes many to one 
        # mapping of compound_ids and SMILES. dataset.ids is either compound_id or
        # SMILES depending on the call to self.needs_smiles(). Later expand_selection
        # will expect SMILES or compound_ids in dataset.ids depending on needs_smiles
        # passed into the constructor
        dm = DatasetManager(dataset=dataset, attr_df=attr_df, smiles_col=smiles_col,
            needs_smiles=self.needs_smiles())
        dataset = dm.compact_dataset()
        train, valid, test = self.splitter.train_valid_test_split(dataset, seed=self.seed)
        
        # After splitting unique compound_ids or SMILES are expanded 
        train, train_attr = dm.expand_selection(train.ids)
        valid, valid_attr = dm.expand_selection(valid.ids)
        test, test_attr = dm.expand_selection(test.ids)

        return [(train, valid)], test, [(train_attr, valid_attr)], test_attr

def _copy_modify_NumpyDataset(dataset, **kwargs):
    """Create a copy of the DeepChem Dataset object `dataset` and then modify it based on the given keyword arguments.
    This is useful for updating attributes like dataset.w or dataset.id
    """
    if isinstance(dataset, DiskDataset):
        raise NotImplementedError("Need to implement a version of _copy_modify_NumpyDataset for DiskDataset")

    args = {'X':dataset.X,
        'y':dataset.y,
        'w':dataset.w,
        'ids':dataset.ids,
        'n_tasks':dataset.y.shape[1]
        }
    args.update(kwargs)
    return NumpyDataset(**args)

class DatasetManager:
    """Different splitters have different dataset requirements.
        - unique compound ids
        - smiles used as ids

    This object transforms datasets to satisfy these requirements then undoes them
    once the splitting is done.
    """
    def __init__(self, dataset, attr_df, smiles_col, needs_smiles):
        """Before splitting we often have to compact the dataset to remove duplicate compound_ids. After
        splitting we will have to expand that dataset again. We save self.dataset_ori so we have
        a copy of the original. We keep self.id_df to know how to map from a set of compound_ids
        or smiles to an expanded set of indicies.

        Args:
            dataset (deepchem Dataset): full featurized dataset

            attr_df (Pandas DataFrame): dataframe containing SMILES strings indexed by compound IDs,

            smiles_col (string): name of SMILES column (hack for now until deepchem fixes scaffold and butina splitters)
        """
        self._dataset_path = None
        if isinstance(dataset, DiskDataset):
            self._dataset_path = dataset.data_dir + "-copy"
            self.dataset_ori = dataset.copy(self._dataset_path)
            log.debug(f"DatasetManager: copied disk dataset to: {self.dataset_ori.data_dir}")
        else:
            self.dataset_ori = copy.deepcopy(dataset)
        log.debug(f"Memory usage after copying dataset in dm constructor: {get_memory_usage():.3f} GiB")
        self.attr_df = attr_df
        self.smiles_col = smiles_col
        self.needs_smiles = needs_smiles

        self.dataset_dup = False

        # sometimes the ids in dataset_ori is already a SMILES string.
        # since we assume that dataset_ori.ids are compound ids, we replace them with attr_df.index
        if self.needs_smiles:
            if isinstance(dataset, DiskDataset):
                warnings.warn("In DatasetManager: check dataset_ori with_smiles...")
                self.dataset_ori.ids[:] = self.attr_df.index[:]
            else:
                self.dataset_ori = _copy_modify_NumpyDataset(self.dataset_ori, ids=self.attr_df.index)

        log.debug(f"DatasetManager init... len(indices)={len(self.dataset_ori.ids)}; len(smiles)={len(self.attr_df[self.smiles_col].values)}")

        # self.id_df will be used to map compound_ids or smiles to a set of indices to be used
        # with self.dataset_ori to map back to an expanded dataset after splitting
        self.id_df = pd.DataFrame({
            "indices" : np.arange(len(self.dataset_ori.ids), dtype=np.int32), 
            "compound_id": [str(e) for e in self.dataset_ori.ids],
            "smiles": self.attr_df[self.smiles_col].values})
        print(f"self.id_df:\n{self.id_df.info()}")
        # add columns for weights
        ws = self.dataset_ori.w # get the weights
        print(f"ws = {ws}")
        self.w_cols = [f'w{c}' for c in range(ws.shape[1])]
        for i, col in enumerate(self.w_cols):
            self.id_df[col] = ws[:,i]
        log.debug(f"Memory usage after building id_df in dm constructor: {get_memory_usage():.3f} GiB")

        # check many to one assumption.
        many_to_one_df(self.id_df, id_col='compound_id', smiles_col='smiles')

    def __del__(self):
        if self._dataset_path is not None:
            if os.path.isdir(self._dataset_path):
                log.debug(f"DatasetManager cleaning up temporary disk dataset copy: {self._dataset_path}")
                shutil.rmtree(self._dataset_path)

    def compact_dataset(self):
        """Returns a dataset with no duplicate compounds ids and smiles strings in the
        id column if necessary.

        Builds a new dataset with no duplicates in ids (compounds or smiles). This assumes
        a many to one mapping between SMILES and compound ids
        """
        sub_dataset = self.dataset_ori
        sel_df = self.id_df
        if check_if_dupe_smiles_dataset(self.dataset_ori, self.attr_df, self.smiles_col):
            warnings.warn("Haven't converted compact_dataset")
            log.info("Duplicate ids or smiles in the dataset, will deduplicate first and assign all records per compound ID to same partition")
            self.dataset_dup = True

            # the problem with this is if compounds with the same compound_id have different SMILES strings
            # If one compound must represent several compounds, the weights row
            # must be updated to contain labels for all tasks e.g.
            # w = [[0, 1, 0], --> w = [[1, 1, 0]]
            #      [1, 0, 0],
            #      [1, 0, 0]]
            def w_agg_func(x):
                return np.clip(np.sum(x, axis=0), a_min=0, a_max=1)
            agg_dict = {col:w_agg_func for col in self.w_cols}
            agg_dict['indices'] = 'first'
            agg_dict['compound_id'] = 'first' # Either they're all the same or they're not used
            agg_dict['smiles'] = 'first' # they're all the same in a group

            if self.needs_smiles:
                sel_df = self.id_df.groupby('smiles', as_index=False).agg(agg_dict)
            else:
                sel_df = self.id_df.groupby('compound_id', as_index=False).agg(agg_dict)

            # sub_dataset no longer contains duplicate compounds
            sub_dataset = sub_dataset.select(sel_df.indices.values)

            # update weight values
            sub_dataset = _copy_modify_NumpyDataset(sub_dataset, w=sel_df[self.w_cols].values)

        if self.needs_smiles:
            # Some DeepChem splitters require compound IDs in dataset to be SMILES strings. Swap in the
            # SMILES strings now; we'll reverse this later.
            sub_dataset = _copy_modify_NumpyDataset(sub_dataset, ids=sel_df.smiles.values)

        return sub_dataset

    def expand_selection(self, ids):
        """Implementation note: Maybe this is better if I used select_attrs_by_dset_smiles
        and select_attrs_by_dset_ids respectively

        Args:
            ids (iterable): Iterable containing either compound_ids or smiles. These
                need to be mapped back to a set of indices using either compound_ids
                or smiles.

        Returns:
            A subset of self.dataset_ori and subset of self.attr_df
        """
        # are we using SMILES or compound_ids as the ID column
        id_col = 'smiles' if self.needs_smiles else 'compound_id'
        sel_df = self.id_df[self.id_df[id_col].isin(ids)]

        data_subset = self.dataset_ori.select(sel_df.indices.values)
        attr_subset = self.attr_df.iloc[sel_df.indices.values]

        return data_subset, attr_subset
