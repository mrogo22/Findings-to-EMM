from collections import namedtuple
from functools import total_ordering
import numpy as np
from pysubgroup.measures import AbstractInterestingnessMeasure, BoundedInterestingnessMeasure,GeneralizationAwareQF_stats
from .subgroup_description import EqualitySelector, get_cover_array_and_size
from .utils import BaseTarget, derive_effective_sample_size
import plspm.config as c
from plspm.plspm import Plspm
from plspm.scheme import Scheme
from plspm.mode import Mode
from plspm.scale import Scale
import pandas as pd

# -------------------------------
# PLS SEM model target
# -------------------------------
@total_ordering
class SEMTarget(BaseTarget):
    """
    Model class for EMM: PLS SEM model
    Takes plspm model config as a parameter
    Multiple quality functions defined. Recommended: SEMQFEntropyCIComplement
    """

    # Define the names for statistics we compute

    statistic_types = (
        "size_sg",
        "disjoint_ci_count",
        "disjoint_ci_paths",
        "size_ds",
        "paths_model",
        "sig_paths_sg",
        "sig_paths_ds",
        "pos_sig_paths_sg",
        "pos_sig_paths_ds",
        "neg_sig_paths_sg",
        "neg_sig_paths_ds",
        "changes_sig_sg",
        "new_sig_sg",
        "flip_sign_sg"
    )

    def __init__(self, config):
        # pass PLS SEM configuration as param
        self.config = config

    def __repr__(self):
        return "SEMTarget: " + str(self.config)

    def __eq__(self, other):
        return self.__dict__ == other.__dict__

    def __lt__(self, other):
        return str(self) < str(other)

    def get_attributes(self):
        return [self.config]

    
    def get_base_statistics(self, subgroup, data):
        cover_arr, size_sg = get_cover_array_and_size(subgroup, len(data), data)
        mask = cover_arr.representation if hasattr(cover_arr, "representation") else cover_arr
        mask = np.asarray(mask)
        sg_data = data.iloc[np.where(mask)[0]]
        sg_complement = data.iloc[np.where(~mask)[0]]
        size_ds = len(data)

        
        if size_sg >= 20:
            pls_sg = Plspm(sg_data, self.config, Scheme.PATH, 1000, 0.00000001, False)
            model_sg = pls_sg.inner_model()
            model_sg = model_sg.rename(columns={"p>|t|": "p", "std error": "std_e"})
            

            pls_ds = Plspm(sg_complement, self.config, Scheme.PATH, 1000, 0.00000001, False)
            model_ds = pls_ds.inner_model()
            model_ds = model_ds.rename(columns={"p>|t|": "p_ds", "estimate": "estimate_ds", "std error": "std_e_ds"})


            paths_model = len(model_ds)

            sig_paths_sg = (model_sg["p"] <= 0.05).sum()
            sig_paths_ds = (model_ds["p_ds"] <= 0.05).sum()

            pos_sig_paths_sg = ((model_sg["p"] <= 0.05) & (model_sg["estimate"] > 0)).sum()
            pos_sig_paths_ds = ((model_ds["p_ds"] <= 0.05) & (model_ds["estimate_ds"] > 0)).sum()

            neg_sig_paths_sg = ((model_sg["p"] <= 0.05) & (model_sg["estimate"] < 0)).sum()
            neg_sig_paths_ds = ((model_ds["p_ds"] <= 0.05) & (model_ds["estimate_ds"] < 0)).sum()
            

            joint_model = model_ds.copy()
            joint_model[["estimate", "p", "std_e"]] = model_sg[["estimate", "p", "std_e"]]
            joint_model["lower_ci"] = joint_model["estimate"] - 1.96 * joint_model["std_e"]
            joint_model["upper_ci"] = joint_model["estimate"] + 1.96 * joint_model["std_e"]
            joint_model["lower_ci_ds"] = joint_model["estimate_ds"] - 1.96 * joint_model["std_e_ds"]
            joint_model["upper_ci_ds"] = joint_model["estimate_ds"] + 1.96 * joint_model["std_e_ds"]
            changes_sig_sg = 0
            flip_sign_sg = 0
            new_sig_sg = 0
            disjoint_ci_count = 0
            disjoint_ci_paths = []

            for _, row in joint_model.iterrows():
                if ((row["p"] <= 0.05 and row["p_ds"] > 0.05) or
                    (row["p"] > 0.05 and row["p_ds"] <= 0.05)):
                    changes_sig_sg += 1

                if (row["p"] <= 0.05 and row["p_ds"] > 0.05):
                    new_sig_sg+=1

                if ((row["p"] <= 0.05 and row["p_ds"] <= 0.05) and
                    (np.sign(row["estimate"]) * np.sign(row["estimate_ds"]) == -1)):
                    flip_sign_sg += 1

                if (row["upper_ci"] < row["lower_ci_ds"] or row["upper_ci_ds"] < row["lower_ci"]):
                    disjoint_ci_count += 1
                    disjoint_ci_paths.append(row.name)

        return(size_sg, disjoint_ci_count, disjoint_ci_paths, size_ds, paths_model, sig_paths_sg, sig_paths_ds, pos_sig_paths_sg, 
               pos_sig_paths_ds, neg_sig_paths_sg, neg_sig_paths_ds, changes_sig_sg,new_sig_sg, flip_sign_sg)

    def calculate_statistics(self, subgroup, data, cached_statistics=None):
        stats = {}
        (size_sg, disjoint_ci_count,disjoint_ci_paths, size_ds, paths_model, sig_paths_sg, sig_paths_ds, pos_sig_paths_sg, 
               pos_sig_paths_ds, neg_sig_paths_sg, neg_sig_paths_ds, changes_sig_sg, new_sig_sg, flip_sign_sg) = self.get_base_statistics(subgroup, data)
        
        stats["size_sg"] = size_sg
        stats["disjoint_ci_count"] = disjoint_ci_count
        stats["disjoint_ci_paths"] = disjoint_ci_paths
        stats["size_ds"] = size_ds
        stats["paths_model"] = paths_model
        stats["sig_paths_sg"] = sig_paths_sg
        stats["sig_paths_ds"] = sig_paths_ds
        stats["pos_sig_paths_sg"] = pos_sig_paths_sg
        stats["pos_sig_paths_ds"] = pos_sig_paths_ds
        stats["neg_sig_paths_sg"] = neg_sig_paths_sg
        stats["neg_sig_paths_ds"] = neg_sig_paths_ds
        stats["changes_sig_sg"] = changes_sig_sg
        stats["new_sig_sg"] = new_sig_sg
        stats["flip_sign_sg"] = flip_sign_sg


        return stats
    
# -------------------------------
# PLS SEM QF
# -------------------------------

class SEMQF(AbstractInterestingnessMeasure):
    """
    Rewards paths with flipped significance. In case of significant both in original and in sg model, rewards sign flip. Penalizes too small subgroups
    Requires minimum desired sg size (under which the QF penalizes)
    Only sg with >=20 elements are considered due to PLS SEM restrictions.
    """
    tpl = namedtuple("SEMQF_tpl", ["changes_sig_sg", "flip_sign_sg", "quality", "size_sg"])

    def __init__(self, config, weight_sig=1, weight_sign=1, min_sg_size=30):
        self.config = config
        self.weight_sig = weight_sig
        self.weight_sign = weight_sign
        self.min_size_sg = min_sg_size
        self.has_constant_statistics = False
        self.required_stat_attrs = self.tpl._fields
        self.dataset_statistics = None

    def calculate_constant_statistics(self, data, target):
         size_dataset = len(data)
         self.dataset_statistics = self.tpl(None, None, None, size_dataset)
         self.has_constant_statistics = True
    
    def calculate_statistics(self, subgroup, target, data, statistics=None):
        cover_arr, size_sg = get_cover_array_and_size(subgroup, len(data), data)

        mask = cover_arr.representation if hasattr(cover_arr, "representation") else cover_arr
        mask = np.asarray(mask)
        sg_data = data.iloc[np.where(mask)[0]]


        if size_sg >= 20:
            try:
                pls_sg = Plspm(sg_data, self.config, Scheme.PATH, 100, 0.00000001, False)

                model_sg = pls_sg.inner_model()
                model_sg = model_sg.rename(columns={"p>|t|": "p"})

                pls_ds = Plspm(data, self.config, Scheme.PATH, 100, 0.00000001, False)
                model_ds = pls_ds.inner_model()
                model_ds = model_ds.rename(columns={"p>|t|": "p_ds", "estimate": "estimate_ds"})

                joint_model = model_ds.copy()
                joint_model[["estimate", "p"]] = model_sg[["estimate", "p"]]
                changes_sig_sg = 0
                flip_sign_sg = 0

                for _, row in joint_model.iterrows():
                    if ((row["p"] <= 0.05 and row["p_ds"] > 0.05) or
                        (row["p"] > 0.05 and row["p_ds"] <= 0.05)):
                        changes_sig_sg += 1

                    if ((row["p"] <= 0.05 and row["p_ds"] <= 0.05) and
                        (np.sign(row["estimate"]) * np.sign(row["estimate_ds"]) == -1)):
                        flip_sign_sg += 1
                
                size_penalty = np.minimum(1, (size_sg/self.min_size_sg))
                quality = size_penalty * (self.weight_sig*changes_sig_sg + self.weight_sign*flip_sign_sg)
            except Exception:
                return self.tpl(0, 0, 0, size_sg)

        else:
            changes_sig_sg = 0
            flip_sign_sg = 0
            quality = 0
            
        return self.tpl(changes_sig_sg, flip_sign_sg, quality, size_sg)
    
    def evaluate(self, subgroup, target, data, statistics=None):
        statistics = self.ensure_statistics(subgroup, target, data, statistics)
        return statistics.quality

    def optimistic_estimate(self, subgroup, target, data, statistics=None):
        statistics = self.ensure_statistics(subgroup, target, data, statistics)
        return statistics.quality
    

class SEMQFNewSig(AbstractInterestingnessMeasure):
    """
    Rewards paths with new significant paths. In case of significant both in original and in sg model, rewards sign flip. Penalizes too small subgroups
    Requires minimum desired sg size (under which the QF penalizes)
    Only sg with >=20 elements are considered due to PLS SEM restrictions.
    """
    tpl = namedtuple("SEMQFNewSig_tpl", ["changes_sig_sg", "flip_sign_sg", "quality", "size_sg"])

    def __init__(self, config, weight_sig=1, weight_sign=1, min_sg_size=30):
        self.config = config
        self.weight_sig = weight_sig
        self.weight_sign = weight_sign
        self.min_size_sg = min_sg_size
        self.has_constant_statistics = False
        self.required_stat_attrs = self.tpl._fields
        self.dataset_statistics = None

    def calculate_constant_statistics(self, data, target):
         size_dataset = len(data)
         self.dataset_statistics = self.tpl(None, None, None, size_dataset)
         self.has_constant_statistics = True
    
    def calculate_statistics(self, subgroup, target, data, statistics=None):
        cover_arr, size_sg = get_cover_array_and_size(subgroup, len(data), data)

        mask = cover_arr.representation if hasattr(cover_arr, "representation") else cover_arr
        mask = np.asarray(mask)
        sg_data = data.iloc[np.where(mask)[0]]

        if size_sg >= 20:
            try:
                pls_sg = Plspm(sg_data, self.config, Scheme.PATH, 100, 0.00000001, False)

                model_sg = pls_sg.inner_model()
                model_sg = model_sg.rename(columns={"p>|t|": "p"})

                pls_ds = Plspm(data, self.config, Scheme.PATH, 100, 0.00000001, False)
                model_ds = pls_ds.inner_model()
                model_ds = model_ds.rename(columns={"p>|t|": "p_ds", "estimate": "estimate_ds"})

                joint_model = model_ds.copy()
                joint_model[["estimate", "p"]] = model_sg[["estimate", "p"]]
                changes_sig_sg = 0
                flip_sign_sg = 0

                for _, row in joint_model.iterrows():
                    if (row["p"] <= 0.05 and row["p_ds"] > 0.05):
                        changes_sig_sg += 1

                    if ((row["p"] <= 0.05 and row["p_ds"] <= 0.05) and
                        (np.sign(row["estimate"]) * np.sign(row["estimate_ds"]) == -1)):
                        flip_sign_sg += 1
                
                size_penalty = np.minimum(1, (size_sg/self.min_size_sg))
                quality = size_penalty * (self.weight_sig*changes_sig_sg + self.weight_sign*flip_sign_sg)
            except Exception:
                return self.tpl(0, 0, 0, size_sg)

        else:
            changes_sig_sg = 0
            flip_sign_sg = 0
            quality = 0
            
        return self.tpl(changes_sig_sg, flip_sign_sg, quality, size_sg)
    
    def evaluate(self, subgroup, target, data, statistics=None):
        statistics = self.ensure_statistics(subgroup, target, data, statistics)
        return statistics.quality

    def optimistic_estimate(self, subgroup, target, data, statistics=None):
        statistics = self.ensure_statistics(subgroup, target, data, statistics)
        return statistics.quality



class SEMQFEntropy(AbstractInterestingnessMeasure):
    """
    Rewards new significant paths. In case of significant both in original and in sg mode, rewards sign flip. 
    Sg size moderated by entropy measure. Does not require min sg size to be defined. 
    Only sg with >=20 elements are considered due to PLS SEM restrictions.
    """
    tpl = namedtuple("SEMQFEntropy_tpl", ["changes_sig_sg", "flip_sign_sg", "quality", "size_sg"])

    def __init__(self, config, weight_sig=1, weight_sign=1):
        self.config = config
        self.weight_sig = weight_sig
        self.weight_sign = weight_sign
        self.has_constant_statistics = False
        self.required_stat_attrs = self.tpl._fields
        self.dataset_statistics = None

    def calculate_constant_statistics(self, data, target):
         size_dataset = len(data)
         self.dataset_statistics = self.tpl(None, None, None, size_dataset)
         self.has_constant_statistics = True
    
    def calculate_statistics(self, subgroup, target, data, statistics=None):
        cover_arr, size_sg = get_cover_array_and_size(subgroup, len(data), data)

        mask = cover_arr.representation if hasattr(cover_arr, "representation") else cover_arr
        mask = np.asarray(mask)
        sg_data = data.iloc[np.where(mask)[0]]

        if size_sg >= 20:
            try:
                pls_sg = Plspm(sg_data, self.config, Scheme.PATH, 100, 0.00000001, False)
                model_sg = pls_sg.inner_model()
                model_sg = model_sg.rename(columns={"p>|t|": "p"})

                pls_ds = Plspm(data, self.config, Scheme.PATH, 100, 0.00000001, False)
                model_ds = pls_ds.inner_model()
                model_ds = model_ds.rename(columns={"p>|t|": "p_ds", "estimate": "estimate_ds"})


                joint_model = model_ds.copy()
                joint_model[["estimate", "p"]] = model_sg[["estimate", "p"]]
                changes_sig_sg = 0
                flip_sign_sg = 0

                for _, row in joint_model.iterrows():
                    if (row["p"] <= 0.05 and row["p_ds"] > 0.05):
                        changes_sig_sg += 1

                    if ((row["p"] <= 0.05 and row["p_ds"] <= 0.05) and
                        (np.sign(row["estimate"]) * np.sign(row["estimate_ds"]) == -1)):
                        flip_sign_sg += 1
                
                p = size_sg / len(data)
                if (p == 0 or p==1):
                    quality = 0
                else:
                    entropy_size = -p * np.log(p) - (1 - p) * np.log(1 - p)
                    quality = entropy_size * (self.weight_sig*changes_sig_sg + self.weight_sign*flip_sign_sg)
                
            except Exception:
                return self.tpl(0, 0, 0, size_sg)

        else:
            changes_sig_sg = 0
            flip_sign_sg = 0
            quality = 0
            
        return self.tpl(changes_sig_sg, flip_sign_sg, quality, size_sg)
    
    def evaluate(self, subgroup, target, data, statistics=None):
        statistics = self.ensure_statistics(subgroup, target, data, statistics)
        return statistics.quality

    def optimistic_estimate(self, subgroup, target, data, statistics=None):
        statistics = self.ensure_statistics(subgroup, target, data, statistics)
        return statistics.quality

class SEMQFEntropyCIBoot(AbstractInterestingnessMeasure):
    """
    *CIs based on bootstrapping!
    Rewards paths for which the SG path estimate confidence interval is completely non-overlapping with the global model path estimate confidence interval. 
    Sg size moderated by entropy measure. Does not require min sg size to be defined. 
    Only sg with >=20 elements are considered due to PLS SEM restrictions.
    """
    tpl = namedtuple("SEMQFEntropyCI_tpl", ["changes_sig_sg", "flip_sign_sg", "quality", "size_sg"])

    def __init__(self, config):
        self.config = config
        self.has_constant_statistics = False
        self.required_stat_attrs = self.tpl._fields
        self.dataset_statistics = None

    def calculate_constant_statistics(self, data, target):
         size_dataset = len(data)
         self.dataset_statistics = self.tpl(None, None, None, size_dataset)
         self.has_constant_statistics = True
    
    def calculate_statistics(self, subgroup, target, data, statistics=None):
        cover_arr, size_sg = get_cover_array_and_size(subgroup, len(data), data)

        mask = cover_arr.representation if hasattr(cover_arr, "representation") else cover_arr
        mask = np.asarray(mask)
        sg_data = data.iloc[np.where(mask)[0]]

        if size_sg >= 20:
            try:
                pls_sg = Plspm(sg_data, self.config, Scheme.PATH, 100, 0.00000001, True)
                model_sg = pls_sg.inner_model()
                model_sg = model_sg.rename(columns={"p>|t|": "p"})
                boot_sg = pls_sg.bootstrap().paths()
                boot_sg = boot_sg.rename(columns={"perc.025": "lower_ci", "perc.975": "upper_ci"})
                model_sg = model_sg.join(boot_sg[["lower_ci", "upper_ci"]])

                pls_ds = Plspm(data, self.config, Scheme.PATH, 100, 0.00000001, True)
                model_ds = pls_ds.inner_model()
                model_ds = model_ds.rename(columns={"p>|t|": "p_ds", "estimate": "estimate_ds"})
                boot_ds = pls_ds.bootstrap().paths()
                boot_ds = boot_ds.rename(columns={"perc.025": "lower_ci_ds", "perc.975": "upper_ci_ds"})
                model_ds = model_ds.join(boot_ds[["lower_ci_ds", "upper_ci_ds"]])

                joint_model = model_ds.copy()
                joint_model[["estimate", "p", "lower_ci", "upper_ci"]] = model_sg[["estimate", "p", "lower_ci", "upper_ci"]]
                changes_sig_sg = 0
                flip_sign_sg = 0

                for _, row in joint_model.iterrows():
                    if (row["upper_ci"] < row["lower_ci_ds"] or row["upper_ci_ds"] < row["lower_ci"]):
                        changes_sig_sg += 1

                    if ((row["p"] <= 0.05 and row["p_ds"] <= 0.05) and
                        (np.sign(row["estimate"]) * np.sign(row["estimate_ds"]) == -1)):
                        flip_sign_sg += 1
                
                p = size_sg / len(data)
                if (p == 0 or p==1):
                    quality = 0
                else:
                    entropy_size = -p * np.log(p) - (1 - p) * np.log(1 - p)
                    quality = entropy_size * changes_sig_sg
                
            except Exception:
                return self.tpl(0, 0, 0, size_sg)

        else:
            changes_sig_sg = 0
            flip_sign_sg = 0
            quality = 0
            
        return self.tpl(changes_sig_sg, flip_sign_sg, quality, size_sg)
    
    def evaluate(self, subgroup, target, data, statistics=None):
        statistics = self.ensure_statistics(subgroup, target, data, statistics)
        return statistics.quality

    def optimistic_estimate(self, subgroup, target, data, statistics=None):
        statistics = self.ensure_statistics(subgroup, target, data, statistics)
        return statistics.quality
    
class SEMQFEntropyCI(AbstractInterestingnessMeasure):
    """
    Rewards paths for which the SG path estimate confidence interval is completely non-overlapping with the global model path estimate confidence interval. 
    Sg size moderated by entropy measure. Does not require min sg size to be defined. 
    Only sg with >=20 elements are considered due to PLS SEM restrictions.
    Requires z for CI. 
    """
    tpl = namedtuple("SEMQFEntropyCI_tpl", ["changes_sig_sg", "flip_sign_sg", "quality", "size_sg"])

    def __init__(self, config, z):
        self.config = config
        self.z = z
        self.has_constant_statistics = False
        self.required_stat_attrs = self.tpl._fields
        self.dataset_statistics = None

    def calculate_constant_statistics(self, data, target):
         size_dataset = len(data)
         self.dataset_statistics = self.tpl(None, None, None, size_dataset)
         self.has_constant_statistics = True
    
    def calculate_statistics(self, subgroup, target, data, statistics=None):
        cover_arr, size_sg = get_cover_array_and_size(subgroup, len(data), data)

        mask = cover_arr.representation if hasattr(cover_arr, "representation") else cover_arr
        mask = np.asarray(mask)
        sg_data = data.iloc[np.where(mask)[0]]

        if size_sg >= 20:
            try:
                pls_sg = Plspm(sg_data, self.config, Scheme.PATH, 1000, 0.00000001, False)
                model_sg = pls_sg.inner_model()
                model_sg = model_sg.rename(columns={"p>|t|": "p", "std error": "std_e"})
                

                pls_ds = Plspm(data, self.config, Scheme.PATH, 1000, 0.00000001, False)
                model_ds = pls_ds.inner_model()
                model_ds = model_ds.rename(columns={"p>|t|": "p_ds", "estimate": "estimate_ds", "std error": "std_e_ds"})
                

                joint_model = model_ds.copy()
                joint_model[["estimate", "p", "std_e"]] = model_sg[["estimate", "p", "std_e"]]
                joint_model["lower_ci"] = joint_model["estimate"] - self.z * joint_model["std_e"]
                joint_model["upper_ci"] = joint_model["estimate"] + self.z * joint_model["std_e"]
                joint_model["lower_ci_ds"] = joint_model["estimate_ds"] - self.z * joint_model["std_e_ds"]
                joint_model["upper_ci_ds"] = joint_model["estimate_ds"] + self.z * joint_model["std_e_ds"]
                disjoint_ci_count = 0
                flip_sign_sg = 0

                for _, row in joint_model.iterrows():
                    if (row["upper_ci"] < row["lower_ci_ds"] or row["upper_ci_ds"] < row["lower_ci"]):
                        disjoint_ci_count += 1

                    if ((row["p"] <= 0.05 and row["p_ds"] <= 0.05) and
                        (np.sign(row["estimate"]) * np.sign(row["estimate_ds"]) == -1)):
                        flip_sign_sg += 1
                
                p = size_sg / len(data)
                if (p == 0 or p==1):
                    quality = 0
                    entropy_size = 0
                else:
                    entropy_size = -p * np.log(p) - (1 - p) * np.log(1 - p)
                    quality = entropy_size * disjoint_ci_count

                
            except Exception:
                return self.tpl(0, 0, 0, size_sg)

        else:
            disjoint_ci_count = 0
            flip_sign_sg = 0
            quality = 0
            entropy_size = 0

        print("disjoint_ci_count = ", disjoint_ci_count)
        print("entropy_size = ", entropy_size)
        print("quality = ", quality)
            
        return self.tpl(disjoint_ci_count, flip_sign_sg, quality, size_sg)
    
    def evaluate(self, subgroup, target, data, statistics=None):
        statistics = self.ensure_statistics(subgroup, target, data, statistics)
        return statistics.quality

    def optimistic_estimate(self, subgroup, target, data, statistics=None):
        statistics = self.ensure_statistics(subgroup, target, data, statistics)
        return statistics.quality
    
class SEMQFEntropyCIComplement(AbstractInterestingnessMeasure):
    """
    Rewards paths for which the SG path estimate confidence interval is completely non-overlapping with the global model path estimate confidence interval. 
    Sg size moderated by entropy measure. Does not require min sg size to be defined. 
    Only sg with >=20 elements are considered due to PLS SEM restrictions.
    Requires z for CI. 
    """
    tpl = namedtuple("SEMQFEntropyCIComplement_tpl", ["changes_sig_sg", "flip_sign_sg", "quality", "size_sg"])

    def __init__(self, config, z):
        self.config = config
        self.z = z
        self.has_constant_statistics = False
        self.required_stat_attrs = self.tpl._fields
        self.dataset_statistics = None

    def calculate_constant_statistics(self, data, target):
         size_dataset = len(data)
         self.dataset_statistics = self.tpl(None, None, None, size_dataset)
         self.has_constant_statistics = True
    
    def calculate_statistics(self, subgroup, target, data, statistics=None):
        cover_arr, size_sg = get_cover_array_and_size(subgroup, len(data), data)

        mask = cover_arr.representation if hasattr(cover_arr, "representation") else cover_arr
        mask = np.asarray(mask)
        sg_data = data.iloc[np.where(mask)[0]]
        sg_complement = data.iloc[np.where(~mask)[0]]

        if size_sg >= 20:
            try:
                pls_sg = Plspm(sg_data, self.config, Scheme.PATH, 1000, 0.00000001, False)
                model_sg = pls_sg.inner_model()
                model_sg = model_sg.rename(columns={"p>|t|": "p", "std error": "std_e"})
                

                pls_ds = Plspm(sg_complement, self.config, Scheme.PATH, 1000, 0.00000001, False)
                model_ds = pls_ds.inner_model()
                model_ds = model_ds.rename(columns={"p>|t|": "p_ds", "estimate": "estimate_ds", "std error": "std_e_ds"})

                
                

                joint_model = model_ds.copy()
                joint_model[["estimate", "p", "std_e"]] = model_sg[["estimate", "p", "std_e"]]
                joint_model["lower_ci"] = joint_model["estimate"] - self.z * joint_model["std_e"]
                joint_model["upper_ci"] = joint_model["estimate"] + self.z * joint_model["std_e"]
                joint_model["lower_ci_ds"] = joint_model["estimate_ds"] - self.z * joint_model["std_e_ds"]
                joint_model["upper_ci_ds"] = joint_model["estimate_ds"] + self.z * joint_model["std_e_ds"]
                disjoint_ci_count = 0
                flip_sign_sg = 0

                for _, row in joint_model.iterrows():
                    if (row["upper_ci"] < row["lower_ci_ds"] or row["upper_ci_ds"] < row["lower_ci"]):
                        disjoint_ci_count += 1

                    if ((row["p"] <= 0.05 and row["p_ds"] <= 0.05) and
                        (np.sign(row["estimate"]) * np.sign(row["estimate_ds"]) == -1)):
                        flip_sign_sg += 1
                
                p = size_sg / len(data)
                if (p == 0 or p==1):
                    quality = 0
                    entropy_size = 0
                else:
                    entropy_size = -p * np.log(p) - (1 - p) * np.log(1 - p)
                    quality = entropy_size * disjoint_ci_count

                
            except Exception:
                return self.tpl(0, 0, 0, size_sg)

        else:
            disjoint_ci_count = 0
            flip_sign_sg = 0
            quality = 0
            entropy_size = 0

        print("disjoint_ci_count = ", disjoint_ci_count)
        print("entropy_size = ", entropy_size)
        print("quality = ", quality)
            
        return self.tpl(disjoint_ci_count, flip_sign_sg, quality, size_sg)
    
    def evaluate(self, subgroup, target, data, statistics=None):
        statistics = self.ensure_statistics(subgroup, target, data, statistics)
        return statistics.quality

    def optimistic_estimate(self, subgroup, target, data, statistics=None):
        statistics = self.ensure_statistics(subgroup, target, data, statistics)
        return statistics.quality
    
class SEMQFEntropyGoF(AbstractInterestingnessMeasure):
    """
    Rewards higher GoF (goodness of fit) than full dataset model.
    Sg size moderated by entropy measure. Does not require min sg size to be defined. 
    Only sg with >=20 elements are considered due to PLS SEM restrictions.
    """
    tpl = namedtuple("SEMQFEntropyGoF_tpl", ["gof", "gof_increase", "quality", "size_sg"])

    def __init__(self, config, weight_sig=1, weight_sign=1):
        self.config = config
        self.weight_sig = weight_sig
        self.weight_sign = weight_sign
        self.has_constant_statistics = False
        self.required_stat_attrs = self.tpl._fields
        self.dataset_statistics = None

    def calculate_constant_statistics(self, data, target):
         size_dataset = len(data)
         self.dataset_statistics = self.tpl(None, None, None, size_dataset)
         self.has_constant_statistics = True
    
    def calculate_statistics(self, subgroup, target, data, statistics=None):
        cover_arr, size_sg = get_cover_array_and_size(subgroup, len(data), data)

        mask = cover_arr.representation if hasattr(cover_arr, "representation") else cover_arr
        mask = np.asarray(mask)
        sg_data = data.iloc[np.where(mask)[0]]
        

        if size_sg >= 20:
            try:
                pls_sg = Plspm(sg_data, self.config, Scheme.PATH, 100, 0.00000001, False)
                gof_sg = pls_sg.goodness_of_fit()

                pls_ds = Plspm(data, self.config, Scheme.PATH, 100, 0.00000001, False)
                gof_ds = pls_ds.goodness_of_fit()

                p = size_sg / len(data)
                if (p == 0 or p==1):
                    quality = 0
                else:
                    entropy_size = -p * np.log(p) - (1 - p) * np.log(1 - p)
                    quality = entropy_size * np.maximum(0, (gof_sg - gof_ds))
                gof_increase = gof_sg - gof_ds
            except Exception:
                return self.tpl(0, 0, 0, size_sg)

        else:
            gof_sg = 0
            gof_increase = 0
            quality = 0
            
        return self.tpl(gof_sg, gof_increase, quality, size_sg)
    
    def evaluate(self, subgroup, target, data, statistics=None):
        statistics = self.ensure_statistics(subgroup, target, data, statistics)
        return statistics.quality

    def optimistic_estimate(self, subgroup, target, data, statistics=None):
        statistics = self.ensure_statistics(subgroup, target, data, statistics)
        return statistics.quality
    
class SEMQFTargetEntropy(AbstractInterestingnessMeasure):
    """
    Only considers direct paths to target variable! Requires PLS-SEM target variable name as parameter.
    Rewards new significant paths. In case of significant both in original and in sg model, rewards sign flip. 
    Sg size moderated by entropy measure. Does not require min sg size to be defined. 
    Only sg with >=20 elements are considered due to PLS SEM restrictions.
    """
    tpl = namedtuple("SEMQFEntropy_tpl", ["changes_sig_sg", "flip_sign_sg", "quality", "size_sg"])

    def __init__(self, config, target_col, weight_sig=1, weight_sign=1):
        self.config = config
        self.weight_sig = weight_sig
        self.weight_sign = weight_sign
        self.target_col = target_col
        self.has_constant_statistics = False
        self.required_stat_attrs = self.tpl._fields
        self.dataset_statistics = None

    def calculate_constant_statistics(self, data, target):
         size_dataset = len(data)
         self.dataset_statistics = self.tpl(None, None, None, size_dataset)
         self.has_constant_statistics = True
    
    def calculate_statistics(self, subgroup, target, data, statistics=None):
        cover_arr, size_sg = get_cover_array_and_size(subgroup, len(data), data)

        mask = cover_arr.representation if hasattr(cover_arr, "representation") else cover_arr
        mask = np.asarray(mask)
        sg_data = data.iloc[np.where(mask)[0]]

        if size_sg >= 20:
            try:
                pls_sg = Plspm(sg_data, self.config, Scheme.PATH, 100, 0.00000001, False)
                model_sg = pls_sg.inner_model()
                model_sg = model_sg.rename(columns={"p>|t|": "p"})

                pls_ds = Plspm(data, self.config, Scheme.PATH, 100, 0.00000001, False)
                model_ds = pls_ds.inner_model()
                model_ds = model_ds.rename(columns={"p>|t|": "p_ds", "estimate": "estimate_ds"})


                joint_model = model_ds.copy()
                joint_model[["estimate", "p"]] = model_sg[["estimate", "p"]]
                # only keep rows going to target col
                target_model = joint_model[joint_model["to"] == self.target_col]

                changes_sig_sg = 0
                flip_sign_sg = 0

                for _, row in target_model.iterrows():
                    if (row["p"] <= 0.05 and row["p_ds"] > 0.05):
                        changes_sig_sg += 1

                    if ((row["p"] <= 0.05 and row["p_ds"] <= 0.05) and
                        (np.sign(row["estimate"]) * np.sign(row["estimate_ds"]) == -1)):
                        flip_sign_sg += 1
                
                p = size_sg / len(data)
                if (p == 0 or p==1):
                    quality = 0
                else:
                    entropy_size = -p * np.log(p) - (1 - p) * np.log(1 - p)
                    quality = entropy_size * (self.weight_sig*changes_sig_sg + self.weight_sign*flip_sign_sg)
                
            except Exception:
                return self.tpl(0, 0, 0, size_sg)

        else:
            changes_sig_sg = 0
            flip_sign_sg = 0
            quality = 0
            
        return self.tpl(changes_sig_sg, flip_sign_sg, quality, size_sg)
    
    def evaluate(self, subgroup, target, data, statistics=None):
        statistics = self.ensure_statistics(subgroup, target, data, statistics)
        return statistics.quality

    def optimistic_estimate(self, subgroup, target, data, statistics=None):
        statistics = self.ensure_statistics(subgroup, target, data, statistics)
        return statistics.quality
    
class SEMQFTarget(AbstractInterestingnessMeasure):
    """
    Only considers direct paths to target variable! Requires PLS-SEM target variable name as parameter.
    Rewards new significant paths. In case of significant both in original and in sg model, rewards sign flip. 
    Sg size moderated by entropy measure. Does not require min sg size to be defined. 
    Only sg with >=20 elements are considered due to PLS SEM restrictions. 
    """
    tpl = namedtuple("SEMQFEntropy_tpl", ["changes_sig_sg", "flip_sign_sg", "quality", "size_sg"])

    def __init__(self, config, target_col, weight_sig=1, weight_sign=1, min_sg_size=30):
        self.config = config
        self.weight_sig = weight_sig
        self.weight_sign = weight_sign
        self.target_col = target_col
        self.has_constant_statistics = False
        self.min_size_sg = min_sg_size
        self.required_stat_attrs = self.tpl._fields
        self.dataset_statistics = None

    def calculate_constant_statistics(self, data, target):
         size_dataset = len(data)
         self.dataset_statistics = self.tpl(None, None, None, size_dataset)
         self.has_constant_statistics = True
    
    def calculate_statistics(self, subgroup, target, data, statistics=None):
        cover_arr, size_sg = get_cover_array_and_size(subgroup, len(data), data)

        mask = cover_arr.representation if hasattr(cover_arr, "representation") else cover_arr
        mask = np.asarray(mask)
        sg_data = data.iloc[np.where(mask)[0]]

        if size_sg >= 20:
            try:
                pls_sg = Plspm(sg_data, self.config, Scheme.PATH, 100, 0.00000001, False)
                model_sg = pls_sg.inner_model()
                model_sg = model_sg.rename(columns={"p>|t|": "p"})

                pls_ds = Plspm(data, self.config, Scheme.PATH, 100, 0.00000001, False)
                model_ds = pls_ds.inner_model()
                model_ds = model_ds.rename(columns={"p>|t|": "p_ds", "estimate": "estimate_ds"})


                joint_model = model_ds.copy()
                joint_model[["estimate", "p"]] = model_sg[["estimate", "p"]]
                # only keep rows going to target col
                target_model = joint_model[joint_model["to"] == self.target_col]

                changes_sig_sg = 0
                flip_sign_sg = 0

                for _, row in target_model.iterrows():
                    if (row["p"] <= 0.05 and row["p_ds"] > 0.05):
                        changes_sig_sg += 1

                    if ((row["p"] <= 0.05 and row["p_ds"] <= 0.05) and
                        (np.sign(row["estimate"]) * np.sign(row["estimate_ds"]) == -1)):
                        flip_sign_sg += 1
                
                p = size_sg / len(data)
                if (p == 0 or p==1):
                    quality = 0
                else:
                    size_penalty = np.minimum(1, (size_sg/self.min_size_sg))
                    quality = size_penalty * (self.weight_sig*changes_sig_sg + self.weight_sign*flip_sign_sg)
                
            except Exception:
                return self.tpl(0, 0, 0, size_sg)

        else:
            changes_sig_sg = 0
            flip_sign_sg = 0
            quality = 0
            
        return self.tpl(changes_sig_sg, flip_sign_sg, quality, size_sg)
    
    def evaluate(self, subgroup, target, data, statistics=None):
        statistics = self.ensure_statistics(subgroup, target, data, statistics)
        return statistics.quality

    def optimistic_estimate(self, subgroup, target, data, statistics=None):
        statistics = self.ensure_statistics(subgroup, target, data, statistics)
        return statistics.quality
    
class SEMQFAllPosTargetEntropy(AbstractInterestingnessMeasure):
    """
    Only considers direct paths to target variable! Requires PLS-SEM target variable name as parameter.
    Discovers groups where all direct paths to the target variable are significant positive.
    Sg size moderated by entropy measure. Does not require min sg size to be defined. 
    Only sg with >=20 elements are considered due to PLS SEM restrictions. 
    """
    tpl = namedtuple("SEMQFEntropy_tpl", ["changes_sig_sg", "flip_sign_sg", "quality", "size_sg"])

    def __init__(self, config, target_col):
        self.config = config
        self.target_col = target_col
        self.has_constant_statistics = False
        self.required_stat_attrs = self.tpl._fields
        self.dataset_statistics = None

    def calculate_constant_statistics(self, data, target):
         size_dataset = len(data)
         self.dataset_statistics = self.tpl(None, None, None, size_dataset)
         self.has_constant_statistics = True
    
    def calculate_statistics(self, subgroup, target, data, statistics=None):
        cover_arr, size_sg = get_cover_array_and_size(subgroup, len(data), data)

        mask = cover_arr.representation if hasattr(cover_arr, "representation") else cover_arr
        mask = np.asarray(mask)
        sg_data = data.iloc[np.where(mask)[0]]

        if size_sg >= 20:
            try:
                pls_sg = Plspm(sg_data, self.config, Scheme.PATH, 100, 0.00000001, False)
                model_sg = pls_sg.inner_model()
                model_sg = model_sg.rename(columns={"p>|t|": "p"})

                pls_ds = Plspm(data, self.config, Scheme.PATH, 100, 0.00000001, False)
                model_ds = pls_ds.inner_model()
                model_ds = model_ds.rename(columns={"p>|t|": "p_ds", "estimate": "estimate_ds"})


                joint_model = model_ds.copy()
                joint_model[["estimate", "p"]] = model_sg[["estimate", "p"]]
                # only keep rows going to target col
                target_model = joint_model[joint_model["to"] == self.target_col]

                all_pos = 0

                all_pos = int(((target_model["p"] <= 0.05) & (target_model["estimate"] > 0)).all())
                
                p = size_sg / len(data)
                if (p == 0 or p==1):
                    quality = 0
                else:
                    entropy_size = -p * np.log(p) - (1 - p) * np.log(1 - p)
                    quality = entropy_size * all_pos
                
            except Exception:
                return self.tpl(0, 0, 0, size_sg)

        else:
            quality = 0
            
        return self.tpl(0, 0, quality, size_sg)
    
    def evaluate(self, subgroup, target, data, statistics=None):
        statistics = self.ensure_statistics(subgroup, target, data, statistics)
        return statistics.quality

    def optimistic_estimate(self, subgroup, target, data, statistics=None):
        statistics = self.ensure_statistics(subgroup, target, data, statistics)
        return statistics.quality
    
class SEMQFAllNegTargetEntropy(AbstractInterestingnessMeasure):
    """
    Only considers direct paths to target variable! Requires PLS-SEM target variable name as parameter.
    Discovers groups where all direct paths to the target variable are significant negative.
    Sg size moderated by entropy measure. Does not require min sg size to be defined. 
    Only sg with >=20 elements are considered due to PLS SEM restrictions. 
    """
    tpl = namedtuple("SEMQFEntropy_tpl", ["changes_sig_sg", "flip_sign_sg", "quality", "size_sg"])

    def __init__(self, config, target_col):
        self.config = config
        self.target_col = target_col
        self.has_constant_statistics = False
        self.required_stat_attrs = self.tpl._fields
        self.dataset_statistics = None

    def calculate_constant_statistics(self, data, target):
         size_dataset = len(data)
         self.dataset_statistics = self.tpl(None, None, None, size_dataset)
         self.has_constant_statistics = True
    
    def calculate_statistics(self, subgroup, target, data, statistics=None):
        cover_arr, size_sg = get_cover_array_and_size(subgroup, len(data), data)

        mask = cover_arr.representation if hasattr(cover_arr, "representation") else cover_arr
        mask = np.asarray(mask)
        sg_data = data.iloc[np.where(mask)[0]]
        #print(len(sg_data))

        if size_sg >= 20:
            try:
                pls_sg = Plspm(sg_data, self.config, Scheme.PATH, 100, 0.00000001, False)
                model_sg = pls_sg.inner_model()
                model_sg = model_sg.rename(columns={"p>|t|": "p"})

                pls_ds = Plspm(data, self.config, Scheme.PATH, 100, 0.00000001, False)
                model_ds = pls_ds.inner_model()
                model_ds = model_ds.rename(columns={"p>|t|": "p_ds", "estimate": "estimate_ds"})


                joint_model = model_ds.copy()
                joint_model[["estimate", "p"]] = model_sg[["estimate", "p"]]
                # only keep rows going to target col
                target_model = joint_model[joint_model["to"] == self.target_col]

                all_pos = 0

                all_pos = int(((target_model["p"] <= 0.05) & (target_model["estimate"] < 0)).all())
                
                p = size_sg / len(data)
                if (p == 0 or p==1):
                    quality = 0
                else:
                    entropy_size = -p * np.log(p) - (1 - p) * np.log(1 - p)
                    quality = entropy_size * all_pos
                
            except Exception:
                return self.tpl(0, 0, 0, size_sg)

        else:
            quality = 0
            
        return self.tpl(0, 0, quality, size_sg)
    
    def evaluate(self, subgroup, target, data, statistics=None):
        statistics = self.ensure_statistics(subgroup, target, data, statistics)
        return statistics.quality

    def optimistic_estimate(self, subgroup, target, data, statistics=None):
        statistics = self.ensure_statistics(subgroup, target, data, statistics)
        return statistics.quality
    