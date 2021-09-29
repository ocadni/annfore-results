"""
Module to help saving data after whatever algorithm has been run

In some cases, it's also useful to load the data.

Author: Fabio Mazza
"""
from pathlib import Path
import json
import warnings
import numpy as np

from . import io_utils

#from nn_sir_path import SirPathTimesNet,SirPath
#from nn_general import NetContainer

"""
Pylint args to avoid errors "torch has no member..."
--generated-members=torch.*
"""


class ResultsSaver:
    """
    Basic class to create the folders and ease results saving
    (and possibly loading)
    """
    def __init__(self, epinstance, results_root_fold, inter_fold_name=None):
        self.instance = epinstance
        self._root_path_fold = Path(results_root_fold)
        if inter_fold_name is not None:
            second_fold = self._root_path_fold / inter_fold_name
        else:
            second_fold = self._root_path_fold
        self.folder = second_fold / epinstance.type_graph / \
            f"N_{epinstance.n}_T_{epinstance.t_limit}"
        self.base_name = f"lam_{epinstance.lambda_}_mu_{epinstance.mu}"
        self.base_name += f"_s_{epinstance.seed}_pe_{epinstance.p_edge}"

    def folder_exists(self):
        return self.folder.exists()

    def create_folder(self):
        if not self.folder.exists():
            self.folder.mkdir(parents=True)
    def check_folder(self,err_message=""):
        """
        Check if folder exists, raise error if you need it
        """
        if not self.folder.exists():
            if len(err_message) > 0:
                raise ValueError(err_message)
            else:
                return False
        else:
            return True
    
    def get_file_path(self, name_file, suffix=None):
        """
        Get the file path, from name, setting its additional suffix
        """
        outfile = self.folder / name_file
        if suffix is not None:
            outfile_with_ext = outfile.with_suffix(outfile.suffix + suffix)
        else:
            outfile_with_ext = outfile
        return outfile_with_ext


class NetResultsSaver(ResultsSaver):

    BASE_NAME = "lam_{}_mu_{}_hsrc_{}_hii_{}"

    def __init__(self, epinstance, net_dirname, model_pars, training_pars, results_root="res",
                 allow_overwriting=False,legacy=False):
        super().__init__(epinstance, results_root, net_dirname)
        #self.problem_pars = problem_pars
        self.type_graph = epinstance.type_graph
        self.net_dirname = net_dirname
        self.training_pars = training_pars.get_for_saving()
        self.model_pars = model_pars
        self.overwriting_allowed = allow_overwriting
        self.current_base_saving = None

        self.base_name = f"lam_{epinstance.lambda_}_mu_{epinstance.mu}"
        self.base_name += f"_hii_{model_pars['h_ii']}_hsrc_{model_pars['h_source']}"


    def get_train_name(self):
        return self.BASE_NAME

    def get_nn_save_format(self, base_name, it, overwrite=False):
        trial_name = base_name + "_nn_{}".format(it)
        base_path = self.folder / trial_name
        path_sources = base_path.with_suffix(base_path.suffix+".sources")
        if path_sources.exists() and not (self.overwriting_allowed or overwrite):
            raise ValueError(
                "Sources file already exist. This iteration has already been run.")
        self.current_base_saving = base_path.as_posix()
        return base_path.as_posix()

    def save_inst_params(self, indent=None):
        out_data = {
            "training": self.training_pars,
            "model": self.model_pars
        }
        f = open(self.current_base_saving+"_pars.json", "w")
        json.dump(out_data, f, indent=indent)
        f.close()


class SibResultsSaver(ResultsSaver):
    """
    Saver for a sib instance
    """

    MARGINALS_NAME = "marg"
    INFOS_NAME = "info"

    def get_name_file(self, t_obs, what):
        """
        Get correct file name
        """
        return "{}_{}_tobs_{}".format(self.base_name, what, t_obs)

    def get_file_path_ext(self, t_obs, name, suffix):
        """
        Get file path with extension
        """
        name_file = self.get_name_file(t_obs, name)

        outfile = self.folder / name_file
        outfile_with_ext = outfile.with_suffix(outfile.suffix + suffix)

        return outfile, outfile_with_ext
    
    def save_extra_info(self, info, t_obs, overwrite=False):
        """
        Save extra info
        """
        self.create_folder()
        _, outfile_with_ext = self.get_file_path_ext(t_obs, self.INFOS_NAME, ".json.gz")

        if not overwrite and outfile_with_ext.exists():
            raise ValueError("File already exists. Rerun with overwrite=True to overwrite")
        io_utils.save_json_gzip(outfile_with_ext, info)
        print(outfile_with_ext)

    def load_extra_info(self, t_obs):
        """
        Load info on sib run
        """
        _, outfile_with_ext = self.get_file_path_ext(t_obs, self.INFOS_NAME, ".json.gz")
        return io_utils.load_json_gzip(outfile_with_ext)

    def save_margs_npz(self,margs_list, m_file, overwrite=False):
        """
        Global saving method for the marginals
        """
        self.create_folder()
        names = [f"{i}" for i in range(len(margs_list))]
        f = Path(m_file)
        print(f)
        if not overwrite and f.exists():
            raise ValueError("File already exists. Rerun with overwrite=True to overwrite")
        np.savez_compressed(f, **dict(zip(names, margs_list)))
    
    def save_margs_time(self, marginals, t_obs, overwrite=False):
        """
        Save marginals in the folder.
        The file in which it's saved is gets the name from the observation
        """
        outfile, outfile_with_ext = self.get_file_path_ext(t_obs, self.MARGINALS_NAME, ".npz")
        self.save_margs_npz(marginals, outfile_with_ext, overwrite=overwrite)

    def save_margs_all(self, marginals, overwrite=False):
        """
        Save marginals for all the times
        """
        _,  outfile_with_ext = self.get_file_path_ext("all", self.MARGINALS_NAME, ".npz")
        self.save_margs_npz(marginals, outfile_with_ext, overwrite=overwrite)

    def _load_margs(self, outfile):
        data = np.load(outfile)
        r = {int(key): data[key] for key in data.files}
        return [r[k] for k in sorted(r.keys())]

    def load_margs_time(self, t_obs):
        """
        Load marginals from the folder, given t_obs
        """
        self.check_folder("Folder does not exist")
        _, outfile_with_ext = self.get_file_path_ext(t_obs, self.MARGINALS_NAME, ".npz")
        print(outfile_with_ext)
       
        return self._load_margs(outfile_with_ext)

    def save_rest_data(self, sib_mod, convergence, sib_params, t_obs, extra_pars={},overwrite=False):
        """
        Save all data of a sib run
        """
        #self.save_margs_time(marginals, t_obs, overwrite)
        out_info = {
            "sib_version":sib_mod.version(),
            "converged": convergence,
            "params":{
                "pseed":sib_params.pseed,
                "psus": sib_params.psus
            }
        }
        out_info["params"].update(extra_pars)
        extra_info = io_utils.convert_dtypes_py(out_info)
        self.save_extra_info(extra_info, t_obs, overwrite)
