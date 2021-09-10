from pathlib import Path
import warnings
import json
import numpy as np
from io_m.libsaving import ResultsSaver
from io_m import io_utils

class SoftMarginSaver(ResultsSaver):

    SOURCES_SAVE = "src_probs"
    OBSERV_SAVE = "obs_t_{}_nso_{}"
    PARAM_SAVE = "pars"
    MARGINALS_SAVE = "margs"

    def __init__(self, epinstance, results_root_fold, n_sims_per_iter, n_iterations,
        inter_fold_name=None):
        super().__init__(epinstance, results_root_fold, inter_fold_name)
        self.n_sims_p_it = n_sims_per_iter
        self.n_iter = n_iterations

    
    def get_save_file_name(self, what, version=2):
        """
        Get the base name for the files, according to the format
        """
        if version < 2:
            return "{}_{}".format(self.base_name, what)
        n_epi_print = self.n_sims_p_it/1000
        return "{}_nepi_{}_niter_{}_{}".format(self.base_name, n_epi_print, self.n_iter, what)

    def _save_a_pars_inst_p(self, a_pars, probs_dict, name_file, overwrite):
        """
        DO THE SAVING
        """
        outfile = self.folder / name_file
        self.save_apars_srcprobs(a_pars, probs_dict, outfile, overwrite)

    @staticmethod
    def save_apars_srcprobs(a_pars, probs_dict, out_file, overwrite=False):
        """
        Save a parameters and source probabilities
        """
        try:
            out_file.suffix
        except AttributeError:
            out_file = Path(out_file)
        if out_file.suffix != ".npz":
            ## if we don't have the correct file type, append it
            out_file = out_file.with_suffix(out_file.suffix + ".npz")
        print(out_file)

        out_dict = {"inst_{}".format(i) : prob for i, prob in probs_dict.items()}
        out_dict["a_params"] = a_pars
        if not overwrite and out_file.exists():
            raise ValueError("File already exists. Rerun with overwrite=True to overwrite")

        np.savez_compressed(out_file, **out_dict)

    @staticmethod
    def save_margs_(margs, out_file, n_insts, overwrite=False):
        """
        Save marginals
        """
        try:
            out_file.suffix
        except AttributeError:
            out_file = Path(out_file)
        if out_file.suffix != ".npz":
            ## if we don't have the correct file type, append it
            out_file = out_file.with_suffix(out_file.suffix + ".npz")
        print(out_file)
        mg_good = set(range(n_insts))
        for i in range(n_insts):
            try:
                l = margs[i]
                if len(l) == 0:
                    mg_good.remove(i)
            except KeyError:
                continue
        
        out_dict = {"margs_{}".format(i) : margs[i] for i in sorted(mg_good)}
        if not overwrite and out_file.exists():
            raise ValueError("File already exists. Rerun with overwrite=True to overwrite")

        np.savez_compressed(out_file, **out_dict)

    @staticmethod
    def load_margs_(outfile_with_ext):
        """
        Load marginals
        """
        data = np.load(outfile_with_ext)
        margs = {}
        for name in data.files:
            if "margs" in name:
                idx = int(name.split("_")[-1])
                margs[idx] = data[name]
            else:
                warnings.warn("No matches for name "+name+" in the marginals file")

        data.close()
        return margs

    @staticmethod
    def load_apars_srcprobs(outfile_with_ext):

        data = np.load(outfile_with_ext)
        inst_probs = {}
        for name in data.files:
            try:
                idx = int(name.split("_")[-1])
            except ValueError as err:
                if name == "a_params":
                    continue
                else:
                    raise IndexError from err
            inst_probs[idx] = data[name]

        a_pars = data["a_params"]

        data.close()
        return a_pars, inst_probs
    
    def save_src_probabilities(self, dict_probabs, a_pars, overwrite=False):
        """
        Save final source probabilities and parameters
        """
        self.create_folder()
        name_file = self.get_save_file_name(self.SOURCES_SAVE)
        
        self._save_a_pars_inst_p(a_pars, dict_probabs, name_file, overwrite)

    def save_marginals(self, margins, n_insts, overwrite=False):
        self.create_folder()
        name_file = self.get_save_file_name(self.MARGINALS_SAVE)
        self.save_margs_(margins, name_file, n_insts, overwrite)

    def load_src_probabilities(self, version=2):
        """
        Load probabilities from the folder
        """
        self.check_folder("Folder doesn't exists")

        name_file = self.get_save_file_name(self.SOURCES_SAVE, version)
        outfile_with_ext = self.get_file_path(name_file, ".npz")

        print(outfile_with_ext)

        return self.load_apars_srcprobs(outfile_with_ext)

    def save_probs_observ(self, dict_probabs, a_pars, n_epi_obs, t_obs, overwrite=False):
        
        self.create_folder()
        name = self.OBSERV_SAVE.format(t_obs,n_epi_obs/1000)
        name_file = self.get_save_file_name(name)

        self._save_a_pars_inst_p(a_pars, dict_probabs, name_file, overwrite)

    def load_probs_observ(self, n_epi_obs, t_obs, version=2):

        self.check_folder("Folder doesn't exists")
        name = self.OBSERV_SAVE.format(t_obs,n_epi_obs/1000)
        name_file = self.get_save_file_name(name, version)
        outfile_with_ext = self.get_file_path(name_file, ".npz")

        print(outfile_with_ext)

        return self.load_apars_srcprobs(outfile_with_ext)

    def save_params(self, extra_parameters=None):
        """
        Save parameters
        """
        p = self.get_file_path(self.get_save_file_name(self.PARAM_SAVE),".json")

        pars_out = {"n_iter":self.n_iter, "epi_per_a":self.n_sims_p_it}
        if extra_parameters is not None:
            pars_out.update(extra_parameters)
        
        with open(p,"w") as f:
            json.dump(pars_out,f)
        print("Written pars on {}".format(p))

def load_data_softmargin(file_name):
    """
    Load softmargin data from custom file
    """
    fname_pars = file_name +"_pars.json"
    fname_data = file_name + "_probs.npz"

    a_pars, src_probs = SoftMarginSaver.load_apars_srcprobs(fname_data)

    pars_extra = io_utils.load_json(fname_pars)

    return a_pars, src_probs, pars_extra