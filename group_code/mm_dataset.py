from group_code.helper import combine_dataframes
from group_code.uni_model.ecg import ECG_uni
from group_code.uni_model.echo import ECHO_uni
from group_code.uni_model.ehr import EHR_uni
from mmai25_hackathon.dataset import BaseDataset


class EEE_dataset(BaseDataset):
    def __init__(self, data_roots, data_mods):
        self.uni_dict = {}
        for mod, r_path in zip(data_mods, data_roots):
            if mod == "ecg":
                self.uni_dict[mod] = ECG_uni(r_path)
            elif mod == "ehr":
                self.uni_dict[mod] = EHR_uni(r_path)
            elif mod == "echo":
                self.uni_dict[mod] = ECHO_uni(r_path)
            else:
                print("Modality not supported.")

        self.combined_records = combine_dataframes(
            [val.records for key, val in self.uni_dict.items() if key != "ehr"],
            [key for key, val in self.uni_dict.items() if key != "ehr"],
        )

    def __len__(self) -> int:
        return len(self.combined_records)

    def __getitem__(self, idx: int):
        subject_id = self.combined_records.iloc[idx]["subject_id"]
        return_dict = {}
        for key, val in self.uni_dict.items():
            return_dict[key] = val.fetch(subject_id)
        return return_dict

    def __add__(self, data_roots, data_mods):
        for mod, r_path in zip(data_mods, data_roots):
            if mod == "ecg":
                self.uni_dict[mod] = ECG_uni(r_path)
        self.combined_records = combine_dataframes(
            [val.records for key, val in self.uni_dict.items() if key != "ehr"],
            [key for key, val in self.uni_dict.items() if key != "ehr"],
        )

    def get_idx_from_sub_id(self, subject_id):
        return self.combined_records.index[self.combined_records["subject_id"] == subject_id][0]
