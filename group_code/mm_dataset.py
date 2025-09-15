from group_code.uni_model.ehr import EHR_uni
from mmai25_hackathon.dataset import * 
from group_code.uni_model.ecg import * 
from group_code.helper import * 
from group_code.uni_model.ehr import * 

class EEE_dataset(BaseDataset):
    def __init__(self, data_roots, data_mods):
        uni_dict = {}
        for mod, r_path in zip(data_mods, data_roots):
            if mod == 'ecg':
                uni_dict[mod] = ECG_uni(r_path)
            elif mod == 'ehr':
                uni_dict[mod] = EHR_uni(r_path)
            else: 
                print("Modality not supported.")

        self.combined_records = combine_dataframes(
            [uni.records for uni in uni_dict.values()], 
            uni_dict.keys())

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
            if mod == 'ecg':
                self.uni_dict[mod] = ECG_uni(r_path)
