from mmai25_hackathon.dataset import * 
from mmai25_hackathon.load_data.ecg import * 
from torch.utils.data import Dataset

class ECG_uni(Dataset):
    def __init__(self, mod_root):
        self.records = load_mimic_iv_ecg_record_list(mod_root)
        
    def __len__(self) -> int:
        return len(self.records)
    
    def __getitem__(self, idx: int):
        subject_id = self.records.iloc[idx]["subject_id"]
        sig, fields = load_ecg_record(self.records.iloc[idx]["hea_path"])

        return {subject_id: [sig, fields]}
    
    def get_idx_by_subject(self, subject_id):
        return self.records.index[self.records["subject_id"] == subject_id][0]

    def fetch(self, subject_id):
        idx = self.get_idx_by_subject(subject_id)
        return self.__getitem__(idx)
