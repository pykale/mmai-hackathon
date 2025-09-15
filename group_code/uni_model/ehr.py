from mmai25_hackathon.dataset import * 
from mmai25_hackathon.load_data.ehr import * 

class EHR_uni(Dataset):
    def __init__(self, mod_root):
        self.root = mod_root
    
    def __len__(self) -> int:
        return len(self.records)
    
    def __getitem__(self, subject_id):
        return self.fetch(subject_id)

    def fetch(self, subject_id):
        dfs_new = load_mimic_iv_ehr(
            ehr_path=self.root,
            module="both",
            tables=["icustays", "admissions"],
            index_cols=["subject_id", "hadm_id"],
            subset_cols={
                "icustays": ["first_careunit"],
                "admissions": ["admittime"],
            },
            filter_rows={"subject_id": [int(subject_id)]},
            merge=True,
            join="inner",
        )
        return {subject_id: dfs_new}