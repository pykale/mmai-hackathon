# noqa: F403
from torch.utils.data import Dataset

from mmai25_hackathon.load_data.echo import load_echo_dicom, load_mimic_iv_echo_record_list


class ECHO_uni(Dataset):
    def __init__(self, mod_root):
        self.records = load_mimic_iv_echo_record_list(mod_root)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int):
        subject_id = self.records.iloc[idx]["subject_id"]
        frames, meta = load_echo_dicom(self.records.iloc[idx]["echo_path"])
        meta_filtered = {
            k: meta[k] for k in ("NumberOfFrames", "Rows", "Columns", "FrameTime", "CineRate") if k in meta
        }
        return {subject_id: meta_filtered}

    def get_idx_by_subject(self, subject_id):
        return self.records.index[self.records["subject_id"] == subject_id][0]

    def fetch(self, subject_id):
        idx = self.get_idx_by_subject(subject_id)
        return self.__getitem__(idx)
