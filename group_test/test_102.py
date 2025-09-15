from group_code.mm_dataset import EEE_dataset
from group_code.uni_model.ecg import ECG_uni
from group_code.uni_model.echo import ECHO_uni
from group_code.uni_model.ehr import EHR_uni

ecg_root = "mimic-iv/mimic-iv-ecg-diagnostic-electrocardiogram-matched-subset-1.0/"
echo_root = "mimic-iv/mimic-iv-echo-0.1.physionet.org/"
ehr_root = "mimic-iv/mimic-iv-3.1/"


def test_by_id(id):
    ds = EEE_dataset([ecg_root, ehr_root, echo_root], ["ecg", "ehr", "echo"])

    ds_idx = ds.get_idx_from_sub_id(id)
    full_results = ds[ds_idx]

    ecg = ECG_uni(ecg_root)
    ehr = EHR_uni(ehr_root)
    echo = ECHO_uni(echo_root)

    ecg_res = ecg.fetch(id)
    ehr_res = ehr.fetch(id)
    echo_res = echo.fetch(id)

    assert full_results["ecg"][id][1]["comments"] == ecg_res[id][1]["comments"]
    assert full_results["echo"][id]["Rows"] == echo_res[id]["Rows"]
    assert len(full_results["ehr"][id]) == len(ehr_res[id])


if __name__ == "__main__":
    test_by_id(102)
    test_by_id(101)
