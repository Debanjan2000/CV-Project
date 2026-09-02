from huggingface_hub import hf_hub_download
import os

def download_LTDv2():
    # download the LTDv2 dataset annotations from huggingface
    hf_hub_download(repo_id="vapaau/LTDv2", repo_type="dataset", filename="data/Valid.json")
    hf_hub_download(repo_id="vapaau/LTDv2", repo_type="dataset", filename="data/Train.json")
    hf_hub_download(repo_id="vapaau/LTDv2", repo_type="dataset", filename="data/TestNoLabels.json")
    # hf_hub_download(repo_id="vapaau/LTDv2", repo_type="dataset", filename="data/frames.zip")

if __name__ == "__main__":
    download_LTDv2()
