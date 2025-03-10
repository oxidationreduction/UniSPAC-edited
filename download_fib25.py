import openxlab

openxlab.login(ak="lyobvxrgzazmxlg4r0qx", sk="zabpo5x9d1g28vl4ypyw7opnqbn7w3qmn6rqxkly")

from openxlab.dataset import download
download(dataset_repo='OpenScienceLab/FIB-25',source_path='/README.md', target_path='/home/liuhongyu2024/sshfs_share/liuhongyu2024/project/unispac/UniSPAC-edited/data/')
