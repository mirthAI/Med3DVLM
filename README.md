# Med3DVLM: An Efficient Vision-Language Model for 3D Medical Image Analysis

<font size=3><div align='center' > <a href=https://arxiv.org/abs/2503.20047/>**Paper**</a> | [**Datasets**](#datasets) | [**Model**](#model) | [**Training**](#training) | [**Evaluation**](#evaluation) | [**Results**](#results)</div></font>

This branch of the Med3DVLM repository is fine-tuned on the FLARE-Task5-MLLM-3D dataset, which is a 3D medical image-text dataset for report generation and vision question answering tasks.

## Requirements
* Python==3.12.8
* torch==2.7.1
* torchvision==0.22.1
* monai==1.5.0
* deepspeed==0.17.2
* transformers==4.52.3

## Installation
First, clone the repository to your local machine:
```bash
git clone https://github.com/mirthAI/Med3DVLM.git
cd Med3DVLM
git checkout FLARE_2025
```
To install the required packages, you can use the following command:

```bash
pip install -r requirements.txt
```

## Datasets
In this branch, we train and evaluate our model on report generation and vision question answering tasks using the FLARE-Task5-MLLM-3D dataset.

 Dataset  | Type | Download Link |
| ------------- | ------------- | ------------- |
| FLARE-Task5-MLLM-3D | 3D image-text pairs | [HuggingFace](https://huggingface.co/datasets/FLARE-MedFM/FLARE-Task5-MLLM-3D)|

### Prepare data
Use the following command to download the datasets and convert them into 128x256x256 NIfTI format:

```bash
python flare_preprocess.py --data_dir path_to_FLARE-Task5-MLLM-3D --output_dir FLARE_npy
```

The directory structure after data preparation should look like this:

```bash
│
├── FLARE_npy
│
├── Other Folders
│
└── Other Files
```

## Pre-trained Models
We provide pre-trained weights for the Med3DVLM model. You can download them from the following link:
[HuggingFace](https://huggingface.co/MagicXin/Med3DVLM-Qwen-2.5-7B-FLARE2025)


## Training
We provide pre-trained weights for both the vision encoder and the multi-modal projector. You can download them from the following link: [HuggingFace](https://huggingface.co/MagicXin/Med3DVLM-Qwen-2.5-7B-FLARE2025/tree/main/pretrained)

### FLARE Fine-tuning
To fine-tune the VLM model, use the following command:

```bash
sh scripts/flare/finetune_flare.sh
```

The model will be saved in the `output/FLARE_VLM` folder.

## Evaluation
To evaluate the model, you need finish the data preparation first. 

### Report Generation
To evaluate the report generation task, use the following command:

```bash
sh scripts/flare/eval_report_generation.sh
```

### VQA Evaluation
To evaluate the VQA task, use the following command:

```bash
sh scripts/flare/eval_vqa.sh
```

## Results

### Report Generation
```json
{
    "green": 0.41844785827867276,
    "region_means": {
        "liver": 0.2775261904761905,
        "biliary system": 0.5122192273135671,
        "spleen": 0.6345744680851063,
        "pancreas": 0.5835138136887407,
        "kidneys": 0.24700117889773066,
        "endocrine system": 0.28041362530413627,
        "lymphatic system": 0.5643418467583496,
        "gastrointestinal tract": 0.11991951710261568,
        "abdominal cavity and peritoneum": 0.3961981566820276,
        "blood vessels": 0.12614555256064688,
        "musculoskeletal system": 0.53087899543379,
        "lungs and pleura": 0.2728758675187248,
        "respiratory tract": 0.7794486215538847,
        "heart": 0.6545246277205033,
        "mediastinum": 0.6117396593673964,
        "esophagus": 0.6408163265306124,
        "urinary system": 0.0,
        "breast tissue": 0.0,
        "diaphragm": 0.0
    }
}
```

### VQA
```json
{
    "global_accuracy": 0.235759,
    "local_accuracy": 0.417964
}
```

## Citations and Acknowledgements
The code is only for research purposes. If you have any questions regarding how to use this code, feel free to contact Yu Xin at yu.xin@ufl.edu.

Kindly cite the following papers if you use our code.

```bibtex
@article{xin2025med3dvlm,
  title={Med3DVLM: An Efficient Vision-Language Model for 3D Medical Image Analysis},
  author={Xin, Yu and Ates, Gorkem Can and Gong, Kuang and Shao, Wei},
  journal={arXiv preprint arXiv:2503.20047},
  year={2025}
}

@article{ates2025dcformer,
  title={DCFormer: Efficient 3D Vision-Language Modeling with Decomposed Convolutions},
  author={Ates, Gorkem Can and Gong, Kuang and Shao, Wei},
  journal={arXiv preprint arXiv:2502.05091},
  year={2025}
}

@article{tolstikhin2021mlp,
  title={Mlp-mixer: An all-mlp architecture for vision},
  author={Tolstikhin, Ilya O and Houlsby, Neil and Kolesnikov, Alexander and Beyer, Lucas and Zhai, Xiaohua and Unterthiner, Thomas and Yung, Jessica and Steiner, Andreas and Keysers, Daniel and Uszkoreit, Jakob and others},
  journal={Advances in neural information processing systems},
  volume={34},
  pages={24261--24272},
  year={2021}
}

@inproceedings{zhai2023sigmoid,
  title={Sigmoid loss for language image pre-training},
  author={Zhai, Xiaohua and Mustafa, Basil and Kolesnikov, Alexander and Beyer, Lucas},
  booktitle={Proceedings of the IEEE/CVF international conference on computer vision},
  pages={11975--11986},
  year={2023}
}
```
