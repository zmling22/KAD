# World Knowledge-Enhanced Reasoning Using Instruction-guided Interactor in Autonomous Driving

Offical implementation for AAAI 2025 paper "World Knowledge-Enhanced Reasoning Using Instruction-guided Interactor in Autonomous Driving" [[paper link]](https://arxiv.org/pdf/2412.06324)

![](./docs/images/zml-aaai-25.png) 

## Todo List
- [x] train code
- [x] inference code
- [ ] eval code
- [x] dataset

## Quick Start
### Environments
- CUDA and cuDNN

    We use CUDA 11.8 and cuDNN 8.7.0. We actually use the CUDA docker by NVIDIA: docker pull nvcr.io/nvidia/cuda:11.8.0-cudnn8-devel-ubuntu20.04. CUDA 12 is fine, too.

- Create a conda virtual environment and activate it:
    ```shell
    conda create -n kad python=3.10
    conda activate kad
    ```
- Basic requirements
    ```shell
    pip install --upgrade pip
    pip install transformers
    pip install torch torchvision xformers --index-url https://download.pytorch.org/whl/cu118
    ```
- Install flash-attention
  ```shell
  # https://github.com/Dao-AILab/flash-attention?tab=readme-ov-file#installation-and-features
  pip install packaging
  pip install flash-attn --no-build-isolation
  ```
- Install KAD and other requirements
  ```shell
  git clone https://github.com/KAD.git
  cd KAD
  pip install -e .
  ```
- Lora finetune
  ```shell
  sh ./script/train/finetune_lora.sh
  ```
- mult-node lora finetune
  ```shell
  sh ./script/train/finetune_lora_multi-nodes.sh
  ```

## Dataset
- Baidu Cloud: [[Link]](https://pan.baidu.com/s/1ie3kJaOLkNzwPIvjq0tH6A?pwd=ura9) 提取码: ura9
- Google Dive: [[Link]](https://drive.google.com/file/d/1oxGD4EDIGL_xX-jy9BAO9E_4nke_J2Wk/view?usp=sharing)
- Huggingface: [[Link]](https://huggingface.co/datasets/zmling/KAD_Datasets)

## Citation
If you find our paper and code useful in your research, please consider giving a star ⭐ and citation 📝 (´▽`ʃ♡ƪ)
```
@article{zhai2024world,
  title={World knowledge-enhanced Reasoning Using Instruction-guided Interactor in Autonomous Driving},
  author={Zhai, Mingliang and Li, Cheng and Guo, Zengyuan and Yang, Ningrui and Qin, Xiameng and Wu, Yuwei and Zhao, Sanyuan and Han, Junyu and Tao, Ji and Jia, Yunde},
  journal={arXiv preprint arXiv:2412.06324},
  year={2024}
}
```


## Acknowledgements
This work was supported by the Natural Science Foundation of Shenzhen under Grant No. JCYJ20230807142703006, Natural Science Foundation of China (NSFC) under Grants No. 62176021 and No. 62172041, and Key Research Platforms and Projects of the Guangdong Provincial Department of Education under Grant No.2023ZDZX1034.

Our project is built upon [LLaVA](https://github.com/haotian-liu/LLaVA) and [Bunny-Llama](https://github.com/BAAI-DCAI/Bunny), leveraging their robust codebases and the exceptional language capabilities of base model.