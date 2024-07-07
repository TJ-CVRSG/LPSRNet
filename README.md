# LPSRNet

This is the official implementation for "Style Reconstruction-Driven Networks for Occlusion-aware License Plate Recognition" Paper.


## Usage
### Installation
1. Clone this repository.
2. Install the required packages by running:
```bash
pip install -r requirements.txt
```

### Training
- Note: We provide the training code for the CCPD dataset. You can modify the code to train on other datasets.
- You can train the model with `CCPD_lite` which is a lite version of the CCPD dataset. The lite version contains 100 images for training and 50 images for validation. You can train the model by running the `train.py` script after modifying the `configs/train_config.yaml` file to set the correct paths for the dataset.
- To train the model on the full CCPD dataset, you need to follow these steps: 
1. Download the dataset and put it in the `data` folder.
2. Run the `tools/crop_ccpd.py` script to crop the license plates from the CCPD dataset. We will use the cropped license plates for training. Other datasets can be processed in a similar way.
3. Run the `tools/generator/generate_opencv_syn.py` script to generate synthetic images using OpenCV. We will use the generated images for training.
4. Modify the `configs/train_config.yaml` file to set the correct paths for the dataset.
5. Run the following command to train the model:
```bash
python train.py
```
6. (optional) You can use `wandb` to log the training process. You need to set the `wandb` key in the `configs/train_config.yaml` file to enable logging. Or you can use the following command to run the training with `wandb`:
```bash
python train.py wandb.enabled=True wandb.project=project_name wandb.name=run_name
```

### Inference
- You can use the trained model to recognize license plates by running the `test.py` script.
- We provide a pre-trained model that you can use to recognize license plates. `weight/best_valacc_0.9940.pth` is the pre-trained model.

### Evaluation
- You can evaluate the model by running the `evaluate.py` script.


## CBLP Dataset
Due to the privacy issues, we cannot provide the CBLP dataset publicly. However, you can send an email to Dr.Zhang (zhangshaoming@tongji.edu.cn) to request the dataset. We will provide the dataset for research purposes only. You should not use the dataset for commercial purposes.

## Citation
If you find this work useful for your research, please cite our paper:
```
@ARTICLE{10579856,
  author={Liu, Weijia and Zhang, Shaoming and Tang, Yan and Wang, Zhong and Wang, Jianmei},
  journal={IEEE Transactions on Circuits and Systems for Video Technology}, 
  title={Style Reconstruction-Driven Networks for Occlusion-aware License Plate Recognition}, 
  year={2024},
  volume={},
  number={},
  pages={1-1},
  keywords={License plate recognition;Image segmentation;Accuracy;Image reconstruction;Character recognition;Standards;Data models;License plate recognition;style transfer;image generation},
  doi={10.1109/TCSVT.2024.3421559}}
```

## Acknowledgment
- The code for generating synthetic license plates is based on the [chinese_license_plate_generator](https://github.com/Pengfei8324/chinese_license_plate_generator). We would like to thank the authors for their great work.
- The code for augmenting the dataset is based on the [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR). We would like to thank the authors for their great work.


## Contribution
If you have any ideas or suggestions, feel free to submit them through Github issues or propose pull requests directly to us. We are looking forward to your contributions!
