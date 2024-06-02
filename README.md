# AICUP2024 Project Repository

This repository is a revised version of [BASnet](https://github.com/xuebinqin/BASNet), a Boundary-Aware Salient Object Detection Network. The original BASnet implementation has been modified to enhance performance, add new features, and improve usability.

## Introduction

This repository contains the related code used for our 2024 AICUP implementation. We have augmented the dataset significantly (such as horizontal and vertical flipping, and cropping). When training the model, we trained the lane and river models separately, so we have two trained models. Finally, we trained multiple models and used an ensemble method to increase our accuracy.

### Instructions for Using This Code

To use this code, you must first place the model weights and dataset into the specified folders. Since we have done a lot of processing, we will directly provide all the processed images on the cloud for the judges to use, so there is no need to re-execute the processing steps (more detailed instructions will follow).

### Training and Prediction Workflow

To increase the model accuracy, we first filled the original labels with solid colors. Therefore, the final predicted results are also solid colors. We then use boundary extraction methods to extract the boundaries. The execution sequence of the code is as follows:

1. Train the river and lane models separately.
2. Predict the images.
3. Extract the boundaries.
4. Repeat the model training with different parameters (steps 1 to 3), save the models, and then use the ensemble method to combine multiple trained results into the final answer.

### Important Notes

- Our parameters are modified within the code, so if the judges wish to retrain the models, they may need to manually change the hyperparameters in the training files.
- Since we have already included the `.pth` files, you only need to execute steps 2 to 4 to reproduce the results more quickly.

## Download and Installation

1. **Clone the repository:**

    ```sh
    git clone https://github.com/yourusername/your-repo.git
    cd your-repo
    ```

2. **Install the required dependencies:**

    ```sh
    pip install -r requirements.txt
    ```

## Dataset Preparation

To train and test the model, you need to put our dataset and pre-trained model file. Follow these steps:

1. **Download the dataset and model file:**
    - [Download the dataset and model pth file from Google Drive](https://drive.google.com/drive/folders/1Mxp4B1yuSctp7qlfUX5wPFpy57u8t_vQ).

2. **Organize the dataset:**
    Place the dataset in the `dataset/train` directories. The directory structure should look like this:
    ```
    your-repo/
    ├── dataset/
    │   ├── train/
    │   │   ├── img/
            ├── label_img/
    │   │   └── new_label_img/
    │   ├── test/
    │   │   
    │   ├── run.py
    ```

3. **Place the pre-trained model file:**
    Put the downloaded `.pth` file in the `saved_model` directory and there are about twenty pth files because we use ensemble approach in final step by [ensemble.py](/ensemble.py):

    ```
    your-repo/
    ├── saved_model/
    │   │── best_model1.pth
    │   │── best_model2.pth
    │   │── best_model3.pth
    │   │.
    │   │.
    │   │.
    │   │.
    ```

## Usage

### Training the Model

1. Train the river and lane models separately:

    ```sh
    python train_4river.py  # For training the river model
    python train_4road.py   # For training the road model
    ```

    Adjust the following parameters in the training scripts:
    - Line 125: Modify `epoch` and `batch_size`.
    - Line 169 (river) / 180 (road): Toggle loading of weights.
    - Line 173 (river) / 184 (road): Adjust `learning_rate` (lr) and `Tmax`.

2. Predict the images and extract boundaries:

    ```sh
    python predict.py && python boundary.py
    ```

3. The predicted results are saved in `your_repo/test/`.

4. Repeat model training with different parameters (steps 1 to 3), save the output photo and , and then use following method to combine multiple trained results into the final answer the final file is `your_repo/test/answer`.

    ```sh
    python ensemble.py
    ```

## Acknowledgements

This project is based on the original work of[BASnet](https://github.com/xuebinqin/BASNet) by Xuebin Qin et al. We would like to thank the authors for their pioneering work and making their code available.

## Citation

If you use this code in your research, please consider citing the original BASnet paper:

```bibtex
    @article{DBLP:journals/corr/abs-2101-04704,
    author    = {Xuebin Qin and
                Deng{-}Ping Fan and
                Chenyang Huang and
                Cyril Diagne and
                Zichen Zhang and
                Adri{\`{a}} Cabeza Sant'Anna and
                Albert Su{\`{a}}rez and
                Martin J{\"{a}}gersand and
                Ling Shao},
    title     = {Boundary-Aware Segmentation Network for Mobile and Web Applications},
    journal   = {CoRR},
    volume    = {abs/2101.04704},
    year      = {2021},
    url       = {https://arxiv.org/abs/2101.04704},
    archivePrefix = {arXiv},
    }
```
