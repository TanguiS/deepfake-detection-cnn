# Deepfake Detection project based on CNNs.

## Getting Started

### Prerequisites

- Install [conda](https://conda.io/projects/conda/en/latest/user-guide/install/index.html)
- Environment creation :

```bash
$ conda create -n dd-cnn python=3.8
$ conda activate dd-cnn
$ pip install -r requirments.txt
```

- **Warning**: You should install *TensorFlow* manually to install the proper binary file for your system and then the requirements. Ex :
- **ROCM**: ([compatibility](https://github.com/ROCmSoftwarePlatform/tensorflow-upstream/blob/develop-upstream/rocm_docs/tensorflow-rocm-release.md))
```bash
# My version: 
$ pip install tensorflow-rocm==2.11.0.540
```
- **CUDA**: ([tensorflow-official](https://www.tensorflow.org/install/pip?hl=fr))
```bash
$ conda install -c conda-forge cudatoolkit=11.8.0
$ python3 -m pip install nvidia-cudnn-cu11==8.6.0.163 tensorflow==2.11.*
```

- Create or Download Datasets: No datasets are provided with this project, here are the prerequisites to have a usable dataset with this project:
  - Images / Frames should be in a *root* directory (no specified tree) and the faces must be pre-extracted, [my-extraction-method](./bench/extract/face) using YuNet program.
  - The frames referencing must be made by a ***dataframe*** with pandas' pickle ext format.
  - **dataframe** mandatory content: 
    - Relative path to the frames in relation to the *root* directory as the index of the dataframe.
    - *label* column with **True** or **False** values respectively for **Fake** and **Real** frames.

### Available Architecture / Net :

- ***Xception***
- ***EfficientNetB4***
- ***ResNet152V2***

You can easily add a new *Keras* model by copying the code in the [models](./models) folder, for instance: [Xception](./models/Model_Xception/Model.py)

### Training Run Command Example on EfficientNetB4:

```bash
$ conda activate dd-cnn

# Create a save directory for models and history:
$ mkdir ./log

# Specify the necessaries path:
$ ROOT_DATASET_PATH=/path/to/your/dataset/folder
$ DF_DATASET_PATH=/path/to/your/dataframe.pkl

# Args:
$ python main.py train -h

# If using ROCM and get an error with tensorflow (current GPU is ignored) a possible solution:
$ HSA_OVERRIDE_GFX_VERSION=10.3.0

# Run command:
$ python main.py train --arch "EfficientNetB4" \
-root $ROOT_DATASET_PATH \
-df $DF_DATASET_PATH \
-o "./log" \
-s 41 \
-d "70-20-10" \
--shape 256 \
-epoch 100 \
-b 8
```

### Plot Training History Run Command Example on EfficientNetB4

```bash
$ conda activate dd-cnn

# Args:
$ python main.py plot -h

# Path to the '.log' directory previously created in the above example
$ MODELS_LOG_DIR=/path/to/your/models/dir

$ python main.py plot -root "./log" --arch "EfficientNetB4" -m "latest"
```

Good to know: It does not save the graphs.

### Evaluation Run Command Example on EfficientNetB4

```bash
$ conda activate dd-cnn

# Args:
$ python main.py eval -h

# Specify the necessaries path:
# Path to the '.log' directory previously created in the above example
$ MODELS_LOG_DIR=/path/to/your/models/dir
$ ROOT_DATASET_PATH=/path/to/your/dataset/folder
$ DF_DATASET_PATH=/path/to/your/dataframe.pkl
# Path to the YuNet face extraction model (This is the algorithm I used to extract the faces, feel free to change)
$ YUNET_MODEL_PATH=/path/to/your/yunet_model.onnx

$ python main.py eval --shape 256 \
-r $ROOT_DATASET_PATH \
-df $DF_DATASET_PATH \
-o $MODELS_LOG_DIR \
--arch "EfficientNetB4" \
--seed 41 \
-d "70-20-10" \
-yunet $YUNET_MODEL_PATH
```

### Results Example on EfficientNetB4:

<p align='center'>
  <img src='./resources/graphs/training_history_latest.png' alt="Training History Graph On EfficientNetB4" width="500"/>
  <img src='./resources/graphs/val_history_latest.png' alt="Training History Graph On EfficientNetB4" width="500"/>
  <img src='./resources/graphs/confusion_matrix_latest.png' alt="Training History Graph On EfficientNetB4" height="500"/>
</p>


## Credits

- Tangui Steimetz, [ENSICAEN](https://www.ensicaen.fr/).
- Guoqiang Li, [MOBAI](https://www.mobai.bio/) - [NTNU](https://www.ntnu.edu/).