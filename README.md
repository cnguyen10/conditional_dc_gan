## Deep convolutional generative adversarial networks in Jax

The repository provide an implementation of DC-GAN in Jax - a functional machine learning framework from Google. The implementation includes a vanilla DC-GAN and a class-conditional DC-GAN.

The data loading is based on the `mlx-data` provided by Apple. The tracking experiment is based on `mlflow`. Please see the `requirements.txt` for the full list of packages used in this experiment.

### Dataset structure

To simplify, the implementation does not require any specific folder structure of a dataset. Instead, a JSON-based file is provided to load each image. The JSON file specifies a list of objects (or Python dictionaries), each has two attributes:
- file: path to each image,
- label: the integer number indicates the class index of the corresponding image.
For example:
```json
[
    {
        "file": "path/to/datasets/img_align_celeba/124640.jpg",
        "label": 0
    },
    {
        "file": "/path/to/datasets/img_align_celeba/046612.jpg",
        "label": 0
    }
]
```

### Up and running

> **Image size**: the current implementation is to resize all images to (64, 64, 3). The generator and discriminator are also designed to handle such an image size. If one wants to work with a different image size, please change the `image_resize` when loading data and re-design the architectures of both the generator and discriminator to suit their need.

First, fire a command on the terminal to run the `mlflow` server:
```bash
bash mlflow_server.sh
```

Then, modify the `run.sh` to point to your targeted dataset, then run another command in a separated terminal:
```bash
bash run.sh
```