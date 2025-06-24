# seg: tools for evaluating binary semantic image segmentation models

Features:
  - 5+1 frameworks:
    - [torchvision](https://pytorch.org/vision/main/models.html)
    - [segmentation_models.pytorch](https://github.com/qubvel-org/segmentation_models.pytorch)
    - [Mask2Former](https://github.com/facebookresearch/Mask2Former)
    - [segment-anything (SAM)](https://github.com/facebookresearch/segment-anything)
    - [MobileSAM](https://github.com/ChaoningZhang/MobileSAM)
    - [segmentation_of_roots_in_soil_with_unet](https://github.com/Abe404/segmentation_of_roots_in_soil_with_unet)
  - Automatic mixed precision training
  - On-the-fly loading images from disk for training
  - Auto-patchification and some augmentation
  - 876 LOC and a 6-line requirements.txt file
  - Almost 300 model \& backbone combinations

# Installation
Some of the included packages have different requirements. E.g. in an Ubuntu box,
one will need g++ and python3-dev.

**IMPORTANT** Some Mask2Former custom ops require torch == 2.5.1

We need some custom steps wrt detectron and mask2former
`./install.sh`

Download the pretrained models
`./download_pretrained.sh`

Download the datasets
`cd data; for f in *_import.py; do ./$f; done`

# Usage
The data folder should contain a "train" and "val" directory, each containing an
"annotations" and "images":
```
- data
  data/train
  data/train/annotations
  data/train/annotations/0000.png
  ...
  data/train
  data/train/images
  data/train/images/0000.png
  ...
  data/val
  data/val/annotations
  data/val/annotations/0000.png
  ...
  data/val
  data/val/images
  data/val/images/0000.png
  ...
```

The samples will be cropped to `-s,--size` if larger (default: 1024 x 1024):

Run an experiment using:
`./seg.py $data_path unet/vanilla -o output`

Then in the output directory we get:
- `results`: A text file with the epoch, step, seconds per epoch,
  and best f1 score achieved
- `*-img.png`: A sample image from the validation set
- `*-pred.png`: The prediction of the model for the sample image
- `*-true.png`: The ground truth for the segmentation
- `checkpoint_best.pth`: The best model

# Command-line options
```
options:
  -h, --help            show this help message and exit
  -C, --clip-gradients
  -P, --pretrained
  -a LEARNING_RATE, --learning-rate LEARNING_RATE
  -b BATCH_SIZE, --batch-size BATCH_SIZE
  -e EPOCHS, --epochs EPOCHS
  -f EVAL_FREQUENCY, --eval-frequency EVAL_FREQUENCY
  -j NUM_WORKERS, --num-workers NUM_WORKERS
  -m, --mixed-precision
  -o CHECKPOINT_DIR, --checkpoint-dir CHECKPOINT_DIR
  -s SHAPE SHAPE, --shape SHAPE SHAPE
  -w WARMUP_STEPS, --warmup-steps WARMUP_STEPS
```

# Developing
Every model file should define
 - A `models` dict with keys the name of each model
 - A `new(name, pretrained: Bool)` function that returns a model
 - A model that takes a `(N x 3 x H x W)` image and returns a
   `(N x 1 x H x W)` prediction

# Models
```
m2f/R101
m2f/R50
m2f/swin-base
m2f/swin-small
m2f/swin-tiny
mb_sam/vit_t
sam/vit-base
sam/vit-huge
sam/vit-large
segmentation_pytorch/deeplabv3+-ENCODER
segmentation_pytorch/deeplabv3-ENCODER
segmentation_pytorch/linknet-ENCODER
segmentation_pytorch/manet-ENCODER
segmentation_pytorch/pan-ENCODER
segmentation_pytorch/pspnet-ENCODER
segmentation_pytorch/segformer-ENCODER
segmentation_pytorch/unet++-ENCODER
segmentation_pytorch/unet-ENCODER

ENCODER:
        densenet{121,161,169,201}
        efficientnet-b{0..6}
        inceptionv4
        mobilenet_v2
        mobileone_s{0..4}
        resnet{18,50,101}
        timm-efficientnet-b{0..6}
        vgg{11,19}
        xception

torchvision_models/deeplabv3-mobilenet_v3_large
torchvision_models/deeplabv3-resnet101
torchvision_models/deeplabv3-resnet50
torchvision_models/fcn-resnet101
torchvision_models/fcn-resnet50
torchvision_models/lraspp-mobilenet_v3_large
unet/root_painter
unet/vanilla
```
