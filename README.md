# Features
  - Only 629 LOC
  - On-the-fly loading images from disk for training
  - Auto-patchification and some augmentation

# Limitations
  - Only metric is IoU
  - Every model should output the same size as the input

# Installation
We need some custom steps wrt detectron and sam
`./install.sh`

Download the pretrained models
`./download_pretrained.sh`

Run the tests
`./make_test_images.sh TEST_DATA_DIR`
`pytest -v`

# Usage
`./seg.py $data_path unet/vanilla`

# Script options
models:
  - m2f/R50
  - m2f/R101
  - m2f/swin-base
  - m2f/swin-small
  - m2f/swin-tiny
  - sam/vit-base
  - sam/vit-large
  - segmentation\_pytorch/deeplabv3+-resnet50
  - segmentation\_pytorch/deeplabv3+-resnet101
  - segmentation\_pytorch/linknet-resnet50
  - segmentation\_pytorch/linknet-resnet101
  - segmentation\_pytorch/manet-resnet50
  - segmentation\_pytorch/manet-resnet101
  - segmentation\_pytorch/pan-resnet50
  - segmentation\_pytorch/pan-resnet101
  - segmentation\_pytorch/pspnet-resnet50
  - segmentation\_pytorch/pspnet-resnet101
  - segmentation\_pytorch/unet++-resnet50
  - segmentation\_pytorch/unet++-resnet101
  - torchvision\_models/deeplabv3-mobilenet\_v3\_large
  - torchvision\_models/deeplabv3-resnet50
  - torchvision\_models/deeplabv3-resnet101
  - torchvision\_models/fcn-resnet50
  - torchvision\_models/fcn-resnet101
  - torchvision\_models/lraspp-mobilenet\_v3\_large
  - unet/vanilla

```
usage: seg.py [-h] [-a LEARNING_RATE] [-b BATCH_SIZE] [-e EPOCHS] [-P]
              [-o CHECKPOINT_DIR] [-j NUM_WORKERS]
              data_path
              {MODEL}

positional arguments:
  data_path
  {MODEL}

options:
  -h, --help            show this help message and exit
  -a LEARNING_RATE, --learning-rate LEARNING_RATE
  -b BATCH_SIZE, --batch-size BATCH_SIZE
  -e EPOCHS, --epochs EPOCHS
  -P, --pretrained
  -o CHECKPOINT_DIR, --checkpoint-dir CHECKPOINT_DIR
  -j NUM_WORKERS, --num-workers NUM_WORKERS
```

# Developing
Every model file should define
 - A `models` dict with keys the name of each mode
 - A `new(name, pretrained: Bool)` function that returns a model
 - A model that takes a `(N x 3 x H x W)` image and returns a
   `(N x 1 x H x W)´ prediction
