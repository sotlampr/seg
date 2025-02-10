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
We need some custom steps wrt detectron and mask2former
`./install.sh`

Download the pretrained models
`./download_pretrained.sh`

Run the tests
`./make_test_images.sh TEST_DATA_DIR`
`pytest -v`

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
 - A `models` dict with keys the name of each mode
 - A `new(name, pretrained: Bool)` function that returns a model
 - A model that takes a `(N x 3 x H x W)` image and returns a
   `(N x 1 x H x W)´ prediction
