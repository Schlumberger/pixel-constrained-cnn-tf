# Probabilistic Semantic Inpainting with Pixel Constrained CNNs 

TensorFlow implementation of [Probabilistic Semantic Inpainting with Pixel Constrained CNNs](https://arxiv.org/abs/1810.03728) (2018).

This repo contains an implementation of Pixel Constrained CNN, a framework for performing probabilistic inpainting of images with arbitrary occlusions.

## Abstract
Semantic inpainting is the task of inferring missing pixels in an image given surrounding pixels and high level image semantics. Most semantic inpainting algorithms are deterministic: given an image with missing regions, a single inpainted image is generated. However, there are often several plausible inpaintings for a given missing region. In the [paper](https://arxiv.org/abs/1810.03728), we propose a method to perform probabilistic semantic inpainting by building a model, based on PixelCNNs, that learns a distribution of images conditioned on a subset of visible pixels.

## Network Architecture
Pixel Constrained CNN consists of a prior network (maksed convolutions as described in PixelCNNs) and a conditioning network (regular convolutions). During the training phase, the complete image is passed through the prior network and the masked image is passed through the conditioning network.

<img src="https://github.com/Schlumberger/ML_code_playground/blob/master/pixel_cnn_pattern_modeling/imgs/network_architecture.png" width='400'>

## Examples
#### CelebA 1-channel
<img src="https://github.com/Schlumberger/ML_code_playground/blob/master/pixel_cnn_pattern_modeling/imgs/bottom_mask_gif.gif" width='150'>
<img src="https://github.com/Schlumberger/ML_code_playground/blob/master/pixel_cnn_pattern_modeling/imgs/celeba_1_channel.png" width='400'>
<img src="https://github.com/Schlumberger/ML_code_playground/blob/master/pixel_cnn_pattern_modeling/imgs/likelihood_image.png" width='400'>

#### CelebA 3-channel
<img src="https://github.com/Schlumberger/ML_code_playground/blob/master/pixel_cnn_pattern_modeling/imgs/celeba_3_channel.png" width='400'>

## Running the code

### Training
Firstly, clone the repository and install all dependencies for the code.

#### In a directory of your choice, run the following
```
git clone https://github.com/Schlumberger/pixel-constrained-cnn-tf
cd pixel-constrained-cnn-tf
pip install -r requirements.txt # Install dependencies
```

#### Creating image_list file
Given a dataset, create the image_list file using `tools/create_img_lists.py`. For example, if you download the celebA dataset and and the images used for training are located in the 'data/celebA_train' folder, then the following command helps you to create a image_list file named 'train.txt'. 

```
python tools/create_img_lists.py --dataset=data/celebA_train --outfile=data/train.txt
```

A list of all the different training attributes, like learning rate, mask type and attributes, can be found in `tools/train.py`. For example, to train a pixel constrained CNN model on RGB celebA images(of size 32x32) conditioned on bottom masks with 8 visible rows, with a leaning rate of 4e-4 and 60 epochs, we run the following command:

```
python tools/train.py --img_list_path data/train.txt --mask_type bottom --mask_args 8 --learning_rate 4e-4 --num_epoch 60 --num_channels 3 --im_shape '32,32'  
```

### Inpainting 
To generate images using an already trained pixel constarined CNN model, use `tools/test.py`. For example, to generate 64 completions using a trained TensorFlow model located in `models/model-celeba` for images specified in the image_list located in `data/test.txt` by conditioning the images using random blobs, we run the following command:

```
python tools/test.py --imgs_list_path data/test.txt --model_name models/model-celeba --gen_type sample --num_samples 64 --mask_type random_blob 
```

A complete list of test options can be found in `tools/tets.py`. For example, there are 3 generation types to choose from while inpainting the images:
1. 'sample': Samples images for each masked image and plots each sampled image separately
2. 'uncertainty': Samples images for each masked image and plots the log-likelihood plot in descending order (most likely to least likely)
3. 'logits': (valid only for num_colors = 2) For each sampled image, plots the pixel probability progression one row at a time.

## Trained models
The trained models for inpainting 1-channel(black&white) celebA images can be downloaded from [here](https://drive.google.com/drive/folders/1YhijDv2PBN1DoyBEk2AMThqhcxO3r7zb?usp=sharing). The trained models for inpainting 3-channel(RGB) celebA images can be downloaded from [here](https://drive.google.com/drive/folders/1ItRGQMd8h0037mCvvMyvh72srwR06FA4?usp=sharing). 

## Citing
If you find this work useful in your research, please cite using:

```
@article{dupont2018probabilistic,
  title={Probabilistic Semantic Inpainting with Pixel Constrained CNNs},
  author={Dupont, Emilien and Suresha, Suhas},
  journal={arXiv preprint arXiv:1810.03728},
  year={2018}
}
```

## License

[Apache License 2.0](LICENSE)

