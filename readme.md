
# AutoUX

This project uses architectures from pix2code and image style transfer.

pix2code is a network introduced by Tony Beltramelli which transforms image designs for different UIs into their respective DSL code. 

The original paper found [here](https://arxiv.org/abs/1705.07962) and the original demo code found [here](https://github.com/tonybeltramelli/pix2code)

pix2code2 is an attempt to improve that network through the use of autoencoders which is borrowed for the pre-trained weight and the code can be found here https://github.com/fjbriones/pix2code2

A tensorflow implementation of style transfer (neural style) described in the papers:
* [A Neural Algorithm of Artistic Style](https://arxiv.org/pdf/1508.06576v2.pdf) : *submitted version*
* [Image Style Transfer Using Convolutional Neural Networks](http://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Gatys_Image_Style_Transfer_CVPR_2016_paper.pdf) : *published version*  
by Leon A. Gatys, Alexander S. Ecker, Matthias Bethge

An implementation of the code was borrowed from https://github.com/hwalsuklee/tensorflow-style-transfer/

## Usage for Image Style Transfer

### Prerequisites
1. Tensorflow
2. Python packages : numpy, scipy, PIL(or Pillow), matplotlib
3. Pretrained VGG19 file : [imagenet-vgg-verydeep-19.mat](http://www.vlfeat.org/matconvnet/models/imagenet-vgg-verydeep-19.mat)

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;* Please download the file from link above.  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;* Save the file under `pre_trained_model`

### Running
```
python run_main.py --content <content file> --style <style file> --output <output file>
```
*Example*:
`python run_main.py --content images/tubingen.jpg --style images/starry-night.jpg --output result.jpg`

#### Arguments
*Required* :  
* `--content`: Filename of the content image. *Default*: `images/tubingen.jpg`
* `--style`: Filename of the style image. *Default*: `images/starry-night.jpg`
* `--output`: Filename of the output image. *Default*: `result.jpg`  

*Optional* :  
* `--model_path`: Relative or absolute directory path to pre trained model. *Default*: `pre_trained_model`
* `--loss_ratio`: Weight of content-loss relative to style-loss. Alpha over beta in the paper. *Default*: `1e-3`
* `--content_layers`: *Space-separated* VGG-19 layer names used for content loss computation. *Default*: `conv4_2`
* `--style_layers`: *Space-separated* VGG-19 layer names used for style loss computation. *Default*: `relu1_1 relu2_1 relu3_1 relu4_1 relu5_1`
* `--content_layer_weights`: *Space-separated* weights of each content layer to the content loss. *Default*: `1.0`
* `--style_layer_weights`: *Space-separated* weights of each style layer to loss. *Default*: `0.2 0.2 0.2 0.2 0.2`
* `--max_size`: Maximum width or height of the input images. *Default*: `512`
* `--num_iter`: The number of iterations to run. *Default*: `1000`
* `--initial_type`: The initial image for optimization. (notation in the paper : x) *Choices*: content, style, random. *Default*: `'content'`
* `--content_loss_norm_type`: Different types of normalization for content loss. *Choices*: [1](https://arxiv.org/pdf/1508.06576v2.pdf), [2](https://arxiv.org/abs/1604.08610), [3](https://github.com/cysmith/neural-style-tf). *Default*: `3`

## Pix2code Code:
Parts of the code here are made by Tony Beltramelli and is subject to its corresponding licenses, see LINCENSE_pix2code file for more information. Some are modified such as to fit the proposed pixc2code2 network. The datasets are also from the original pix2code network.

So far, the network has been trained on the web-dataset and weights are included in this repository

## Usage of Pix2code2:

Prepare the data:
```sh
# reassemble and unzip the data
cd datasets
zip -F pix2code_datasets.zip --out datasets.zip
unzip datasets.zip

cd ../model

# split training set and evaluation set while ensuring no training example in the evaluation set
# usage: build_datasets.py <input path> <distribution (default: 6)>
./build_datasets.py ../datasets/web/all_data

# transform images (normalized pixel values and resized pictures) in training dataset to numpy arrays (smaller files if you need to upload the set to train your model in the cloud)
# usage: convert_imgs_to_arrays.py <input path> <output path>
./convert_imgs_to_arrays.py ../datasets/web/training_set ../datasets/web/training_features
```
Train the model:
```sh
cd model

# provide input path to training data and output path to save trained model and metadata
# usage: train.py <input path> <output path> <train_autoencoder (default: 0)>
./train.py ../datasets/web/training_set ../bin

# train on images pre-processed as arrays
./train.py ../datasets/web/training_features ../bin

# train with autoencoder
./train.py ../datasets/web/training_features ../bin 1
```

Generate code for batch of GUIs in web evaluation set, if you want to use pretrained weights, unzip this [file](https://1drv.ms/u/s!Ao8Y5FscWK9imo0GtV5u3sXOr6sc_A) and copy the pix2code2.h5 file to the bin folder:
```sh
mkdir code
cd model

# generate DSL code (.gui file), the default search method is greedy
# usage: generate.py <trained weights path> <trained model name> <input image> <output path> <search method (default: greedy)>
./generate.py ../bin pix2code2 ../datasets/web/eval_set ../code

# equivalent to command above
./generate.py ../bin pix2code2 ../datasets/web/eval_set ../code greedy

# generate DSL code with beam search and a beam width of size 3
./generate.py ../bin pix2code2 ../datasets/web/eval_set ../code 3
```

Generate code for a single GUI image:
```sh
mkdir code
cd model

# generate DSL code (.gui file), the default search method is greedy
# usage: sample.py <trained weights path> <trained model name> <input image> <output path> <search method (default: greedy)>
./sample.py ../bin pix2code2 ../test_gui.png ../code

# equivalent to command above
./sample.py ../bin pix2code2 ../test_gui.png ../code greedy

# generate DSL code with beam search and a beam width of size 3
./sample.py ../bin pix2code2 ../test_gui.png ../code 3
```

Compile generated code to target language:
```sh
cd compiler

# compile .gui file to HTML/CSS (Bootstrap style)
./web-compiler.py <input file path>.gui
```
Wireframes and Color palettes used for testing is under '/images' folder.
