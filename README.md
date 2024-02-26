DEF-GAN2: Deep Embedding Fusion-Generative Adversarial Networks
====================================== 
![Uploading A small bird with varying shades of brown and white under the eyes-dc.jpg…]()
[Uploading cosmos-apricot-lemonade-gap-1_b116772e-c341-44b9-b0dd-0b081c02f9f2.webp…]()
![pizza with eggs DEFGAN2](https://github.com/ALIMDYOUSUF/DEF-GAN2/assets/91628312/750cb7bb-1111-43ce-a53b-ef69fe005d86)


**Data**
https://www.kaggle.com/datasets/awsaf49/coco-2017-dataset
1. Download our preprocessed char-CNN-RNN text embeddings for [birds](https://drive.google.com/open?id=0B3y_msrWZaXLT1BZdVdycDY5TEE) and [flowers](https://drive.google.com/open?id=0B3y_msrWZaXLaUc0UXpmcnhaVmM) and save them to `Data/`.
  - [Optional] Follow the instructions [reedscot/icml2016](https://github.com/reedscot/icml2016) to download the pretrained char-CNN-RNN text encoders and extract text embeddings.
2. Download the [birds](http://www.vision.caltech.edu/visipedia/CUB-200-2011.html) and [flowers](http://www.robots.ox.ac.uk/~vgg/data/flowers/102/) image data. Extract them to `Data/birds/` and `Data/flowers/`, respectively.
3. Preprocess images.
  - For birds: `python misc/preprocess_birds.py`
  - For flowers: `python misc/preprocess_flowers.py`

**Training**
- The steps to train a DEF-GAN model on the CUB dataset using our preprocessed data for birds.
  - Initial: train initial GAN (e.g., for 500-600 epochs) `python initial/run_exp.py --cfg initial/cfg/birds.yml --gpu 0`
  - Final: train final GAN (e.g., for another 500-600 epochs) `python final/run_exp.py --cfg final/cfg/birds.yml --gpu 1`
- Change `birds.yml` to `flowers.yml` to train a DEF-GAN model on LSUN dataset using our preprocessed data for chourch and bedroom.
- `*.yml` files are example configuration files for training/testing our models.
- If you want to try your own datasets, [here](https://github.com/soumith/ganhacks) are some good tips about how to train GAN. Also, we encourage to try different hyper-parameters and architectures, especially for more complex datasets.

**Pretrained Model**
(https://drive.google.com/drive/folders/1eijllkMJTb9ENLWx-4J0YrUYm188sd9r)(Just used the same setting as the char-CNN-RNN. We assume better results can be achieved by playing with the hyper-parameters).
**5 Pretrained Models above this link**
- `Birds`
-  `Bedroom`
-  `Church`
-  `Cats`
-  `Dogs`
**Sampling**
Run python main.py --cfg cfg/eval_bird.yml --gpu 1 to generate examples from captions in files listed in "./data/birds/example_filenames.txt". 
Change the eval_*.yml files to generate images from other pre-trained models.
Input your own sentence in "./data/birds/example_captions.txt" if you wannt to generate images from customized sentences.

**Validation**
To generate images for all captions in the validation dataset, change B_VALIDATION to True in the eval_*.yml. and then run python main.py --cfg cfg/eval_bird.yml --gpu 1
We compute inception score for models trained on birds using the DpDCAE-GAN-inception-model.
We compute inception score for models trained on coco using improved-gan/inception_score.
### Dependencies
- `Windows 10 Pro`
- `Processor	Intel(R) Core(TM) i7-1035G1`
- `Installed RAM	32.00 GB`
- `SSD 1.00 GB`

python 3.10

[TensorFlow Latest version](https://www.tensorflow.org/get_started/os_setup)
[Torch 2](http://torch.ch/docs/getting-started.html#_) is needed, if use the pre-trained char-CNN-RNN text encoder.

In addition, please add the project folder to PYTHONPATH and `pip install` the following packages:
 - `A suitable conda environment named DEF can be created and activated with:`

 - `conda env create -f environment.yaml`
- `conda activate DEF`
- `pip install`
- `prettytensor`
- `progressbar`
- `python-dateutil`
- `easydict`
- `pandas`
- `torchfile`

# Summary

## Introduction

* This paper combines DEF into an unsupervised generative model that simultaneously learns to encode, generate and compare dataset samples.

* It shows that generative models trained with learned similarity measures produce better image samples than models trained with element-wise error measures.

* It demonstrates that unsupervised training results in a latent image representation with disentangled factors of variation (Bengio et al., 2013). This is illustrated in experiments on a dataset of face images labelled with visual attribute vectors, where it is shown that simple arithmetic applied in the learned latent space produces images that reflect changes in these attributes.

## Attribute Editor

A AE consists of two networks that autoencode data samples to a latent representation z and decode the latent representation back to data space, respectively.                                         
                                              
The AE regularizes the encoder by imposing a prior over the latent distribution p(z). Typically z ∼ N (0, I) is chosen. The VAE loss is minus the sum of the expected log-likelihood (the reconstruction error) and a prior regularization term.

## Generative Adversarial Network

A GAN consists of two networks: the generator network Gen(z) maps latents z to data space while the discriminator network assigns probability y = Dis(x) ∈ [0, 1] that x is an actual training sample and probability 1 − y that x is generated by our model through x = Gen(z) with z ∼ p(z). The GAN objective is to find the binary classifier that gives the best possible discrimination between true and generated data and simultaneously encouraging Gen to fit the true data distribution. We thus aim to maximize/minimize the binary cross entropy with respect to Dis / Gen with x being a training sample
and z ∼ p(z).


## Implementation and Model Architecture:

For all our experiments, we use convolutional architectures and use backward convolution (aka.fractional striding) with stride 2 to upscale images in Dec. Backward convolution is achieved by flipping the convolution direction such that striding causes upsampling. Our models are trained with RMSProp using a learning rate of 0.0003 and a batch size of 64.

**Reference**
- `StackGAN++: Realistic Image Synthesis with Stacked Generative Adversarial Networks`
- `Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks`
- `Generative Adversarial Text to Image Synthesis`
- `AttnGAN: Fine-Grained Text to Image Generation with Attentional Generative Adversarial Networks`


Train a new model
-----------------
1. Construct dataset following [origin guide](https://github.com/olofmogren/c-rnn-gan) for CRNN and Dc (https://github.com/ALIMDYOUSUF/DEF-GAN). If you want to train with variable length images (keep the origin ratio for example), please modify the `tool/create_dataset.py` and sort the image according to the text length.
2. Execute ``python train.py --adadelta --trainRoot {train_path} --valRoot {val_path} --cuda``. Explore ``train.py`` for details.

