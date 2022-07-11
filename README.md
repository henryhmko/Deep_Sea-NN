# Deep Sea-NN
Enhancing underwater images using a truncated U-Net architecture for real-time color correction applications.

---

Truncated U-Net architecture inspired by [UWGAN: Underwater GAN for Real-world Underwater Color Restoration and Dehazing](https://arxiv.org/abs/1912.10269).

Usage of U-Net for image-to-image-translation inspired [Image-to-Image Translation with Conditional Adversarial Networks](https://arxiv.org/abs/1611.07004).


## Overview

I love diving and filming but my underwater photos come out with distorted colors(greenish, blueish, distorted due to refraction etc). Thanks to Deep Sea-NN I don't have to learn photoshop to edit my favorite underwater pics. YaAY!
- Focused on two factors:
  1) Training / Inference Speed ...since i don't have good gpus :'(
  2) Image Quality(especially crispness)

- Deep Sea-NN does NOT require the following:
  1) $$ GPUs to train & run
  2) Depth map for training images
  3) a lot of images to train(apparently)

## Background
> "Underwater environment...**distortion is extremely non-linear in nature, and is affected by a large number of factors, such as the amount of light present(overcast versus sunny, operational depth), amount of particles in the water, time of day, and the camera being used.**" from [Enhancing Underwater Imagery using Generative Adversarial Networks](https://arxiv.org/abs/1801.04011).

- Underwater image correction has been a task worked on for years especially with the coming of AUVs. Underwater images representing correct color schemes are necessary for effective object identification/segmentation tasks. [Traditional methods](https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.60.9572&rep=rep1&type=pdf) relied on physics or stats(tldr: used an energy minimization formulation based on Markov Random Fields to learn color correction patterns; pretty cool but looses a lot of detail since they divide single images into patches during training). 

- Even rather recent [models relied partly on physics](https://arxiv.org/abs/1702.07392#:~:text=Using%20WaterGAN%2C%20we%20generate%20a,correction%20of%20monocular%20underwater%20images.) to deal with attenuation, scattering, and color correction(tldr: complex and hard to generalize to different types of waters due to different lighting conditions for each ocean + requires depth map of training images).

- GAN based models are shown to be effective too, but didn't fit my goal of wanting to iterate through multiple tweaks of the same model doing ablation experiments. + I don't have effective GPUs to run through GAN training at such a fast pace like CNNs would. 

- I needed a model which had the potential of allowing real-time inference of underwater footages so it could be used within an underwater AUV system while performing at a good level for color correction.


## Dataset

Dataset from U of Minnesota's [EUVP(Enhancing Underwater Visual Perception) dataset](https://irvlab.cs.umn.edu/resources/euvp-dataset).
- 3700 Paired Underwater ImageNet + 1270 for validation
- 2185 Paired Underwater Scenes + 130 for validation
- Total: 5885 Paired Underwater Image Sets for training + 1400 for validation


## Results

<p align="center">
  <img src="https://github.com/henryhmko/Deep_Sea-NN/blob/main/result_imgs/results_smol.png" width="640"/>
</p>

### More results shown below. All images are of size 512x512.

![large many](result_imgs/results_many.png)

### Comparison with U of Minnesota's [UGAN](https://arxiv.org/abs/1801.04011).
<p align="center">
  <img src="https://github.com/henryhmko/Deep_Sea-NN/blob/main/result_imgs/ugan_comparison.png" width="740"/>
</p>

- UGAN outputs tend to lose image detail upon correction and include checkerboard artifacts in its background. This could be a problem for tasks such as color correction for marine biological research where exact details of an organism(e.g. size of spots on the porcupine puffer_row1) must be preserved.
- This is not the case with Deep Sea-NN where original content detail is preserved while correcting distorted colors.

# Extra stories

## Truncated U-Net architecture

- Inspired by [Shallow UWnet's](https://arxiv.org/abs/2101.02073) unbelievably simple(yet really good results considering its simplicity) architecture of just 3 densely connected convolutional blocks with skip connections, I decided to stick with CNNs with skip layers. It's just that...
  1) Shallow UWnet created blurry outputs and lost some details when testing. They did use the VGG perceptual loss + MSE loss, which has been said to create blotchy results by the [NVIDIA/MIT MEDIA LAB paper](https://arxiv.org/pdf/1511.08861.pdf).
  2) Did not work well with resized images of larger sizes than 256x256(size it was trained upon), therefore making it unappealing for people like me who want personal semi-quality diving photos(well inference image size should at least be over 256x256..it's 2022...). Maybe this was due to a rather shallow model architecture not being able to capture many color features patterns to generalize to larger input image sizes.
  - This led me to look for a CNN architecture which has skip connections(for content preservation) and a deep architecture(for generalization), while capturing features well...which naturally led to an U-Net. Its flexibiility in working with various input image sizes during inference was a big plus.
  
- Realized I didn't need the traditionally deep U-Net since even the Shallow-UWnet performs so well, so I took out one max-pool-depth layer and worked with a shallower U-Net to accelerate training time.


## Choice of Loss function: MS-SSIM + L1
### VGG+MSE vs MS-SSIM+L1 (ABLATION STUDY)
<p align="center">
  <img src="https://github.com/henryhmko/Deep_Sea-NN/blob/main/result_imgs/loss_comparison.png" width="740"/>
</p>

- MS-SSIM+L1(right) captures fine-grain details better than the VGG+MSE(left).
  1) MS-SSIM show distinct anemone tentacles(yellow box) while vgg+mse show a blob
  2) Effectiveness of MS-SSIM in darker regions(green box) is less apparent, but still looks a bit crispier(would have to compare SSIM, PSNR, or [UIQM metrics](https://ieeexplore.ieee.org/document/7305804)(for images with no ground truth=>i.e. images that I took on my own) for exact quantitative results..which i am too lazy to do for this project... 

## Deeper Dive into Loss Function Choice: optimal gaussian sigma values

- There's two choices of loss functions when it comes to the SSIM family: the classic SSIM or the beefed-up MS-SSIM.
  1) SSIM loss functions are sensitive to the specific gaussian sigma value it uses. A smaller gaussian sigma value works well with correcting edges in an image while leaving splotchy artifacts in flat areas while a higher gaussian sigma value does the exact opposite(see results below from [this paper](https://arxiv.org/pdf/1511.08861.pdf)). 
  
<p align="center">
  <img src="https://github.com/henryhmko/Deep_Sea-NN/blob/main/result_imgs/gaussian_sigmas.png" width="740"/>
</p>

  2) Enter the beefed up MS-SSIM. Instead of going through the work of fine-tuning your gaussian sigma values, MS-SSIM calculates SSIM scores on difference scales on the image and works with multiple gaussian sigma values. The authors of the original paper set gaussian sigmas to be [0.5, 1, 2, 4, 8].


### To be SSIM or MS-SSIM? 
MS-SSIM will outperform SSIM in most common scenarios, but the paper also mentions...
> "Thanks to its multi-scale nature, MS-SSIM solves the issue of noise around edges, but does not solve the problem of the change colors in flat areas, in particular when at least one channel is strong, as is the case with the sky." - from [Loss Functions for Image Restoration with Neural Networks](https://arxiv.org/pdf/1511.08861.pdf)

MS-SSIM is, after all, a better generalization of SSIM where we can avoid the hassle of finding the optimal gaussian sigma value, but it is only a convenient generalization.

- BUT *what if we know the exact distribution of the images our model will see when it is deployed?* (in other words, what if we know whether if our model will see images consisted mostly of edges instead of flat areas or vice versa?)

I believe we can put an answer to this question when working with underwater environments.

### Special Case for the Underwater Environment

Unlike the surface where a picture of a street will include both edges(pedestrians, signs, etc) and flat areas(open skies), many under

- If, for example, a model is aimed to take a lot of pictures of marine organisms then it will process images that are bound to include a lot of edges. Those edges may be the fins of a particular fish, tentacles of an anemone, geometric patterns for camouflage, or such. With such a distribution, we could directly train the model with a SSIM loss using a small gaussian sigma.

- On the contrary, the opposite example would be a model exposed to images which are mostly filled with flat areas. In the underwater environment, this would be open water images(like the one below-it's me in jeju yeee) where images are ~80% filled with different shades of a single color. For this case, it would be appropriate to train the model with a SSIM loss using a high gaussian sigma.
  - A potential application for this kind of model could be an AUV following a human diver and, thus the need to consistently process images like the one below to track the diver's current location.

<p align="center">
  <img src="https://github.com/henryhmko/Deep_Sea-NN/blob/main/result_imgs/noice_diver.JPG" width="740"/>
</p>

This method would be a good fine-tuning practice to specifically tailor your model to the data it will mostly see upon deployment.




