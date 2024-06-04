## HRFAE Variation: Dispatch GAN + VGG + AdaIN

This implementation of the High Resolution Face Age Editing (HRFAE) model uses a combination of Dispatch GAN, VGG16 and AdaIN  to perform high-quality face age editing. This variation leverages the capabilities of both models to achieve detailed and realistic age transformations.

- dispatchGAN: A specialized generative adversarial network designed for high-resolution image generation tasks, ensuring detailed and high-quality output.
- VGG16: A renowned convolutional neural network architecture used for its powerful feature extraction capabilities, enhancing the model's understanding and manipulation of facial features for accurate age modification.
- AdaIN: A simple yet effective approach that for the first time enables arbitrary style transfer in real-time. At the heart of our method is a novel adaptive instance normalization (AdaIN) layer that aligns the mean and variance of the content features with those of the style features. 

Official implementation for paper [High Resolution Face Age Editing](https://arxiv.org/pdf/2005.04410.pdf).

## Load and test pretrained network 

1. You can download the working notebook `dl_final.ipynb` where everything is set up.

2. Upload test images in the folder `/test/input` and run all cells then it will test after training. The output images will be saved in the folder `/test/output`. You can change the desired target age with `--target_age`.

## License

Copyright © 2020, InterDigital R&D France. All rights reserved.

This source code is made available under the license found in the LICENSE.txt in the root directory of this source tree.




