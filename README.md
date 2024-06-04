## HRFAE Variation: Dispatch GAN + ResNet 50


This implementation of the High Resolution Face Age Editing (HRFAE) model combines the dispatchGAN architecture and Res Net 50 as the age classifier. The HRFAE model is designed for advanced face age editing, providing high-quality results by leveraging the strengths of each component.

- dispatchGAN: This generative adversarial network is optimized for dispatching high-resolution image generation tasks.
- ResNet50: Residual Network is known for its skip connections, allowing convolutional networks to be able to learn without gradient vanishing at extreme depth, and thus increased the accuracy of the network.


Official implementation for paper [High Resolution Face Age Editing](https://arxiv.org/pdf/2005.04410.pdf).

## Load and test pretrained network 

1. You can download the working notebook `dl_final.ipynb  where everything is set up.

2. Upload test images in the folder `/test/input` and run all cells then it will test after training. The output images will be saved in the folder `/test/output`. You can change the desired target age with `--target_age`.

## License

Copyright © 2020, InterDigital R&D France. All rights reserved.

This source code is made available under the license found in the LICENSE.txt in the root directory of this source tree.




