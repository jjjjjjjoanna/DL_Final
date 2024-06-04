# HRFAE Variation: dispatchGAN + VGG16 + BMILoss + CroppedDataset

This implementation of the High Resolution Face Age Editing (HRFAE) model combines the dispatchGAN architecture, the VGG16 network, and BMILoss with a cropped dataset. The HRFAE model is designed for advanced face age editing, providing high-quality results by leveraging the strengths of each component.

- dispatchGAN: This generative adversarial network is optimized for dispatching high-resolution image generation tasks.
- VGG16: This convolutional neural network architecture is used for feature extraction, enhancing the model's ability to understand and manipulate facial features.
- BMILoss: A custom loss function that helps in maintaining the balance and integrity of facial features during the age transformation process.
- CroppedDataset: Utilizing cropped images focuses the model on the essential parts of the face, leading to more accurate and visually pleasing age modifications.

Official implementation for paper [High Resolution Face Age Editing](https://arxiv.org/pdf/2005.04410.pdf).

## Load and test pretrained network 

1. You can download the working notebook `dl_final.ipynb  where everything is set up.

2. Upload test images in the folder `/test/input` and run all cells then it will test after training. The output images will be saved in the folder `/test/output`. You can change the desired target age with `--target_age`.

## License

Copyright © 2020, InterDigital R&D France. All rights reserved.

This source code is made available under the license found in the LICENSE.txt in the root directory of this source tree.




