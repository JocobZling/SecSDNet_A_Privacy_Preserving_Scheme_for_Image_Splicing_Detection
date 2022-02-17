# SecSDNet: A Privacy-Preserving Scheme for Image Splicing Detection  
### Overview

Third-party services are usually used for deep learning based digital image splicing detection but the input data may contain private and sensitive information, e.g. faces, that must be protected. This paper proposes a secure image splicing detection network called SecSDNet supporting privacy protection. The SecSDNet uses additive secret sharing based on our designed secure interactive protocols and the inference of our improved pre-trained plaintext EfficientNet model. Specifically, on the one hand, an adaptive residual module (ARM) and a Squeeze and Excitation (SE) are incorporated into the EfficientNet backbone to learn residual features adaptively and reduce the redundancy among channels after the ARM; on the other hand, a series of secure interactive protocols are designed to support the complex operations of deep neural networks between two parties, such as secure sigmoid (SecSigmoid), ReLU (SecReLU), SiLU (SecSiLU), ARM (SecARM), and SE (SecSE) protocols, with a few rounds and a small total amount of communication. The SecSDNet can detect image splicing forgery securely. Theoretical analysis of the security and communication complexity of the proposed  SecSDNet  protocols is provided. The experimental results on four publicly available datasets demonstrate that the proposed SecSDNet can securely detect image splicing forgeries with a similar accuracy as the improved plaintext EfficientNet model.

### Prerequisites

- Ubuntu 18.04 and Windows10
- NVIDIA GPU+CUDA CuDNN (CPU mode may also work, but untested)
- Install Torch1.8 and dependencies

### Training and Test Details

- When you train or test improved plaintext EfficientNet model
  - Change the `Plaintext/train.py`;`Plaintext/test.py`
  - Run `train.py` or `test.py` directly
- When you test the SecSDNet 
  - Please use Windows10 environment (Ubuntu may also work, but untested)
  - Change the model position in `Ciphertext/main.py`
  - Run `main.py`

### Related Works

- Z. Xia, Q. Gu, W. Zhou, L. Xiong, J. Weng, and N. Xiong, "STR: Secure computation on additive shares using the share-transform-reveal strategy," *IEEE Transactions on Computers,* 2021.

