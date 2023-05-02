# GigaGAN-Implementation, a 16-726 project

<img src="./gigagan-generator.png" height=190 alt="GigaGAN Generator" /><img src="./gigagan-discriminator.png" height=190 alt="GigaGAN Discriminator" />

## GigaGAN
Implementation of [GigaGAN: Scaling up GANs for Text-to-Image Synthesis](https://arxiv.org/pdf/2303.05511).


## Install
```shell
pip install transformers==4.27.4 datasets
pip install -r requirements.txt
```

## Colab
or check our Colab for a quick start.
- https://github.com/linjiw/GigaGAN-Implementation/blob/main/16726_GigaGANipynb.ipynb

## References
- https://github.com/rosinality/stylegan2-pytorch

## Implementation Tasks
- [x] Unconditional
- [x] CLIP loss
- [x] Matching-aware loss
- [x] Text Conditional
- [ ] Muti-Scale
