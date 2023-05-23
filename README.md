|  Type  |                            Model                             | Version | Val acc <sup>1</sup> | Test acc <sup>2</sup> |
| :----: | :----------------------------------------------------------: | :-----: | :-----: | :------: |
| Frames | MobileNet [[ckpt](https://github.com/pau-fabregat/quantization/blob/main/qat_checkpoints/qat_mobilenet_v2.ckpt) \| [ptl](https://github.com/pau-fabregat/quantization/blob/main/deployed_models/mobilenetv2_456_256_bs32.ptl)] |   v2    |  92.7   |   93.8   |
|        | ResNet  [[ckpt](https://github.com/pau-fabregat/quantization/blob/main/qat_checkpoints/qat_resnet_18.ckpt) \| [ptl](https://github.com/pau-fabregat/quantization/blob/main/deployed_models/resnet18_456_256_bs32.ptl)] |   18    |  94.1   |   94.2   |
|        | ShuffleNet [[ckpt](https://github.com/pau-fabregat/quantization/blob/main/qat_checkpoints/qat_shufflenet_05.ckpt) \| [ptl](https://github.com/pau-fabregat/quantization/blob/main/deployed_models/shufflenet05_456_256_bs32.ptl)] |  0.5x   |  92.0   |   92.0   |
|        | ShuffleNet [[ckpt](https://github.com/pau-fabregat/quantization/blob/main/qat_checkpoints/qat_shufflenet_10.ckpt) \| [ptl](https://github.com/pau-fabregat/quantization/blob/main/deployed_models/shufflenet10_456_256_bs32.ptl)] |  1.0x   |  92.9   |   92.9   |
| Video  | X3D  [[ckpt](https://github.com/pau-fabregat/quantization/blob/main/qat_checkpoints/qat_x3d_xs.ckpt) \| ptl] |   xs    |  93.1   |    -     |
|        | X3D  [[ckpt](https://github.com/pau-fabregat/quantization/blob/main/qat_checkpoints/qat_x3d_s.ckpt) \| ptl] |    s    |  92.2   |    -     |
|        | MoviNet  [[ckpt](https://github.com/pau-fabregat/quantization/blob/main/qat_checkpoints/qat_movinet_a0.ckpt) \| [ptl](https://github.com/pau-fabregat/quantization/blob/main/deployed_models/movinet_a0_1s6f.ptl)] |   a0    |  87.2   |    -     |
|        | MoviNet  [[ckpt](https://github.com/pau-fabregat/quantization/blob/main/qat_checkpoints/qat_movinet_a1.ckpt) \| [ptl](https://github.com/pau-fabregat/quantization/blob/main/deployed_models/movinet_a1_1s6f.ptl)] |   a1    |  91.2   |    -     |

<sup>1</sup> See Weights&Biases report.
<sup>2</sup> See logs in the test_logs folder. Video models have not been tested as deployed versions seem not to work properly.
The number of necessary FLOPS can be registered with the FV-Core library ([see this](https://github.com/facebookresearch/fvcore/blob/main/docs/flop_count.md)).