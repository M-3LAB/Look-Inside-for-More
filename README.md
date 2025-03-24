# Look Inside for More: Internal Spatial Modality Perception for 3D Anomaly Detection
Hanzhe Liang, Guoyang Xie, Chengbin Hou, Bingshu Wang, Can Gao†, Jinbao Wang†
(† Corresponding authors)

# Overview
This is the Reproducible Realisation of the AAAI25 paper ["Look Inside for More: Internal Spatial Modality Perception for 3D Anomaly Detection"](https://arxiv.org/abs/2412.13461). Because of a server storage disaster, our initial version of the code was lost, but, thanks to the help of some researchers, we have reproduced an approximation of the code for this paper. If you have a better reproduction, please get in touch with us at 2023362051@email.szu.edu.cn.

# ISMP
![ISMP](./images/example.png)

## Before that, a few caveats:

Our code implementation is based on the Nips23 paper "Real3D-AD: A Dataset of Point Cloud Anomaly Detection" and we thank them for their work!

Similar to their work, our code is also stochastic, and the results in the paper are obtained by means of the mean. If there are some discrepancies between your implementation and the values in the paper, it may be due to randomness and we are working on addressing it. To try to be as consistent as possible, you can use either the RTX3090 (24GB) graphics card mentioned in the paper or the A100 (40GB) graphics card from when we declassied the code.
