# [WACV2021] PDAN
Implementation for the paper ["PDAN: Pyramid Dilated Attention Network for Action Detection"](https://openaccess.thecvf.com/content/WACV2021/html/Dai_PDAN_Pyramid_Dilated_Attention_Network_for_Action_Detection_WACV_2021_paper.html).

The code is tested in Python3.7 + PyTorch1.2 environment with one Tesla V100 GPU and the overall code framework is adapted from the [Superevent](https://github.com/piergiaj/super-events-cvpr18).

## Toyota Smarthome Untrimmed
The evaluation code and pre-trained model for PDAN on [Toyota Smarthome Untrimmed (TSU) dataset](https://project.inria.fr/toyotasmarthome/) can be found in this [repository](https://github.com/dairui01/Toyota_Smarthome/blob/main/pipline/). With the pretrained model, the mAP should be more than 32.7 % for f-map CS protocol. 

## Charades
For training and testing this code on **Charades**, please download the dataset from this [link](https://prior.allenai.org/projects/charades) and follow this [repository](https://github.com/piergiaj/pytorch-i3d) to extract the snippet-level I3D feature. The RGB-Pretrained PDAN can be downloaded via this [Link](https://mybox.inria.fr/f/9fa53012b2684cb588b5/?dl=1). If the I3D feature is well extracted, the pretrained RGB model should achieve ~ 23.8% per frame-mAP on Charades. Note that this mAP  is the one reported in the paper which is computed by all the timesteps and not the weighted mAP. While using the original setting (25 sampled version, w/o weighted) for Charades localization, the mAP should be more than 24 %. 

If you find this work useful for your research, please cite our [paper](https://openaccess.thecvf.com/content/WACV2021/html/Dai_PDAN_Pyramid_Dilated_Attention_Network_for_Action_Detection_WACV_2021_paper.html):
```bibtex
@inproceedings{dai2021pdan,
  title={Pdan: Pyramid dilated attention network for action detection},
  author={Dai, Rui and Das, Srijan and Minciullo, Luca and Garattoni, Lorenzo and Francesca, Gianpiero and Bremond, Francois},
  booktitle={Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision},
  pages={2970--2979},
  year={2021}
}
```
Contact: rui.dai@inria.fr

