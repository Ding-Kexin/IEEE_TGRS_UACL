# UACL
The repository contains the implementations for "**Cross-Scene Hyperspectral Image Classification With Consistency-Aware Customized Learning**".You can find [the PDF of this paper](https://ieeexplore.ieee.org/document/10659915).
![UACL](https://github.com/Ding-Kexin/UACL/blob/main/UACL_framework.jpg)
****
# Datasets
- [Houston](https://hyperspectral.ee.uh.edu/?page_id=459)
- [Trento](https://github.com/danfenghong/IEEE_GRSL_EndNet/blob/master/README.md)
- [MUUFL](https://github.com/GatorSense/MUUFLGulfport/)
- [Augsburg](https://github.com/danfenghong/ISPRS_S2FL/blob/main/README.md)
****
# Train UACL
``` 
python Model/demo_singleDA.py for single-modal cross-scene classification or Model/demo_multiDA.py for multi-modal cross-scene classification
``` 
****
# Results
| Dataset | OA (%) | AA (%) | Kappa (%) |
| :----: | :----: | :----: | :----: |
| Houston  | 95.37 | 95.99 | 95.00 |
| Trento  | 99.11 | 98.10 | 98.81 |
| MUUFL  | 88.29 | 89.11 | 84.78 |
| Augsburg  | 89.19 | 74.30 | 84.80 |
****
# Citation
If you find this paper useful, please cite:
```
@ARTICLE{10659915,
  author={Ding, Kexin and Lu, Ting and Fu, Wei and Fang, Leyuan},
  journal={IEEE Transactions on Circuits and Systems for Video Technology}, 
  title={Cross-Scene Hyperspectral Image Classification With Consistency-Aware Customized Learning}, 
  year={2024},
  volume={},
  number={},
  pages={1-1},
  doi={10.1109/TCSVT.2024.3452135}}
```
****
# Contact
Kexin Ding: [dingkexin@hnu.edu.cn](dingkexin@hnu.edu.cn)
