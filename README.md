# UACL
The repository contains the implementations for "**Uncertainty-Aware Contrastive Learning for Semi-Supervised Classification of Multimodal Remote Sensing Images**".You can find [the PDF of this paper](https://ieeexplore.ieee.org/document/10540387).
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
python Model/demo.py
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
@ARTICLE{10540387,
  author={Ding, Kexin and Lu, Ting and Li, Shutao},
  journal={IEEE Transactions on Geoscience and Remote Sensing}, 
  title={Uncertainty-Aware Contrastive Learning for Semi-Supervised Classification of Multimodal Remote Sensing Images}, 
  year={2024},
  volume={62},
  number={},
  pages={1-13},
  doi={10.1109/TGRS.2024.3406690}}
```
****
# Contact
Kexin Ding: [dingkexin@hnu.edu.cn](dingkexin@hnu.edu.cn)
