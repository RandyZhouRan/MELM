# MELM
Code for "MELM: Data Augmentation with Masked Entity Language Modeling for Low-Resource NER" - https://aclanthology.org/2022.acl-long.160.pdf

## Usage
* Train MELM on labeled NER data. You should place `train.txt`, `dev.txt` inside the `data` directory.
```
sh 01_train.sh
```

* Generate augmented data using trained MELM checkpoint
```
sh 02_generate.sh
```

## Citation
If you find this repository useful, please cite our paper
```
@inproceedings{zhou2022melm,
  title={MELM: Data Augmentation with Masked Entity Language Modeling for Low-Resource NER},
  author={Zhou, Ran and Li, Xin and He, Ruidan and Bing, Lidong and Cambria, Erik and Si, Luo and Miao, Chunyan},
  booktitle={Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)},
  pages={2251--2262},
  year={2022}
}

