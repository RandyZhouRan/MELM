# MELM
Code for "MELM: Data Augmentation with Masked Entity Language Modeling for Low-Resource NER" - https://arxiv.org/abs/2108.13655

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
@article{zhou2021melm,
  title={MELM: Data Augmentation with Masked Entity Language Modeling for Cross-lingual NER},
  author={Zhou, Ran and He, Ruidan and Li, Xin and Bing, Lidong and Cambria, Erik and Si, Luo and Miao, Chunyan},
  journal={arXiv preprint arXiv:2108.13655},
  year={2021}
}
```
