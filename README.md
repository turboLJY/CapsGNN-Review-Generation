# CapsGNN-Review-Generation

This repository contains the source code for the CIKM 2020 paper "[Knowledge-Enhanced Personalized Review Generation with Capsule Graph Neural Network](https://arxiv.org/abs/2010.01480)".

# Datasets

Our datasets, including Amazon Book, Electronic and IMDB movie, can be downloaded through [this link](https://drive.google.com/drive/folders/1xvAkWs8JXKRigMH68mK2zbhoqzvfcvou?usp=sharing). The following table presents the statistics of our datasets after preprocessing.

| Datasets  | | Electronic | Book | Movie |
|:----:|:----|---------:|----:|----:|
|           | #Users     | 46,086 | 61,722 |  46,893  | 
|   Review  | #Items     | 11,216 | 19,129 |  21,023   |
|           | #Reviews   | 199,617 | 731,800 | 1,152,912 |
|           | #Entities  | 30,301  | 96,173 | 594,452 |
|   KG      | #Relations | 20      | 12 | 22 |
|           | #Triples   | 68,620  | 285,118 | 2,130,368 |

Note that, the #Entities includes the aligned entities and its one-hop entities (neighbors), the #Relations includes the forward and backward relations, and the #Triples removes the triples with ''name'' and ''description'' relations. The following table demonstrates the included forward relations.

| Datasets | Electronic |
|:----:|:---------|
|  Electronic  | computer_peripheral, product_line, category, manufacturer, ad_campaigns, digital_camera_manufacturer, developer, brand, supported_storage_types, industry  | 
| Book | genre, subject, author, edition, character, language |
| Movie | genre, actor, director, writer, producer, music, country, language |

# Citation

If this work is useful in your research, please cite our paper.

```
@inproceedings{junyi2020review,
  title={{K}nowledge-{E}nhanced {P}ersonalized {R}eview {G}eneration with {C}apsule {G}raph {N}eural {N}etwork},
  author={Junyi Li, Siqing Li, Wayne Xin Zhao, Gaole He, Zhicheng Wei, Nicholas Jing Yuan, Ji-Rong Wen},
  booktitle={CIKM},
  year={2020}
}
```
