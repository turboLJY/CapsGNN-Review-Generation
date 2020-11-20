# CapsGNN-Review-Generation

This repository contains the source code for the CIKM 2020 paper "[Knowledge-Enhanced Personalized Review Generation with Capsule Graph Neural Network](https://arxiv.org/abs/2010.01480)".

# Directory

- [Requirements](#Requirements)

# Requirements

- Python 3.6
- Pytorch 1.4
- torch-geometric 
- Anaconda3

# Datasets
Our review datasets, including Amazon Book, Electronic and IMDB movie, can be downloaded from [Amazon Review](http://jmcauley.ucsd.edu/data/amazon/links.html) and crawled from [IMDB](https://www.imdb.com/). In addition, the entity-item linkage can be accessed from [KB4Rec](https://github.com/RUCAIBox/KB4Rec). Note that, to crawling the IMDB review, we first take the linked items in KB4Rec as spider seeds, and then crawl their corresponding review text.

Finally, the review text (labelled with entities) in three domains can be downloaded through [this link](https://drive.google.com/drive/folders/1xvAkWs8JXKRigMH68mK2zbhoqzvfcvou?usp=sharing). The following table presents the statistics of our datasets after preprocessing.

| Datasets  | | Electronic | Book | Movie |
|:----:|:----|---------:|----:|----:|
|           | #Users     | 46,086 | 61,722 |  46,893  | 
|   Review  | #Items     | 11,216 | 19,129 |  21,023   |
|           | #Reviews   | 199,617 | 731,800 | 1,149,294 |
|           | #Entities  | 30,301  | 96,173 | 594,452 |
|   KG      | #Relations | 20      | 12 | 22 |
|           | #Triples   | 68,620  | 285,118 | 2,130,368 |

Note that, the #Entities includes the aligned entities and its one-hop entities (neighbors), the #Relations includes the forward and backward relations, and the #Triples removes the triples with ''name'' and ''description'' relations. The following table demonstrates the included forward relations.

| Datasets | Relations |
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
