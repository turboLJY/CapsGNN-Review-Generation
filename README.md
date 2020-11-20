# CapsGNN-Review-Generation

This repository contains the source code for the CIKM 2020 paper "[Knowledge-Enhanced Personalized Review Generation with Capsule Graph Neural Network](https://arxiv.org/abs/2010.01480)".

# Directory

- [Requirements](#Requirements)
- [Datasets](#Datasets)
- [Training Instructions](#Training Instructions)
- [Testing Instructions](#Testing Instructions)
- [License](#License)
- [Reference](#References)

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

# Training Instructions

Before implement our code, you need to prepare the following dictionaries related to review text:

```
context.pkl and context_rev.pkl: the mapping from context into idx, and vice versa (e.g., Item_IDxxx -> 1010, 1010 -> Item_IDxxx).
aspect.pkl and aspect_rev.pkl: the mapping from aspects into idx, and vice versa (e.g., director -> 3, 3 -> director).
token.pkl and token_rev.pkl: the mapping from tokens into idx, and vice versa (e.g., Tim_burton -> 199, 199 -> Tim_burton).
```

and following dictionaries related to knowledge graph:

```
node.pkl and node_rev.pkl: the mapping from graph nodes into idx, and vice versa (e.g., m.01758 -> 435, 435 -> m.01758).
relation.pkl and relation_rev.pkl: the mapping from graph relations into idx, and vice versa (e.g., film.director.film -> 10, 10 -> film.director.film).
node_2_neighbor.pkl: the mapping from graph nodes into their one-hop neighbors.
node_2_name.pkl: the mapping from graph nodes into their names.
user_item_graph.pkl: the mapping from (user, item) pair to their heterogeneous KG, and the HKG is organized as Pytorch geometric structure.
```

After preparing all the dictionaries, you can run the run.sh file in aspect and review directories to train the two modules, respectively. 

```
sh run.sh
```

# Testing Instructions

At the test stage, you should (only) run the run.sh file in review directories to test the generation performance of our model.

# License

```
License agreement
This dataset is made freely available to academic and non-academic entities for non-commercial purposes such as academic research, teaching, scientific publications, or personal experimentation. Permission is granted to use the data given that you agree:
1. That the dataset comes “AS IS”, without express or implied warranty. Although every effort has been made to ensure accuracy, we do not accept any responsibility for errors or omissions. 
2. That you include a reference to the dataset in any work that makes use of the dataset. For research papers, cite our preferred publication as listed on our References; for other media cite our preferred publication as listed on our website or link to the dataset website.
3. That you do not distribute this dataset or modified versions. It is permissible to distribute derivative works in as far as they are abstract representations of this dataset (such as models trained on it or additional annotations that do not directly include any of our data) and do not allow to recover the dataset or something similar in character.
4. That you may not use the dataset or any derivative work for commercial purposes as, for example, licensing or selling the data, or using the data with a purpose to procure a commercial gain.
5. That all rights not expressly granted to you are reserved by us (Wayne Xin Zhao, School of Information, Renmin University of China).
```

# References

If this work is useful in your research, please cite our paper.

```
@inproceedings{junyi2020review,
  title={{K}nowledge-{E}nhanced {P}ersonalized {R}eview {G}eneration with {C}apsule {G}raph {N}eural {N}etwork},
  author={Junyi Li, Siqing Li, Wayne Xin Zhao, Gaole He, Zhicheng Wei, Nicholas Jing Yuan, Ji-Rong Wen},
  booktitle={CIKM},
  year={2020}
}
```
