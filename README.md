# A Conditional Splitting Framework for Efficient Constituency Parsing
This repository contains the source code of our paper "A Conditional Splitting Framework for Efficient Constituency Parsing" in ACL 2021.
## Requirements
* `python`: 3.7
* `pytorch`: 1.4
* `transformers`: 3.0

## Usage
Put the data into data folder

To train the English constituency parser:

    ./run_en.sh

To train the SPMRL constituency parser:

    ./run_spmrl.sh
    
To train the sentence level discourse parser:
 
    ./run_discourse.sh


  
## Citation
Please cite our paper if you found the resources in this repository useful.

    @inproceedings{nguyen-etal-2021-conditional,
    title = "A Conditional Splitting Framework for Efficient Constituency Parsing",
    author = "Nguyen, Thanh-Tung  and
      Nguyen, Xuan-Phi  and
      Joty, Shafiq  and
      Li, Xiaoli",
    booktitle = "Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers)",
    month = aug,
    year = "2021",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.acl-long.450",
    doi = "10.18653/v1/2021.acl-long.450",
    pages = "5795--5807",
    abstract = "We introduce a generic seq2seq parsing framework that casts constituency parsing problems (syntactic and discourse parsing) into a series of conditional splitting decisions. Our parsing model estimates the conditional probability distribution of possible splitting points in a given text span and supports efficient top-down decoding, which is linear in number of nodes. The conditional splitting formulation together with efficient beam search inference facilitate structural consistency without relying on expensive structured inference. Crucially, for discourse analysis we show that in our formulation, discourse segmentation can be framed as a special case of parsing which allows us to perform discourse parsing without requiring segmentation as a pre-requisite. Experiments show that our model achieves good results on the standard syntactic parsing tasks under settings with/without pre-trained representations and rivals state-of-the-art (SoTA) methods that are more computationally expensive than ours. In discourse parsing, our method outperforms SoTA by a good margin.",}
