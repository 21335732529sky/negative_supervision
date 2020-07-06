#Desc
# Text Classification with Negative Supervision
This repository is the implementation of our work ([Paper](https://www.aclweb.org/anthology/2020.acl-main.33/))
Under consturuction

Bibtex:
```
@inproceedings{ohashi-etal-2020-text,
    title = "Text Classification with Negative Supervision",
    author = "Ohashi, Sora  and
      Takayama, Junya  and
      Kajiwara, Tomoyuki  and
      Chu, Chenhui  and
      Arase, Yuki",
    booktitle = "Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics",
    month = jul,
    year = "2020",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/2020.acl-main.33",
    pages = "351--357",
    abstract = "Advanced pre-trained models for text representation have achieved state-of-the-art performance on various text classification tasks. However, the discrepancy between the semantic similarity of texts and labelling standards affects classifiers, i.e. leading to lower performance in cases where classifiers should assign different labels to semantically similar texts. To address this problem, we propose a simple multitask learning model that uses negative supervision. Specifically, our model encourages texts with different labels to have distinct representations. Comprehensive experiments show that our model outperforms the state-of-the-art pre-trained model on both single- and multi-label classifications, sentence and document classifications, and classifications in three different languages.",
}
```