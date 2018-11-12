# Topic Memory Networks for Short Text Classification
Topic Memory Networks (TMN) employs a novel topic memory mechanism to encode
latent topic representations indicative of class labels in short text classification. 
TMN jointly explores topic inference and text classification with memory networks in an end-to-end manner.

## Requirements
* TensorFlow >= 1.2.1
* Keras >= 2.0.8
* gensim >= 2.1.0

## Input data format
A sampled data is provided in `data/tmn/tmn_data.txt` from [TagMyNews](http://acube.di.unipi.it/tmn-dataset/). One data sample per line, the text and the label are separated by `######`.

```
text1######label1
text2######label2
text3######label3
...
```

## How to run
Preprocess data:
```
$ cd scripts/
$ python process_tmn.py <input_data_file>    
e.g. python process_tmn.py ../data/tmn/tmn_data.txt
```

Run TMN:
```
$ python tmn_run.py <input_data_dir> <embedding_file> <output_dir> <topic_num>     
e.g. python tmn_run.py ../data/tmn /emb/glove.6B.200d.txt ../output 50
```

More detailed configurations can be found in `tmn_run.py`.

## Cite
```
@inproceedings{DBLP:conf/emnlp/Zeng18,
        author    = {Jichuan Zeng and
                    Jing Li and
                    Yan Song and
                    Cuiyun Gao and
		            Michael R. Lyu and
		            Irwin King},
        title     = "{Topic Memory Networks for Short Text Classification}",
        booktitle = {Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing, {EMNLP} 2018, Brussels, Belgium, October 31–November 4, 2018},
        year      = {2018},
}
```

## Disclaimer

The code is for research purpose only and released under the Apache License, Version 2.0 (https://www.apache.org/licenses/LICENSE-2.0).