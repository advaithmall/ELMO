# Embeddings from Language Modelling (ELMO) for contextual embeddings
### Downstream task has also been performed

```
.
├── downstream
│   ├── backward_model.pt
│   ├── data
│   │   ├── test.csv
│   │   └── train.csv
│   ├── elmo.py
│   ├── forward_model.pt
│   ├── idx2word.pt
│   └── word2idx.pt
├── pretrain
│   ├── backward_model.pt
│   ├── data
│   │   ├── test.csv
│   │   └── train.csv
│   ├── downstream_model.pt
│   ├── down_stream.py
│   ├── forward_model.pt
│   ├── idx2word.pt
│   └── word2idx.pt
├── Readme.md
└── Report.pdf

4 directories, 17 files

```

### Please download all the missing files from the provided one drive link, please adhere to the above directory structure: https://iiitaphyd-my.sharepoint.com/:f:/g/personal/advaith_malladi_research_iiit_ac_in/EqzZacucfSdFnOY6UHdE-rgBydnQa8HK_BWotMTVJwAfJg?e=iFyL95


### to run the biLSTM (ELMO) model and to reproduce results, please run the following command:

```
cd pretrain
python3 -W ignore elmo.py
```

### to run the downstream task using the pre-trained ELMO embeddings and to reproduce results, please run the following command:

```
cd downstream
python3 -W ignore down_stream.py

```

### to read answers to my theory question, to look at the analysis and related diagram, read:
```
Report.pdf
```
