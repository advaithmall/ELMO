# ALL MY CODE HAS BEEN STORED IN ONEDRIVE FOR CONVENIENCE, PLEASE FOLLOW INSTRUCTIONS TO DOWNLOAD
## Advaith Malladi
## 2021114005
### DIrectory Structure

```
.
├── c.txt
├── Report.pdf
├── task1
│   ├── dataset.py
│   ├── eval.py
│   ├── Figure_1.png
│   ├── measure.txt
│   ├── model.pt
│   ├── model.py
│   ├── __pycache__
│   │   ├── dataset.cpython-310.pyc
│   │   └── model.cpython-310.pyc
│   ├── senti.pt
│   ├── task2.py
│   └── train.py
└── task2
    ├── dataset.py
    ├── eval.py
    ├── model_2.pt
    ├── model.py
    ├── nli.pt
    ├── nli.py
    └── train.py

3 directories, 20 files

```
### run:

```
cd task1
python3
import torch
from torchtext.vocab import GloVe
global_vectors = GloVe(name='840B', dim=300)

```
### download: https://iiitaphyd-my.sharepoint.com/:f:/g/personal/advaith_malladi_research_iiit_ac_in/EiKSSha4qbtIvj_WGblrc78B3zGkrLxuaakiNsjTG57epQ?e=RL0Irm

### run:

```

cd task2
python3
import torch
from torchtext.vocab import GloVe
global_vectors = GloVe(name='840B', dim=300)

```

### download: https://iiitaphyd-my.sharepoint.com/:f:/g/personal/advaith_malladi_research_iiit_ac_in/Enipyf9dHjlIojW3TMfpOeQBkbTMfoNBl64HOfvquXbvlQ?e=qHUOl4

## in task1

### to train ELMO model, run:

```

python3 train.py

```

### to train downstream task:

```
python3 task2.py

```

### to test downstream task:

```
python3 eval.py

```

## in task2

### to train ELMO model, run:

```

python3 train.py

```

### to train downstream task:

```
python3 nli.py

```

### to test downstream task:

```
python3 eval.py

```
