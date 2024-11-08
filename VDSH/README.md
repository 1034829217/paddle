# Variational Deep Semantic Hashing (SIGIR'2017)
The Paddle implementation of the models and experiments of [Variational Deep Semantic Hashing](http://students.engr.scu.edu/~schaidar/paper/Variational_Deep_Hashing_for_Text_Documents.pdf) (SIGIR 2017).

Author: Suthee Chaidaroon and Yi Fang

# Training and Evaluating the model
To train the unsupervised learning model, run the following command:
```
python train_VDSH.py -d [dataset name] -g [gpu number] -b [number of bits]
```

# Bibtex
```
@inproceedings{Chaidaroon:2017:VDS:3077136.3080816,
 author = {Chaidaroon, Suthee and Fang, Yi},
 title = {Variational Deep Semantic Hashing for Text Documents},
 booktitle = {Proceedings of the 40th International ACM SIGIR Conference on Research and Development in Information Retrieval},
 series = {SIGIR '17},
 year = {2017},
 isbn = {978-1-4503-5022-8},
 location = {Shinjuku, Tokyo, Japan},
 pages = {75--84},
 numpages = {10},
 url = {http://doi.acm.org/10.1145/3077136.3080816},
 doi = {10.1145/3077136.3080816},
 acmid = {3080816},
 publisher = {ACM},
 address = {New York, NY, USA},
 keywords = {deep learning, semantic hashing, variational autoencoder},
}
```

