## Training and evaluation of morphemeic parsing model for Russian language

### Available models:

- LSTM
- GDBT
- CNN

### Available dictionaries:

- CrossLexica 23426 words
- Tikhonov 96046 words
- Merged dictionaries from two previous

### Run

To train and evaluate model you can use following commands:
```(bash)
$ pip3 install -r requirements.txt
$ cd src
$ python3 run.py --models-config=../configs/model_config.json --model LSTM --train-data-path ../data/cross_train.txt --test-data-path ../data/cross_test.txt  --dictionary cross_lexica
```
