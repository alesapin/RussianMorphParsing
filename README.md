## Training and evaluation of morphemeic parsing model for Russian language

To train and evaluate model you can use following commands:
```(bash)
$ pip3 install -r requirements.txt
$ cd src
$ python3 run.py --models-config=../configs/model_config.json --model GBDT --train-data-path ../data/cross_train.txt --test-data-path ../data/cross_test.txt  --dictionary cross_lexica
$ python3 run.py --models-config=../configs/model_config.json --model CNN --train-data-path ../data/cross_train.txt --test-data-path ../data/cross_test.txt  --dictionary cross_lexica
$ python3 run.py --models-config=../configs/model_config.json --model LSTM --train-data-path ../data/cross_train.txt --test-data-path ../data/cross_test.txt  --dictionary cross_lexica
```
