## Training and evaluation of several morphemeic parsing models for Russian language

### Gradient boosted descision trees
To train and evaluate GBDT model you can use following commands:
```(bash)
$ pip3 install -r requirements.txt
$ cd src
$ python3 run.py --models-config=../configs/model_config.json --model GBDT --data-path ../data/cross_lexica.txt --dictionary cross_lexica
```

Or you can use pretrained model with:
```(bash)
$ pip3 install -r requirements.txt
$ cd models
# download model from yandex disk. tikhonov: https://yadi.sk/d/ixy1IqfABa55Lw, cross_lexica: https://yadi.sk/d/BCciTto0sVS1VQ
$ cd ../src
$ python3 run.py --models-config=../configs/model_config.json --model GBDT --data-path ../data/tikhonov.txt --dictionary tikhonov --load ../models/best_tikhonov_model.bin
...
{'catboost': {'quality': [('Precision', 0.9683026866329002), ('Recall', 0.985198151519395), ('F1', 0.9766773562117362), ('Accuracy', 0.962570735650768), ('Word accuracy', 0.8623633524206142)]}}
```
