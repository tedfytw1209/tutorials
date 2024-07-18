### 3. Run the example
#### 3.0 Setup environment
```bash
pip install -r requirements.txt
```

#### [3.1 Detection Training](./training.py)

The LUNA16 dataset was split into two dataset training and testing (9:1).

For training, run:
```bash
python training.py \ 
    -c ./config/config_infer_vitdet2dimg_luna16_80g.yaml \ 
    -d
```
Or modify and run:
```bash
sbatch run_train.sh
```

This python script uses all srtting defined in [./config/config_infer_vitdet2dimg_luna16_80g.yaml](./config/config_infer_vitdet2dimg_luna16_80g.yaml).
Include architecture, dataset & detector, file, training, and pretrain setting.

If you are tuning hyper-parameters, please also add `--verbose` flag.
Details about matched anchors during training will be printed out.

For each fold, 95% of the training data is used for training, while the rest 5% is used for validation and model selection.

#### [3.2 Detection Testing](./testing.py)

For fold i, please run
```bash
python testing.py \ 
    -c ./config/config_infer_vitdet2dimg_luna16_80g.yaml \ 
    -d
```
Or modify and run:
```bash
sbatch run_test.sh
```

#### 3.3 See result
You can see training step result and image by using `tensorboard`.
Run
```bash
tensorboard --logdir [tfevent_path]
```