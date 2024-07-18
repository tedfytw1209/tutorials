### 3. Run the example
#### 3.0 Setup environment
```bash
pip install -r requirements.txt
```

#### [3.1 Super Resolution Training](./training.py)

There are two aviable datasets: [Mednist](./config/environment_mednist.yaml) and [EyeQ](./config/environment_eyeq.yaml).

- Mednist:
    - train_config: [config_infer_vitconv_mednist_80g.yaml](./config/config_infer_vitconv_mednist_80g.yaml)

- EyeQ:
    - train_config: [config_infer_vitconv_eyeq_80g.yaml](./config/config_infer_vitconv_eyeq_80g.yaml)

Run:
```bash
python training.py \ 
    -c [train_config] \ 
    -d
```
Or modify and run:
```bash
sbatch run_train.sh
```

All settings are defined in [config_infer_vitconv_eyeq_80g.yaml](./config/config_infer_vitconv_eyeq_80g.yaml).
Include architecture, dataset & detector, file, training, and pretrain setting.

95% of the training data is used for training, while the rest 5% is used for validation and model selection.

#### [3.2 Super Resolution Testing](./testing.py)

For fold i, please run
```bash
python testing.py \ 
    -c [train_config] \ 
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