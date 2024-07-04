### 3. Run the example
#### 3.0 Setup environment
```bash
pip install -r requirements.txt
```

#### [3.1 Super Resolution Training](./training.py)

There are two aviable datasets: [Mednist](./config/environment_mednist.yaml) and [EyeQ](./config/environment_eyeq.yaml).

- Mednist:
    - environment_config: [environment_mednist.yaml](./config/environment_mednist.yaml)
    - train_config: [config_infer_vitconv_mednist_80g.yaml](./config/config_infer_vitconv_mednist_80g.yaml)

- EyeQ:
    - environment_config: [environment_eyeq.yaml](./config/environment_eyeq.yaml)
    - train_config: [config_infer_vitconv_eyeq_80g.yaml](./config/config_infer_vitconv_eyeq_80g.yaml)

Run:
```bash
python training.py \ 
    -e [environment_config] \ 
    -c [train_config] \ 
    -p ./pretrain_config/config_monai.yaml \ 
    -m /blue/bianjiang/tienyuchang/basemodel/checkpoint_test.pth \ 
    -d
```
Or modify and run:
```bash
sbatch run_train.sh
```

This python script uses batch size and patch size defined in [config_infer_vitconv_eyeq_80g.yaml](./config/config_infer_vitconv_eyeq_80g.yaml).
The environment config include train/test model_path, data_base_dir, data_list_file_path, tfevent_path, and result_list_file_path defined in [environment_eyeq.yaml](./config/environment_eyeq.yaml).
The pre-trained model special setting in [./pretrain_config/config_monai.yaml](./pretrain_config/config_monai.yaml)

95% of the training data is used for training, while the rest 5% is used for validation and model selection.

#### [3.2 Super Resolution Testing](./testing.py)

For fold i, please run
```bash
python testing.py \ 
    -e [environment_config]l \ 
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