### 3. Run the example
#### 3.0 Setup environment
```bash
pip install -r requirements.txt
```

#### [3.1 Detection Training](./training.py)

The LUNA16 dataset was split into 10-fold to run cross-fold training and inference.

Taking fold 0 as an example, run:
```bash
python training.py \ 
    -e ./config/environment_vit_luna16_fold0.yaml \ 
    -c ./config/config_infer_vitdet2dimg_luna16_80g.json \ 
    -m /blue/bianjiang/tienyuchang/basemodel/checkpoint_test.pth \ 
    -d
```
Or modify and run:
```bash
sbatch run_train.sh
```

This python script uses batch size and patch size defined in [./config/config_infer_vitdet2dimg_luna16_80g.yaml](./config/config_infer_vitdet2dimg_luna16_80g.yaml).
The environment config include train/test model_path, data_base_dir, data_list_file_path, tfevent_path, and result_list_file_path defined in [./config/environment_vit_luna16_fold0.yaml](./config/environment_vit_luna16_fold0.yaml).

If you are tuning hyper-parameters, please also add `--verbose` flag.
Details about matched anchors during training will be printed out.

For each fold, 95% of the training data is used for training, while the rest 5% is used for validation and model selection.

#### [3.2 Detection Testing](./testing.py)

For fold i, please run
```bash
python testing.py \ 
    -e ./config/environment_vit_luna16_fold0.yaml \ 
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