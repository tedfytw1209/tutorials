### 3. Run the example
#### 3.0 Setup environment
```bash
pip install -r requirements.txt
```

#### [3.1 Detection Training](./luna16_training.py)

The LUNA16 dataset was split into 10-fold to run cross-fold training and inference.

Taking fold 0 as an example, run:
```bash
python training.py \ 
    -e ./config/environment_vit_luna16_fold0.json \ 
    -c ./config/config_infer_vitdet2dimg_luna16_80g.json \ 
    -m /blue/bianjiang/tienyuchang/basemodel/checkpoint_test.pth \ 
    -d
```
Or modify and run:
```bash
sbatch run_train.sh
```

This python script uses batch size and patch size defined in [./config/config_infer_vitdet2dimg_luna16_80g.json](./config/config_infer_vitdet2dimg_luna16_80g.json).
The environment config include train/test model_path, data_base_dir, data_list_file_path, tfevent_path, and result_list_file_path defined in [./config/environment_vit_luna16_fold0.json](./config/environment_vit_luna16_fold0.json).

If you are tuning hyper-parameters, please also add `--verbose` flag.
Details about matched anchors during training will be printed out.

For each fold, 95% of the training data is used for training, while the rest 5% is used for validation and model selection.
The training and validation curves for 300 epochs of 10 folds are shown below. The upper row shows the training losses for box regression and classification. The bottom row shows the validation mAP and mAR for IoU ranging from 0.1 to 0.5.

#### [3.2 Detection Testing](./luna16_testing.py)

For fold i, please run
```bash
python testing.py \ 
    -e ./config/environment_vit_luna16_fold0.json \ 
    -c ./config/config_infer_vitdet2dimg_luna16_80g.json \ 
    -d
```
Or modify and run:
```bash
sbatch run_test.sh
```

### Reference
[1] [Armato III, S. G., McLennan, G., Bidaut, L., McNitt-Gray, M. F., Meyer, C. R., Reeves, A. P., Zhao, B., Aberle, D. R., Henschke, C. I., Hoffman, E. A., Kazerooni, E. A., MacMahon, H., Van Beek, E. J. R., Yankelevitz, D., Biancardi, A. M., Bland, P. H., Brown, M. S., Engelmann, R. M., Laderach, G. E., Max, D., Pais, R. C. , Qing, D. P. Y. , Roberts, R. Y., Smith, A. R., Starkey, A., Batra, P., Caligiuri, P., Farooqi, A., Gladish, G. W., Jude, C. M., Munden, R. F., Petkovska, I., Quint, L. E., Schwartz, L. H., Sundaram, B., Dodd, L. E., Fenimore, C., Gur, D., Petrick, N., Freymann, J., Kirby, J., Hughes, B., Casteele, A. V., Gupte, S., Sallam, M., Heath, M. D., Kuhn, M. H., Dharaiya, E., Burns, R., Fryd, D. S., Salganicoff, M., Anand, V., Shreter, U., Vastagh, S., Croft, B. Y., Clarke, L. P. (2015). Data From LIDC-IDRI [Data set]. The Cancer Imaging Archive.](https://doi.org/10.7937/K9/TCIA.2015.LO9QL9SX)

[2] [Armato SG 3rd, McLennan G, Bidaut L, McNitt-Gray MF, Meyer CR, Reeves AP, Zhao B, Aberle DR, Henschke CI, Hoffman EA, Kazerooni EA, MacMahon H, Van Beeke EJ, Yankelevitz D, Biancardi AM, Bland PH, Brown MS, Engelmann RM, Laderach GE, Max D, Pais RC, Qing DP, Roberts RY, Smith AR, Starkey A, Batrah P, Caligiuri P, Farooqi A, Gladish GW, Jude CM, Munden RF, Petkovska I, Quint LE, Schwartz LH, Sundaram B, Dodd LE, Fenimore C, Gur D, Petrick N, Freymann J, Kirby J, Hughes B, Casteele AV, Gupte S, Sallamm M, Heath MD, Kuhn MH, Dharaiya E, Burns R, Fryd DS, Salganicoff M, Anand V, Shreter U, Vastagh S, Croft BY.  The Lung Image Database Consortium (LIDC) and Image Database Resource Initiative (IDRI): A completed reference database of lung nodules on CT scans. Medical Physics, 38: 915--931, 2011.](https://doi.org/10.1118/1.3528204)

[3] [Clark, K., Vendt, B., Smith, K., Freymann, J., Kirby, J., Koppel, P., Moore, S., Phillips, S., Maffitt, D., Pringle, M., Tarbox, L., & Prior, F. (2013). The Cancer Imaging Archive (TCIA): Maintaining and Operating a Public Information Repository. Journal of Digital Imaging, 26(6), 1045–1057.](https://doi.org/10.1007/s10278-013-9622-7)
