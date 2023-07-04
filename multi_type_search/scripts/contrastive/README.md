How to train Hotpot
-

### Create training and validation data

In the `{root}/data/full/hotpot` folder this is a script

`create_hotpot_dataset.py` that handles the creation of dataset files for training/validation.

Here's an example of how to create a validation dataset on the fullwiki dataset.

```shell
python create_hotpot_dataset.py -o ./fullwiki_val.json -ds validation -dt fullwiki
```

Note: The datasets make references to docs that do not exist in their context.  This happens in the raw hotpot data too
:/ so expect to see a lot of error output.


Training the model
-

In the `{root}/multi_type_search/scripts/contrastive` there is a script

`train_contrastive_model.py` that handles the training of the model.

If you want to run multiple training jobs across GPUs you need to use  

```shell
CUDA_VISIBLE_DEVICES="0" python train_contrastive_model.py
```

otherwise the training script will try to use all GPUs.