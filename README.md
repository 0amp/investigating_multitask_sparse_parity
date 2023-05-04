# investigating_multitask_sparse_parity

Our writup is here: https://www.overleaf.com/read/xnssphtqqcgd (IN-PROGRESS)

# contents of directory
- `old_work` contains initial experiemnts. Specifically, `Scaling Laws-2.ipynb`, `model_size_sweep`, and the respective `.npy` files contained an initial model sweep. 
- `quantization.py` contains the training script that was used locally by Kevin. `train.py` was a compressed version used to submit jobs on AWS. `analyze_models.ipynb` was used to do analysis after model runs were done. It will not run, because it is specific to the directory structure Kevin had set up. 
- `data-5-3-5pm.pkl.tar.gz` contains the final set of all model losses (in nats). The subtask specific model losses are not in this directory, because the files are all too large.
- `HTesting.ipynb` contains the code Arjun used to do hypothesis testing.