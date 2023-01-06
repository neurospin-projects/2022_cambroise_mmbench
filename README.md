## Usage

![PythonVersion](https://img.shields.io/badge/python-3.6%20%7C%203.7%20%7C%203.8%20%7C%203.9-blue)
![License](https://img.shields.io/badge/License-CeCILLB-blue.svg)
![PoweredBy](https://img.shields.io/badge/Powered%20by-CEA%2FNeuroSpin-blue.svg)

## Development

![Pep8](https://github.com/neurospin-projects/2022_cambroise_mmbench/actions/workflows/pep8.yml/badge.svg)
![Doc](https://github.com/neurospin-projects/2022_cambroise_mmbench/actions/workflows/documentation.yml/badge.svg)
![ExtraDoc](https://readthedocs.org/projects/mmbench/badge/?version=latest)

## Release

![PyPi](https://badge.fury.io/py/mmbench.svg)


# Benchmark Multi-Modal/Multi-View Models 

\:+1: If you are using the code please add a star to the repository :+1:

The availability of multiple data types provides a rich source of information
and holds promise for learning representations that generalize well across
multiple modalities. Multimodal data naturally grants additional
self-supervision in the form of shared information connecting the
different data types. Further, the understanding of different modalities and
the interplay between data types are non-trivial research questions and
long-standing goals in machine learning research.

Objective: studying how views information are disentanled by the models.

You can list all available workflows by running the following command in a
command prompt:

```
limri --help
```

If you have any question about the code or the paper, we are happy to help!

## Important links

* [Official source code repo.](https://github.com/neurospin-projects/2022_cambroise_mmbench)
* HTML documentation (stable release): WIP.
* [HTML documentation (latest release).](https://readthedocs.org/projects/mmbench)
* [Release notes.](https://github.com/neurospin-projects/2022_cambroise_mmbench/blob/master/CHANGELOG.rst)

## References

```
@inproceedings{antelmi2019,
    title   = "Sparse Multi-Channel Variational Autoencoder for the Joint
               Analysis of Heterogeneous Data", 
    author  = "Antelmi, Luigi and Ayache, Nicholas and Robert, Philippe and
               Lorenzi, Marco",
    booktitle = "Proceedings of the 36th International Conference on 
                 Machine Learning",
    pages   = "302-311",
    year    = 2019
}

@misc{sutter2021,
    title   = "Generalized Multimodal ELBO",
    author  = "Sutter, Thomas M. and Daunhawer, Imant and Vogt, Julia E.",
    publisher = "arXiv",
    year    = 2021
}

@article{aglinskas2022,
  title    = "Contrastive machine learning reveals the structure of
              neuroanatomical variation within autism",
  author   = "Aglinskas, Aidas and Hartshorne, Joshua K and Anzellotti, Stefano",
  journal  = "Science",
  volume   =  376,
  number   =  6597,
  pages    = "1070-1074",
  year     =  2022
}
  
```

## Where to start

This code was developed and tested with:
- Python version 3.5.6
- PyTorch version 1.4.0
- CUDA version 11.0
- The environment defined in `environments.txt`

First, install the requirements in your own environment. 

In order to be able to run the experiments, you need to have access to HBN or
EUAIMS data. Then, you must provide each script the path to these data 
by setting the `--dataset` and `--datasetdir` parameters.

Each data folder must contains at least 5 files:
- **rois_data.npy**: an array with 2 dimensions, the first corresponding to
  the subjects, the second to the different metric for each ROI.
- **rois_subjects.npy**: the list of subjects with the same ordering as
  in the previous file.
- **roi_names.npy**: the list of feature names for the `roi_data` file, with
  the same ordering as its columns.
- **clinical_data.npy**: an array with 2 dimensions, the first corresponding
  to the subjects, the second to the different score values.
- **clinical_subjects.npy**: the list of subjects with the same ordering as
  in the previous file.
- **clinical_names.npy** the list of feature names for the `clinical_data`
  file, with the same ordering as its columns.
- **metadata.tsv**: a table containing the metadata. It must contain at least
  4 columns: `participant_id` with the id of the subjects, corresponding
  to the `_subjects` files, `sex` with numerically encoded sex, `age` with
  continuous age, and `site` with acquisition site names. In EUAIMS, it can
  be good to have `asd` containing the 1-2 encoded diagnosis values (for the
  histogram).

Then start by:

* [browsing available examples.](https://mmbench.readthedocs.io/en/latest/auto_gallery/index.html)
* [looking at the list of available workflows.](https://mmbench.readthedocs.io/en/latest/generated/mmbench.workflow.html)
* [searching in the module API documentation.](https://mmbench.readthedocs.io/en/latest/generated/documentation.html)

## Experiments

### Train MoPoE

To choose between running the MVAE, MMVAE, and MoPoE-VAE, one needs to
change the script's `--method` variabe to `poe`, `moe`, or `joint_elbo`
respectively. By default, `joint_elbo` is selected.
Perform training on EUAIMS by running the following commands in a shell:

```
export DATASETDIR=/path/to/my/dataset
export OUTDIR=/path/to/the/output/directory

mmbench train-mopoe --dataset euaims --datasetdir $DATASETDIR --outdir $OUTDIR
--latent_dim 20 --input_dims 7,444 --beta 5 --batch_size 256
--likelihood normal --initial_learning_rate 0.002 --n_epochs 50
--learn_output_scale --allow_missing_blocks
```

### Embeddings

First generate samples using a mmbench sub-command. It will generate a file
named 'latent_vecs.npz' with all samples (n_samples, n_subjects, latent_dim).
All samples must have the same number of samples and subjects, but possibly
different latent dimensions.
In the following example we compare the MoPoe and sMCVE multi-views deep
learning models.

```
export WEIGHT1=/path/to/my/smcvae/weights
export WEIGHT2=/path/to/my/mopoe/weights

mmbench latent --dataset hbn --datasetdir $DATASETDIR --outdir $OUTDIR 
-smcvae_checkpointfile $WEIGHT1 --mopoe_checkpointfile $WEIGHT2
--smcvae_kwargs '{"latent_dim":10,"vae_model":"dense","noise_init_logvar":-3,"noise_fixed":False}'
```

### RSA

Perform a Representational Similarity Analysis (RSA) to compare latent
representations.

```
mmbench rsa --dataset hbn --datasetdir $DATASETDIR --outdir $OUTDIR
```

## Contributing

If you want to contribute to `mmbench`, be sure to review the [contribution guidelines](./CONTRIBUTING.rst).

## Citation

There is no paper published yet about `mmbench`.
We suggest that you aknowledge the brainprep team or reference to the code
repository:

```
Grigis, A. et al. (2023) MMBench source code (Version 0.01) [Source code]
https://github.com/neurospin-projects/2022_cambroise_mmbench.
```

Thank you.
