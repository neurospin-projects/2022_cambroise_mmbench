![Pep8](https://github.com/neurospin-projects/2022_cambroise_mmbench/actions/workflows/pep8.yml/badge.svg)


# Benchmark Multu-Modal/Multi-View Models 

\:+1: If you are using the code please add a star to the repository :+1:

The availability of multiple data types provides a rich source of information
and holds promise for learning representations that generalize well across
multiple modalities. Multimodal data naturally grants additional
self-supervision in the form of shared information connecting the
different data types. Further, the understanding of different modalities and
the interplay between data types are non-trivial research questions and
long-standing goals in machine learning research.

If you have any question about the code or the paper, we are happy to help!

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


## Preliminaries

This code was developed and tested with:
- Python version 3.5.6
- PyTorch version 1.4.0
- CUDA version 11.0
- The environment defined in `environment.txt`

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


## Experiments

First generate samples using a mmbench sub-command. It will generate a file
named 'latent_vecs.npz' with all samples (n_samples, n_subjects, latent_dim).
All samples must have the same number of samples and subjects, but possibly
different latent dimensions.
In the following example we compare the MoPoe and sMCVE multi-views deep
learning models.

```
mmbench bench-latent --dataset hbn --datasetdir $DATASETDIR --outdir $OUTDIR --smcvae_checkpointfile $WEIGHT1 --mopoe_checkpointfile $WEIGHT2 --smcvae_kwargs '{"latent_dim":10,"vae_model":"dense","noise_init_logvar":-3,"noise_fixed":False}'
```

Then perform a Representational Similarity Analysis (RSA) to compare latent
representations.

```
mmbench bench-rsa --dataset hbn --datasetdir $DATASETDIR --outdir $OUTDIR
```

## Citation

There is no paper published yet about this project.
