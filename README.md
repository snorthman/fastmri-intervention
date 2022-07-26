# fastmri-intervention (part of fastMRI, RUMC/DIAG)

#### Py3.10+

Clone this repository to `path/to/module`.

```commandline
pip install git+https://github.com/snorthman/fastmri-intervention
```

where **./settings.json** is
```
{
  "out_dir": "base output directory",
  "archive_dir": "base archive directory",
  "gc_slug": "grand challenge reader study slug",
  "gc_api": "grand challenge API key",
  "task_id": 500,
  "task_name": "fastmri_intervention"
}
```

TODO

convert new set to mha, needs different mapping setting
200 annotated needle images, 100 non-needle images

> there are no more non-biopsy tfi2d scans, need +100 from t2_?

100 needle and 100 non-needle for training
100 needle and 100 non-needle for testing

need code to properly split this

train nnunet on trainset
create statistics code for inference testset
