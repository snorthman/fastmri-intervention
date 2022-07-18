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
  "run_prep": ["dcm", "dcm2mha", "annotate", "mha2nnunet"],
  "task_id": 500,
  "task_name": "fastmri_intervention"
}
```