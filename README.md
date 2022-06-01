# fastmri-intervention (part of fastMRI, RUMC/DIAG)

#### Py3.10+

Clone this repository to `path/to/module`.

```commandline
cd path/to/module
pip install .
python -m ./prep --pelvis path/to/chansey/pelvis --archive path/to/radng_diag_prostate --json ./workflow.json
```

where **./workflow.json** is
```
{
  "out_dir": "--pelvis / base output directory",
  "archive_dir": "--archive /Prostate-mpMRI-ScientificArchive/RUMC?",
  "gc_slug": "grand challenge reader study slug",
  "gc_api": "grand challenge API key",
  "invalidate": ['mha', 'annotations', 'nnunet'] invalidates those data files,
  "task_id": 500-999,
  "task_name": "fastmri_intervention?",
  "docker_version": 1+
}
```
