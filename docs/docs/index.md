# local_llama3 documentation!

## Description

Fine-tune Llama 3 on a dataset of patient-doctor conversations.

## Commands

The Makefile contains the central entry points for common tasks related to this project.

### Syncing data to cloud storage

* `make sync_data_up` will use `aws s3 sync` to recursively sync files in `data/` up to `s3://llama3/data/`.
* `make sync_data_down` will use `aws s3 sync` to recursively sync files from `s3://llama3/data/` to `data/`.


