
Input:
From the Hugging Face README provided in “# README,” extract and output only the Python code required for execution. Do not output any other information. In particular, if no implementation method is described, output an empty string.

# README
---
dataset_info:
  features:
  - name: image
    dtype: image
  - name: label
    dtype:
      class_label:
        names:
          '0': '0'
          '1': '1'
          '2': '2'
          '3': '3'
          '4': '4'
          '5': '5'
          '6': '6'
          '7': '7'
          '8': '8'
          '9': '9'
  splits:
  - name: train
    num_bytes: 128062110.875
    num_examples: 73257
  - name: test
    num_bytes: 44356634.0
    num_examples: 26032
  - name: extra
    num_bytes: 965662156.625
    num_examples: 531131
  - name: train_balanced
    num_bytes: 90862463.02701217
    num_examples: 50000
  - name: test_balanced
    num_bytes: 18172492.605402432
    num_examples: 10000
  download_size: 1321906455
  dataset_size: 1247115857.1324146
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train-*
  - split: test
    path: data/test-*
  - split: extra
    path: data/extra-*
  - split: train_balanced
    path: data/train_balanced-*
  - split: test_balanced
    path: data/test_balanced-*
---

Output:
{
    "extracted_code": ""
}
