
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
    num_bytes: 90862713.05949003
    num_examples: 50000
  - name: test
    num_bytes: 18172542.611898005
    num_examples: 10000
  download_size: 116100876
  dataset_size: 109035255.67138803
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train-*
  - split: test
    path: data/test-*
---

Output:
{
    "extracted_code": ""
}
