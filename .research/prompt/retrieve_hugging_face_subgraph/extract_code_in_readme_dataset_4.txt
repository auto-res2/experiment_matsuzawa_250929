
Input:
From the Hugging Face README provided in “# README,” extract and output only the Python code required for execution. Do not output any other information. In particular, if no implementation method is described, output an empty string.

# README
---
license: mit
---

# A kernel function which improves the accuracy and interpretability of large ensembles of neural networks

We describe a new kernel (i.e. similarity function between pairs of examples) which is computed using an ensemble of neural networks. It has the following properties:

- Using it to predict test labels (via k-nearest neighbors across the training set) yields even higher accuracy than the standard ensemble inference method
of averaging predictions, once the number of networks exceeds about 100. We believe this kernel + k-NN method is the state-of-the-art for inferencing large ensembles
(although such ensembles are rarely used in practice).
- Being a similarity function, it is highly interpretable. For each test example, it allows us to visualize training examples which are deemed to have
similar features by the training process, with much greater fidelity than e.g. penultimate layer embeddings. For instance, we use this to identify the (known) fact that
~10% of the CIFAR-10 test-set examples have a near-duplicate in the training set, and to identify a failure mode.

To compute the kernel for an ensemble of n=500 models, we provide the following simple code (which can be copy-paste run in your environment).

```
import torch
import torchvision
import huggingface_hub

def normalize(logits):
    logits = logits.float()
    logits = logits.log_softmax(-1)
    logits = (logits - logits.mean(0, keepdim=True)) / logits.std(0, keepdim=True)
    return logits

def compute_kernel(logits1, logits2):
    
    logits1 = normalize(logits1)
    logits2 = normalize(logits2)
    assert len(logits1) == len(logits2)
    
    kernel = torch.zeros(logits1.shape[1], logits2.shape[1]).cuda()
    for c in range(10):
        logits1_cls = logits1[..., c].cuda()
        logits2_cls = logits2[..., c].cuda()
        corr_cls = (logits1_cls.T @ logits2_cls) / len(logits1)
        kernel += corr_cls / 10
    return kernel

######################################################################################
#  Setup: Download CIFAR-10 labels and the outputs from 500 repeated training runs.  #
######################################################################################

labels_train = torch.tensor(torchvision.datasets.CIFAR10('cifar10', train=True).targets)
labels_test = torch.tensor(torchvision.datasets.CIFAR10('cifar10', train=False).targets)

api = huggingface_hub.HfApi()
fname = 'logs_saveoutputs_main/06109e85-f5d7-4ac8-b0b0-f03542f23234/log.pt'
obj_path = api.hf_hub_download('kjj0/cifar10-multirun-logits', repo_type='dataset',
                               filename=fname)
obj = torch.load(obj_path, map_location='cpu')

# print(obj['code']) # Uncomment if you want to see the training code

######################################################################################
#     Evaluate both the per-model and ensembled accuracy of the training outputs.    #
######################################################################################

each_acc = (obj['logits'].argmax(-1) == labels_test).float().mean(1)
avg_acc = each_acc.mean()
print('average single-model accuracy \t: %.2f' % (100 * avg_acc))

ens_pred = obj['logits'].mean(0).argmax(1)
ens_acc = (ens_pred == labels_test).float().mean()
print('ensemble accuracy (%d models) \t: %.2f' % (len(obj['logits']), 100 * ens_acc))
# (n.b. averaging probabilities instead of logits makes no difference)

######################################################################################
#               Evaluate the new kernel / ensemble inference method.                 #
######################################################################################

# use correlations between log_softmax outputs as a similarity metric for k-NN inference.
kernel = compute_kernel(obj['logits'], obj['logits_train'])
k = 3
nbrs = kernel.topk(k, dim=1)
nbr_labels = labels_train[nbrs.indices.cpu()]
pred = nbr_labels.mode(1).values
acc = (pred == labels_test).float().mean()
print('kernel accuracy (k-NN w/ k=%d) \t: %.2f' % (k, 100 * acc))

## average single-model accuracy   : 93.26
## ensemble accuracy (500 models)  : 94.69
## kernel accuracy (k-NN w/ k=3)   : 95.01
```

The training configuration we used to generate these 500 models (i.e. the script that we re-ran 500 times with different random seeds) yields a mean accuracy of 93.26%.
If we average the predictions across those 500 models, we attain a much improved accuracy of 94.69%.
If we predict the test-set labels using our kernel applied to pairs of (train, test) examples, using k-nearest neighbors with k=3,
then we attain an even higher accuracy of 95.01%.

We include 20,000 total runs of training for the same training configuration that generated the 500 runs used in the above.
The outputs of those runs (i.e. the logits predicted by the final model on the training and test examples) can be found as the other files in `logs_saveoutputs_main`.
If we compute the kernel with all 20,000 runs instead of 500, and use a weighting scheme based on the correlation values,
then the accuracy can be futher increased to 95.53%.
Note that increasing from 500 to 20,000 does not improve the accuracy of the averaged predictions,
so with 95.53% we have reached 0.84% higher than the standard ensemble accuracy.

We additionally include outputs from three other training configurations; their kernels seem to have the same properties.

## Interpretability-type applications

### Finding similar pairs

(Below:) We rank the CIFAR-10 test-set examples by their similarity to their most similar training-set example.
We show the 601th-648th most highly ranked test examples (out of 10,000), along with their matched training examples.
Many of them turn out to be visually similar pairs.

![the 600-650th most similar pairs](kernel_pairs_600_650.png)

We note that the penultimate-layer features almost entirely lack this property --
if we visualize the most similar pairs across all (test, train) pairs according to distance in penultimate feature space,
we will get not duplicates but instead just random highly confident examples which have all presumably collapsed to a similar point in space.
On the other hand, pairs which are given a high similarity score by our correlation kernel turn out to often be near-duplicates, and this holds true
for the most similar pairs even when we reduce the number of models in the ensemble down to a relatively small value like 10 or 20.

### Diagnosing failure modes

(Below:) We rank the CIFAR-10 test examples by how similar their most similar training-set example is, and then filter for cases where they have different labels.
The first (leftmost) column contains the top 8 such test examples, and then subsequent columns are their 9 nearest neighbors in the training set.
It appears that our network has difficulty seeing small objects.

![the highest-confidence failures](failure_mode.png)

### Some random examples

(Below:) We select 10 CIFAR-10 test examples at random (the first row), and display their two nearest neighbors according to the kernel (second two rows),
and the penultimate features from a single model (next two rows). The kernel yields images which are perceptually similar, whereas penultimate features
select nearly a random image of the same label.

![randomly chosen test examples, with their most similar train examples](random_pairs.png)

## Open questions

* The usage of `log_softmax` in the normalization step seems to be important, especially for making the kernel work with n < 1,000 (where n is the number of networks).
But for n -> infty, it becomes less important. Why -- is it somehow removing noise?
* Via the Neural Network Gaussian Process (NNGP) theory, it is possible to compute the expectation of this kernel for untrained / newly initialized networks
(at least if the log-softmax is removed). Is there any general theory for what this kernel becomes after training (i.e., what we are seeing here)?
* This kernel is implemented as a sum of 10 correlation kernels -- one for each class. But upon inspection, each of those has dramatically worse
k-NN accuracy than their sum, at least until n becomes on the order of thousands. Why?
* Removing log-softmax, despite harming the overall accuracy as discussed earlier,
apparently increases the k-NN accuracy (and generally quality) of the individual kernels. Why??
* How does this kernel compare to [TRAK](https://arxiv.org/abs/2303.14186)
or the datamodel embeddings from [https://arxiv.org/abs/2202.00622](https://arxiv.org/abs/2202.00622)?


Output:
{
    "extracted_code": "import torch\nimport torchvision\nimport huggingface_hub\n\ndef normalize(logits):\n    logits = logits.float()\n    logits = logits.log_softmax(-1)\n    logits = (logits - logits.mean(0, keepdim=True)) / logits.std(0, keepdim=True)\n    return logits\n\ndef compute_kernel(logits1, logits2):\n    \n    logits1 = normalize(logits1)\n    logits2 = normalize(logits2)\n    assert len(logits1) == len(logits2)\n    \n    kernel = torch.zeros(logits1.shape[1], logits2.shape[1]).cuda()\n    for c in range(10):\n        logits1_cls = logits1[..., c].cuda()\n        logits2_cls = logits2[..., c].cuda()\n        corr_cls = (logits1_cls.T @ logits2_cls) / len(logits1)\n        kernel += corr_cls / 10\n    return kernel\n\n######################################################################################\n#  Setup: Download CIFAR-10 labels and the outputs from 500 repeated training runs.  #\n######################################################################################\n\nlabels_train = torch.tensor(torchvision.datasets.CIFAR10('cifar10', train=True).targets)\nlabels_test = torch.tensor(torchvision.datasets.CIFAR10('cifar10', train=False).targets)\n\napi = huggingface_hub.HfApi()\nfname = 'logs_saveoutputs_main/06109e85-f5d7-4ac8-b0b0-f03542f23234/log.pt'\nobj_path = api.hf_hub_download('kjj0/cifar10-multirun-logits', repo_type='dataset',\n                               filename=fname)\nobj = torch.load(obj_path, map_location='cpu')\n\n# print(obj['code']) # Uncomment if you want to see the training code\n\n######################################################################################\n#     Evaluate both the per-model and ensembled accuracy of the training outputs.    #\n######################################################################################\n\neach_acc = (obj['logits'].argmax(-1) == labels_test).float().mean(1)\navg_acc = each_acc.mean()\nprint('average single-model accuracy \\t: %.2f' % (100 * avg_acc))\n\nens_pred = obj['logits'].mean(0).argmax(1)\nens_acc = (ens_pred == labels_test).float().mean()\nprint('ensemble accuracy (%d models) \\t: %.2f' % (len(obj['logits']), 100 * ens_acc))\n# (n.b. averaging probabilities instead of logits makes no difference)\n\n######################################################################################\n#               Evaluate the new kernel / ensemble inference method.                 #\n######################################################################################\n\n# use correlations between log_softmax outputs as a similarity metric for k-NN inference.\nkernel = compute_kernel(obj['logits'], obj['logits_train'])\nk = 3\nnbrs = kernel.topk(k, dim=1)\nnbr_labels = labels_train[nbrs.indices.cpu()]\npred = nbr_labels.mode(1).values\nacc = (pred == labels_test).float().mean()\nprint('kernel accuracy (k-NN w/ k=%d) \\t: %.2f' % (k, 100 * acc))\n\n## average single-model accuracy   : 93.26\n## ensemble accuracy (500 models)  : 94.69\n## kernel accuracy (k-NN w/ k=3)   : 95.01"
}
