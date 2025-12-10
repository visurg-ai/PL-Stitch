**PL-Stitch**
-------------

<!-- Paper -->
[![Paper](https://img.shields.io/badge/Paper-arXiv-b31b1b?style=for-the-badge)](https://www.arxiv.org/abs/2511.17805)
[![Model](https://img.shields.io/badge/Model-HuggingFace-ffcc00?style=for-the-badge&logo=huggingface)](https://huggingface.co/visurg/PL-Stitch)



This is the official repository for the paper [A Stitch in Time: Learning Procedural Workflow via Self-Supervised Plackett-Luce Ranking](https://www.arxiv.org/abs/2511.17805).

*PL-Stitch* is an image foundation model that captures visual changes over time, enabling procedural activity understanding. It takes an image as input and produces a feature vector as output, leveraging the novel Plackett-Luce temporal ranking objective to build a comprehensive understanding of both the static semantic information and the procedural context within each frame.



Star ⭐ us if you like it!

<img src="https://github.com/user-attachments/assets/e7b35eca-ff57-4d3c-960c-ff4c4a93f092" />

## News

<!--<br>-->
* 21/November/2025. The [arXiv](https://www.arxiv.org/abs/2511.17805) version of the paper is released.

<br>





If you use our model or code in your research, please cite our paper:

```
@misc{che2025stitchtimelearningprocedural,
      title={A Stitch in Time: Learning Procedural Workflow via Self-Supervised Plackett-Luce Ranking}, 
      author={Chengan Che and Chao Wang and Xinyue Chen and Sophia Tsoka and Luis C. Garcia-Peraza-Herrera},
      year={2025},
      eprint={2511.17805},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2511.17805}, 
}
```

Abstract
--------
Procedural activities, ranging from routine cooking to complex surgical operations, are highly structured as a set of actions conducted in a specific temporal order. Despite their success on static images and short clips, current self-supervised learning methods often overlook the procedural nature that underpins such activities. We expose the lack of procedural awareness in current SSL methods with a motivating experiment: models pretrained on forward and time-reversed sequences produce highly similar features, confirming that their representations are blind to the underlying procedural order. To address this shortcoming, we propose PL-Stitch, a self-supervised framework that harnesses the inherent temporal order of video frames as a powerful supervisory signal. Our approach integrates two novel probabilistic objectives based on the Plackett-Luce (PL) model. The primary PL objective trains the model to sort sampled frames chronologically, compelling it to learn the global workflow progression. The secondary objective, a spatio-temporal jigsaw loss, complements the learning by capturing fine-grained, cross-frame object correlations. Our approach consistently achieves superior performance across five surgical and cooking benchmarks. Specifically, PL-Stitch yields significant gains in surgical phase recognition (e.g., +11.4 pp k-NN accuracy on Cholec80) and cooking action segmentation (e.g., +5.7 pp linear probing accuracy on Breakfast), demonstrating its effectiveness for procedural video representation learning.

<br>



🔧 Install dependencies
--------------------------------------------------

Install the following dependencies in your local setup:

   ```bash
   $ git clone git@github.com:visurg-ai/PL-Stitch.git
   $ cd PL-Stitch && pip install -r requirements.txt
   ```


🗂️ Data preparation
-------------------
Download the pretraining dataset ([LEMON](https://github.com/visurg-ai/LEMON)) and evaluation datasets ([Cholec80](https://camma.unistra.fr/datasets/), [AutoLaparo](https://autolaparo.github.io/), [M2CAI16](https://camma.unistra.fr/datasets/), [Breakfast](https://serre.lab.brown.edu/breakfast-actions-dataset.html), [GTEA](https://cbs.ic.gatech.edu/fpv/)).

For efficient data loading, we use the LMDB format. To process these datasets into LMDB, run the following code for both pretraining and evaluation.


```bash
$ bash scripts/covert_lmdb.sh
```


🚀 Training
-----------
We provide a script with default parameters for PL-Stitch model pretraining.

```bash
$ bash scripts/pretrain.sh
```



📊 Evaluation
--------------
We provide a script for the downstream task evaluation.
```bash
$ bash scripts/eval.sh
```

   

<br>

**t-SNE visualization of frozen backbone features for Cholec80 phase recognition**

<img src="https://github.com/user-attachments/assets/9266ea99-44c6-4d28-a12c-7bd7e9361168" />



🚩 PL-Stitch model
------------------

You can download the checkpoint at [🤗 PL-Stitch](https://huggingface.co/visurg/PL-Stitch) and run the following code to extract features from your video frames.


   ```python
   import torch
   from PIL import Image
   from pl_stitch.build_model import build_model

   # Load the pre-trained pl_stitch model
   pl_stitch = build_model(pretrained_weights = 'your path to the model')
   pl_stitch.eval()

   # Load the image and convert it to a PyTorch tensor
   img_path = 'path/to/your/image.jpg'
   img = Image.open(img_path)
   img = img.resize((224, 224))
   img_tensor = torch.tensor(np.array(img)).unsqueeze(0).to('cuda')

   # Extract features from the image
   outputs = pl_stitch(img_tensor)
   ```






