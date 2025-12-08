<p align="center">
    <a href="https://visurg.ai/">
    <img src="https://github.com/user-attachments/assets/92fc59cb-5392-4a72-95b5-35104df129e0">
    </a>
</p>


[📚 Paper](https://www.arxiv.org/abs/2511.17805) - [🤖 Code](src) - [🤗 Model](https://huggingface.co/visurg/LemonFM)

Star ⭐ us if you like it!

<div align="center">
  <img src="https://github.com/user-attachments/assets/6250cd6a-1404-4786-9c15-fe396265940d" width="70%" > </img>
</div>


## News

<!--<br>-->
* 21/November/2025. The [arXiv](https://www.arxiv.org/abs/2511.17805) version of the paper is released.

<br>

This is the official repository for the paper [LEMON: A Large Endoscopic MONocular Dataset and Foundation Model for Perception in Surgical Settings](https://arxiv.org/abs/2503.19740).

This repository provides open access to the *LEMON* dataset, *LemonFM* foundation model, and training code. 

[*LEMON*](https://surg-3m.visurg.ai/) is a dataset of 4K surgical high-resolution videos totaling 938 hours from 35 diverse surgical procedure types. Each video is annotated for multi-label classification, indicating the surgical procedures carried out in the video, and for binary classification, indicating if it is robotic or non-robotic. The dataset's annotations can be found in [labels.json](https://github.com/visurg-ai/LEMON/blob/main/labels.json).

[*LemonFM*](https://huggingface.co/visurg/LemonFM) is an image foundation model for surgery, it receives an image as input and produces a feature vector of 1536 features as output. 

<!--The website of our dataset is: [http://LEMON.org](https://LEMON.org)-->

If you use our dataset, model, or code in your research, please cite our paper:

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



Install dependencies to recreate our LEMON dataset
--------------------------------------------------
<!--
* If you want to use Docker**, follow the next steps to download our container:

   ```bash
   # Download the repo
   $ git clone git@github.com:visurg-ai/LEMON.git
   $ cd LEMON/docker

   # Build the docker image
   $ docker build --build-arg USER=$(whoami) --build-arg UID=$(id -u) --build-arg GID=$(id -g) -t chengan/LEMON:latest .

   # Run videosum Docker container
   $ docker run --volume $HOME:/mnt/user_home --name LEMON --runtime nvidia chengan/LEMON:latest &

   # Execute the docker container and get into the terminal
   $ docker exec --user $(whoami) --workdir $HOME -it LEMON /bin/zsh
   ```
-->

* Install the following dependencies in your local setup:

   ```bash
   $ git clone git@github.com:visurg-ai/LEMON.git
   $ cd LEMON && pip install -r requirements.txt
   ```



LEMON dataset
--------------------------

You can use our code of the data curation pipeline and provided annotation file (["labels.json"](https://github.com/visurg-ai/LEMON/blob/main/labels.json)) to recreate the whole LEMON dataset.

1. Get your YouTube cookie:

   You need to provide a "cookies.txt" file if you want to download videos that require Youtube login. 

   Use the [cookies](https://github.com/yt-dlp/yt-dlp/wiki/FAQ#how-do-i-pass-cookies-to-yt-dlp) extension to export your Youtube cookies as "cookies.txt".


2. Download the annotation file (["labels.json"](https://github.com/visurg-ai/LEMON/blob/main/labels.json)) and use the video downloader to download the original selected Youtube videos.

   ```bash
   $ python3 src/video_downloader.py --video-path '../labels.json' --output 'your path to store the downloaded videos' --cookies 'your YouTube cookie file'
   ```

3. Curate the downloaded original videos as LEMON video dataset. In detail, use the video_processor to classify each frame as either 'surgical' or 'non-surgical', then remove the beginning and end segments of non-surgical content from the videos, and mask the non-surgical regions in 'surgical' frames and the entire 'non-surgical' frames.

   ```bash
   $ python3 src/video_processor.py --input 'your original downloaded video storage path' --input-json '../labels.json' --output 'your path to store the curated videos and their corresponding frame annotation files' --classify-models 'frame classification model' --segment-models 'non-surgical object detection models'
   ```


4. Process the LEMON video dataset as LEMON image dataset (For foundation model pre-training).

   ```bash
   $ python3 src/create_lmdb_LEMON.py --video-folder 'your directory containing the curated videos and their corresponding frame annotation files' --output-json 'your path for the json file to verify the videos and labels alignment' --lmdb-path 'your lmdb storage path'
   ```

<br>
The video processing pipeline leading to the clean videos in the LEMON dataset is as follows:

<img src="https://github.com/user-attachments/assets/cb21d841-ad49-4834-b77e-dbc24fe6699e">


LemonFM model
-------------
You can download the LemonFM full checkpoint which contains backbone and projection head weights for both student and teacher networks at [🤗 LemonFM](https://huggingface.co/visurg/LemonFM).

**LemonFM pretraining:**


```bash
$ python3 -m torch.distributed.run --nproc_per_node=8 --nnodes=1 lemonfm/lemonfm.py --arch convnext_large --data_path 'LEMON dataset lmdb path' --output_dir 'your path to store the trained foundation model' --batch_size_per_gpu 40 --num_workers 10
```


**Fine-tuning LemonFM for surgical phase recognition:**


```bash
$ python3 downstream/train_phase_recognition_autolaparo.py --lr 1e-3 --opt adamW --nepochs 100 --bs 512 --cpdir 'path/to/store/checkpoint' --logdir 'path/to/store/log' --lmdb 'path/to/downstream_task/lmdb' --labels 'path/to/downstream_task/annotation' --seed 30 --pretrained-weights 'path/to/our/LemonFM.pth'
```

```bash
$ python3 downstream/test_phase_recognition_autolaparo.py --lmdb 'path/to/downstream_task/lmdb' --models 'path/to/your/cpdir' --labels 'path/to/downstream_task/annotation'
```



How to run our LemonFM foundation model to extract features from your video frames
----------------------------------------------------------------------------------

   ```python
   import torch
   from PIL import Image
   from model_loader import build_LemonFM

   # Load the pre-trained LemonFM model
   LemonFM = build_LemonFM(pretrained_weights = 'your path to the LemonFM')
   LemonFM.eval()

   # Load the image and convert it to a PyTorch tensor
   img_path = 'path/to/your/image.jpg'
   img = Image.open(img_path)
   img = img.resize((224, 224))
   img_tensor = torch.tensor(np.array(img)).unsqueeze(0).to('cuda')

   # Extract features from the image
   outputs = LemonFM(img_tensor)
   ```






