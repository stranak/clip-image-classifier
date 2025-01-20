### Install PyTorch, Transformers and the CLIP models

```.bash
python3 -m venv env;
source ./env/bin/activate;
pip install torch torchvision torchaudio;
pip install transformers ftfy regex tqdm;
pip install git+https://github.com/openai/CLIP.git
```

`cat images.filelist| python clip-classify.py`

On the first run the CLIP model used in the script (several options commented there) will be downloaded. It is about 900MB.

The model creates semantic models of categories from the category descriptions
in the `categories.tsv`, and matches those to the image. You can define any
categories (just labels) and the category descriptions (the semantically
relevant part). For details see the [CLIP model website](https://openai.com/index/clip/) and [the paper](https://arxiv.org/abs/2103.00020).
