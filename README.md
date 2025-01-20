### Install PyTorch, Transformers and the CLIP models
Use the nightly build of PyTorch hoping that MPS gets more and more operations implemented for Apple Sillicon GPU

```.bash
python3 -m venv env
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cpu
pip install transformers ftfy regex tqdm
pip install git+https://github.com/openai/CLIP.git
```

On the first run the CLIP model used in the script (several options commented there) will be downloaded. It is about 600MB.
