# PyTorch Template for DL projects

<p align="center">
  <a href="#installation">Installation</a> •
  <a href="#how-to-use">How To Use</a> •
  <a href="#useful-links">Useful Links</a> •
  <a href="#credits">Credits</a> •
  <a href="#license">License</a>
</p>

## About

This repository implements HiFiGAN vocoder model for Text-to-Speech task for the Deep Learning in Audio course at Higher School of Economics.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install --upgrade hydra-core
```

The last step breaks dependencies but we have to do it because of this stupid fairseq package.

## How To Use

To train a model, run the following command:

```bash
python train.py
```

To use DDP, run

```bash
accelerate config
accelerate launch train.py
```

You can also download pretrained model with

```bash
gdown --fuzzy "https://drive.google.com/file/d/1iYcckQwjIOjdgVbGN3UDRv4mg8DHeFtE/view?usp=sharing"
```

To run inference, use the following command.

```bash
python synthesize.py inferencer.from_pretrained="<PATH_TO_CHECKPOINT>.pth" inferencer.save_path="<SAVE_NAME>" datasets.val.root="<PATH_TO_DATASET>"
```

The dataset should contain a `transcriptions` folder with `.txt` files. After that, output files will be stored in `data/saved/<SAVE_NAME>/val` directory.

You can also pass the text you want to synthesize using the following command.

```bash
python synthesize.py datasets=inline inferencer.from_pretrained="<PATH_TO_CHECKPOINT>.pth" inferencer.save_path="<SAVE_NAME>" datasets.val.transcription="<TEXT_TO_SYNTHESIZE>"
```

After that, the output file will be saved as `data/saved/<SAVE_NAME>/val/output.wav`.
