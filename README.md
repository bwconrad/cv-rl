# Computer Vision Training using Policy Optimization

This repo contains experiments using policy optimization to train non-differentiable objectives for computer vision tasks. These experiments are inspired by the work "[Tuning computer vision models with task rewards](https://arxiv.org/abs/2302.08242)".

## Requirements 

- Python 3.10+
- `pip install -r requirements.txt`

## Experiments

### Binary Segmentation - F1 Score

In this experiment, a binary segmentation model is trained using the REINFORCE algorithm to optimize for the F1 score.
The experiment is performed on the [TNBC](https://zenodo.org/record/1175282#.YMisCTZKgow) using a U-Net with a ResNet-18 encoder.

The baseline cross-entropy model can be run with the following:
```python
train_segmentation.py fit --trainer.accelerator gpu --trainer.devices 1 --trainer.precision 16-mixed --data.root data/tnbc --data.batch_size 8 --trainer.max_steps 1000 --trainer.val_check_interval 100 --model.lr 0.0005 --model.schedule cosine 
```

The F1 score model can be run with the following:
```python
train_segmentation_reinforce.py fit --trainer.accelerator gpu --trainer.devices 1 --trainer.precision 16-mixed --data.root data/tnbc --data.batch_size 8 --trainer.max_steps 1000 --trainer.val_check_interval 100 --model.lr 0.0005 --model.schedule cosine 
```

Fine-tuning the cross-entropy model on the F1 score can be run with the following:
```python
train_segmentation_reinforce.py fit --trainer.accelerator gpu --trainer.devices 1 --trainer.precision 16-mixed --data.root data/tnbc --data.batch_size 8 --trainer.max_steps 1000 --trainer.val_check_interval 100 --model.lr 0.00005 --model.schedule cosine  --model.weights output/weights-ce.ckpt
```

#### Results

| Objective | F1 | Dice |
|:-:|:-:|:-:| 
| Cross-entropy | 0.7729 | 0.7585 |
| F1 | 0.4152 | 0.4369 |
| Cross-entropy -> F1 | 0.7615 | 0.7611 |


