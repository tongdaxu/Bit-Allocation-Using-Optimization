## Code for Bit Allocation using Optimization Sec 5.1 
* prerequisite
  * python
  * pytorch
  * numpy
  * pandas
  * compressai
## Dataset
* the dataset used is MNIST and can be automatically downloaded from torchvision
## Obtain a 2 level VAE base model for SAVI
* you may train the model by yourself
    ```bash
    python train.py --model=VAE_TwoLayer_Alt
    ```
* for more detailed cli information, do:
    ```bash
    python train.py --help
    ```
* or you can also directly adopt the pre-trained model we provide in ./VAE_TwoLayer/model.ckpt-3280.pt

## Perform SAVI on 2 level VAE
* with a pre-trained checkpoint, you can run SAVI directly
    ```bash
    python infer.py --resume=$VAE_Checkpoint
    ```
* for more detailed cli information, do:
    ```bash
    python infer.py --help
    ```
