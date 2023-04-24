## Code for Bit Allocation using Optimization Sec 5.2

## Requirements
* python==3.8
* pytorch==1.9
* tqdm
* compressai
* PIL

## Dataset for testing
* the dataset required is HEVC CTC Class BCDE and UVG, the sequences are as follows:
    * Class B: BasketballDrive_1920x1080_50.yuv, BQTerrace_1920x1080_60.yuv, Cactus_1920x1080_50.yuv, Kimono1_1920x1080_24.yuv, ParkScene_1920x1080_24.yuv
    * Class C: BasketballDrill_832x480_50.yuv, BQMall_832x480_60.yuv, PartyScene_832x480_50.yuv, RaceHorses_832x480_30.yuv
    * Class D: BasketballPass_416x240_50.yuv, BlowingBubbles_416x240_50.yuv, BQSquare_416x240_60.yuv, RaceHorses_416x240_30.yuv
    * Class E: FourPeople_1280x720_60.yuv, Johnny_1280x720_60.yuv, KristenAndSara_1280x720_60.yuv
    * UVG: Beauty_1920x1080_120fps_420_8bit_YUV.yuv, Jockey_1920x1080_120fps_420_8bit_YUV.yuv, YachtRide_1920x1080_120fps_420_8bit_YUV.yuv, Bosphorus_1920x1080_120fps_420_8bit_YUV.yuv, ReadySteadyGo_1920x1080_120fps_420_8bit_YUV.yuv, HoneyBee_1920x1080_120fps_420_8bit_YUV.yuv, ShakeNDry_1920x1080_120fps_420_8bit_YUV.yuv
    * they can be obtained from: https://hevc.hhi.fraunhofer.de/ and https://ultravideo.fi/#main
* the dataset need to be splitted into png images and cut into multiplier of 64, we recommend the following command with ffmpeg
    ```bash
    ffmpeg -pix_fmt yuv420p -s $W2x$H2 -i $Sequence_name.yuv -vf crop=$W2:$H2:0:0 $Sequence_name/im%03d.png
    ```
    * where H2, W2 are the cutted height and width into multiplier of 64. the path of png images should also be set when running test, see details in each folder
    * after that, set the root path of dataset in dataset.py line 19 and 59

## Pre-trained Baseline Model

* Download following pre-trained models and put them into ./Checkpoints folder:
* DVC: https://drive.google.com/drive/folders/1M54MPrAzaA0QVySnzUu9HZWx1bfIrTZ6), then rename them to ```DVC_{\lambda}.pth```;
* DCVC: https://onedrive.live.com/redir?resid=2866592D5C55DF8C!1198&authkey=!AGZwZffbRsVcjSQ&e=iMeykH, then rename them to ```DCVC_{\lambda}.pth```;
* HSTEM: https://1drv.ms/u/s!AozfVVwtWWYoiUAGk6xr-oELbodn?e=kry2Nk;
* About intra coding for DVC and DCVC, download pre-trained models of Cheng et al. 2020 from CompressaAI via 
    ```bash
    python -u ./Checkpoints/download_compressai_models.py
    ```

## Perform bit allocation: Proposed-Scalable and Proposed-Approx

* we provide code for proposed bit allocation method on DVC (Lu et al. 2019), DCVC (Li et al. 2021) and HSTEM (Li et al. 2022).
* For example, the following instruction is used for testing Proposed-Scalable with DVC on HEVC Class D dataset:
    ``` bash
    python main.py --test_model="DVC" --test_lambdas=(2048, 1024, 512, 256) --factor=-1 --overlap=0 --test_class="HEVC_D" --gop_size=10 --test_gop_num=1, --optimize_range=1
    ```
* the gop_size parameter should be set to 10 for HEVC BCDE dataset, and 12 for UVG dataset
* the optimize_range parameter is just the C parameter in Sec. 4.2. Setting it < gop size makes Proposed-Scalable, and set it == gop size makes Proposed-Approx. To reproduce the result in Tab. 1, use optimize_range = 2
* As for DCVC (Li et al. 2021) or HSTEM, you only need to change the setting `--test_model="DCVC"` or `--test_model="HSTEM"`.
* for more detailed CLI, use:
    ```bash
    python main.py --help
    ```
* If it occupies more memory than the maximum capacity of GPU, you can crop the input frame by changing `--factor` and `--overlap`. Here we don't provide the cropping mode for DVC because it's model size is very small. And if it's necessary to crop the input frame for DVC, please refer to the practice of DCVC or HSTEM in `main.py`.
