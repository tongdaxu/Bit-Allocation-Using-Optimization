import urllib.request

# The model weights of intra coding come from CompressAI.
root_url = "https://compressai.s3.amazonaws.com/models/v1/"

model_names = [
           "bmshj2018-hyperprior-ms-ssim-3-92dd7878.pth.tar",
           "bmshj2018-hyperprior-ms-ssim-4-4377354e.pth.tar",
           "bmshj2018-hyperprior-ms-ssim-5-c34afc8d.pth.tar",
           "bmshj2018-hyperprior-ms-ssim-6-3a6d8229.pth.tar",
           "cheng2020-anchor-3.pth.tar",
           "cheng2020-anchor-4.pth.tar",
           "cheng2020-anchor-5.pth.tar",
           "cheng2020-anchor-6.pth.tar",
]

for model in model_names:
    print(f"downloading {model}")
    urllib.request.urlretrieve(root_url+model, model)