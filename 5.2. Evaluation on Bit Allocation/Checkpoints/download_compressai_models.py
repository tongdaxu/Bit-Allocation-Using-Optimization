import urllib.request

# The model weights of intra coding come from CompressAI.
root_url = "https://compressai.s3.amazonaws.com/models/v1/"

model_names = [
           "cheng2020-anchor-3.pth.tar",
           "cheng2020-anchor-4.pth.tar",
           "cheng2020-anchor-5.pth.tar",
           "cheng2020-anchor-6.pth.tar",
]

for model in model_names:
    print(f"downloading {model}")
    urllib.request.urlretrieve(root_url+model, model)