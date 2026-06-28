import os
import gdown

os.makedirs("index", exist_ok=True)

gdown.download(
    "https://drive.google.com/file/d/13m1LhWzM9UCcHMkcgNvLr8CBCLifh1gM/view?usp=sharing",
    "index/metadata.parquet",
    quiet=False,
)

gdown.download(
    "https://drive.google.com/file/d/17GIpJsqd6MUlWSvcq1ZxvWuobfLnJvZN/view?usp=sharing",
    "index/index.faiss",
    quiet=False,
)