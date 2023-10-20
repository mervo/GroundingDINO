**Installation:**

1. Clone repository and sub-modules from GitHub.

```bash
git clone --recurse-submodules -j4 https://github.com/mervo/GroundingDINO.git
```

2. Change the current directory to the GroundingDINO folder.

```bash
cd GroundingDINO/
```

3. Install the required dependencies in the current directory. Choose a PyTorch version that is the same as your Cuda version. Current instructions are for Cuda 12.1.

```bash
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu121
pip install -e .
pip install ./video_utils
pip install sahi
```

4. Download pre-trained model weights.

```bash
mkdir weights
cd weights
wget -q https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
wget -q https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha2/groundingdino_swinb_cogcoor.pth
cd ..
```


**Running Inference on Gradio App:**

1. Run container with volume mount and port forwarding to localhost
```bash 
docker run -it --rm -v ~/Desktop/GroundingDINO:/home/GroundingDINO -p 7579:7579 <IMAGE_NAME:TAG>  
```
2. Spin up Gradio App, should use GPU if torch.cuda.is_available() is True
```bash
python3 demo/gradio_app.py 
```
