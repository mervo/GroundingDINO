**Installation:**

1.Clone repository and sub-modules from GitHub.

```bash
git clone --recurse-submodules -j4 https://github.com/IDEA-Research/GroundingDINO.git
```

2. Change the current directory to the GroundingDINO folder.

```bash
cd GroundingDINO/
```

3. Install the required dependencies in the current directory. Choose a PyTorch version that is the same as your Cuda version. Current instructions are for Cuda 12.1.

```bash
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu121
pip install -e .
```

4. Download pre-trained model weights.

```bash
mkdir weights
cd weights
wget -q https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
wget -q https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha2/groundingdino_swinb_cogcoor.pth
cd ..
```
