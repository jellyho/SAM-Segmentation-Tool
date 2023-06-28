# SAM-Segmentation-Tool
Simple mask making using SAM, qt ui included.

### Installation
- torch cuda envrionment required
- PyQt5
- segment_anything
  ```
  pip install git+https://github.com/facebookresearch/segment-anything.git
  pip install opencv-python pycocotools matplotlib onnxruntime onnx
  ```
- download model weights to your BASE_DIR ![](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth)

### Prepare Images
Put your Image that you want to make Masks for to BASE_DIR/Img

### Excute
- Excute SST.py
  ![image](https://github.com/jellyho/SAM-Segmentation-Tool/assets/20741606/e0082620-7241-49e0-aed5-f919a9c32794)

- Select Images and Click to Make Mask
  ![image](https://github.com/jellyho/SAM-Segmentation-Tool/assets/20741606/f2549d1b-65fa-4c8d-b1b7-f883a347c70a)

- Then Press Make, .npy file will be stored at BASE_DIR/Mask/(img_name)_mask.npy
