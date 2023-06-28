# SAM-Segmentation-Tool
Simple mask making using SAM, QT UI included.

### Installation
- torch cuda envrionment required
- PyQt5
- segment_anything
  ```
  pip install git+https://github.com/facebookresearch/segment-anything.git
  pip install opencv-python pycocotools matplotlib onnxruntime onnx
  ```
- download model weights to your BASE_DIR ![https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth]()

### Prepare Images
Put your Image that you want to make Masks for to BASE_DIR/Img

### Excute
- Excute SST.py
  ![image](https://github.com/jellyho/SAM-Segmentation-Tool/assets/20741606/a9b2233a-c11f-4df7-b815-d1de2362e351)
  
- Select images and click to create Masks
  ![image](https://github.com/jellyho/SAM-Segmentation-Tool/assets/20741606/a8028278-76f1-4715-9b33-f8afeb2e9721)

- Select Mask layer that you want to make
  
  ![image](https://github.com/jellyho/SAM-Segmentation-Tool/assets/20741606/def85590-2f50-4367-96ed-5a77af7e4afd)
  ![image](https://github.com/jellyho/SAM-Segmentation-Tool/assets/20741606/ca4e1157-200c-4bfd-9e63-2ff495011227)
  ![image](https://github.com/jellyho/SAM-Segmentation-Tool/assets/20741606/52587fa6-7bb0-4617-b228-5cc1bfe2f9e4)


- Then press Make, .npy file will be stored at BASE_DIR/Mask/(img_name)_mask.npy
  ![image](https://github.com/jellyho/SAM-Segmentation-Tool/assets/20741606/c29d5fe2-fdbb-4c05-9c86-3319a4477363)

