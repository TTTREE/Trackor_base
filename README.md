# Trackor_base
Implementation of "Tracking without bells and whistles” and the multi-object tracking "Tracktor" .

本项目基于  
[phil-bergmann/tracking_wo_bnw](https://github.com/phil-bergmann/tracking_wo_bnw)

# 修改内容

1.仅使用基础的trackor，去掉Motion和Reid模块;  
2.内含Frcnn模块，无需install；  
3.可处理视频/实时摄像头数据。

# Pre-trained model
[Download from here](https://drive.google.com/open?id=1E0seC4zSdAsKUNScv4M0eAu7fG_v65_Q)  
Extract in *output* directory.

# 使用方法

1.`git clone https://github.com/TTTREE/Trackor_base.git`  
`cd Trackor_base`  
根据需要可修改 experiments/cfgs/tracktor.yaml  
2.`python temp_main.py`

本项目仅供学习参考，后续会改进。

