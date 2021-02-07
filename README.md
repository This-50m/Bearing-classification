## 此项目针对于美国西储大学数据集做的轴承故障多分类

#### 目录结构

dataset文件夹存放对应的训练数据

logs文件夹存放对应的模型及训练过程图片

temp文件夹存放的是数据分析过程图片

data.py =>数据读取

data_analysis.ipynb =>数据分析

inference.py => 数据测试

main.py => 训练

README.md => 注意事项

repvgg.py => 模型

requirements.txt => 依赖环境

```
>dataset
	>12k Drive End Bearing Fault Data
		105.mat
		106.mat
		...
	>12k Fan End Bearing Fault Data
	>48k Drive End Bearing Fault Data
	>Normal Baseline Data
>logs
>temp
data.py
data_analysis.ipynb
inference.py
main.py
README.md
repvgg.py
requirements.txt
```

#### 训练

```
python main.py --batch_size 8 --epochs 20 --num_classes 15 --learning_rate 0.01 --optimizer adam --logs logs/ --n_mels 64 --is_split True
```

如果num_classes = 3 对外圈故障 or 内圈故障 or 滚动体故障 做3分类

如果num_classes = 11 对外圈故障 or 内圈故障 or 滚动体故障 以及它们各自故障直径 做11分类

如果num_classes = 15 对外圈故障（包含3个方位） or 内圈故障 or 滚动体故障 以及它们各自故障直径 做15分类

如果 is_split = False 训练数据为10s （单个数据时长为10s左右）

如果 is_split = True 训练数据为1s（切片训练）

推荐 n_mels=64 is_split=True	or  	n_mels=256 is_split=False

#### 部署（可选）

```
python repvgg.py --weight_path './logs/xxx.pth' --save_path './logs/repvgg_deploy.pth'
```

#### 测试

```
python inference.py --sample_path 'dataset/12k Drive End Bearing Fault Data/199.mat' --weights './logs/xxx.pth' --num_classes 15 --n_mels 64 --is_split  True --deploy True
```

sample_path 为相对于当前目录下的数据地址 

#### 训练过程

15分类，数据时长1s

![](./logs/one_second_sample/Data(15)_VGGA0.png)

