# Large-homework-code

包含《数字图像处理》课程作业所涉及到的所有代码及一些没有放入正文中的材料。

其中 
- `train_fashion_models.py` 为主要实验的代码
- `train_gce_noise.py` 为加入抗噪机制后的消融实验代码

实验结果文件说明：
- `main`：四个模型的主实验结果
- `head`：三个分类头的实验结果
- `noise`：未加入抗噪机制的实验结果
- `ablation_results`：消融实验的实验结果
- `复杂数据集`：实验三
- `深层网络复杂数据集`：实验四

补充实验的代码参见：[Google Colab Notebook](https://colab.research.google.com/drive/1qYSnc_-gfnTOZQWMOZNAYpwijIX7BY9Q?usp=sharing)

在 Colab 中您可以配置免费限额的 GPU 来完成实验。

`彩蛋` 为在完成文章撰写后对SE模块的进一步反思，文章中我们采用了传统的SE模块，但效果不尽如人意，同时我们也进一步实验验证，说明注意力机制确实起了效果，但我们还是希望能进一步探究在传统SE模块上能否做出一些基于文章已经构建的微小改进来取得更理想的结果，具体内容可进一步通过该文件夹查看。
