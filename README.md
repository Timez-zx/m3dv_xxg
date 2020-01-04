test.py文件是用于模型浮现的主文件，其中有两个路径设置，路径一名字是npz_path,这个路径是测试集的路径，我在文件中初始化成了./test的相对路径
路径二的名字是weight_path,是权重文件的保存路径，使用的是./weight_used的相对路径，最终输出的是final.csv文件，是最后输出的结果，保存在相对路径里面  
train.py是训练过程中使用的文件  
model and evalue是交叉验证使用的代码  
weight_used保存的是我用的5个权重文件  
mylib是使用网络的补充文件
