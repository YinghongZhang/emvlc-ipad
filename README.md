#### 代码运行说明
##### 路径问题
这里分别有四个需要输入的路径
* dataset_path 存放数据的文件夹，默认是在 dataset\iiitd\images 下，这个文件夹里面分别有train, test文件夹，里面都是图片
* ground_truth_path 有标注的数据集，其实就是 dataset\iiitd 下的 trainset, testset .csv
* iris_location 虹膜的坐标，dataset\iiitd\osiris\osiris_coords.csv
* output_path 自己定

##### 版本问题
代码需要的是老版本的tensorflow等等，在colab中可以用以下的命令来切换到 tensorflow 1.15.0版本
在还没clone代码下来前，先运行%tensorflow 1.x然后再重启内核（重置colab）
```
%tensorflow 1.x
import tensorflow as tf
tf.__version__
```
我在代码中也加入了print tensorflow版本的语句，注意看代码的输出，如果tensorflow不是1.15.0版本会出bug

##### dataset加载文件
antispoofing\mcnns\datasets\livdet_iiitd_val.py
这个文件第68行可以修改读进去的数据集的大小，为了测试代码快点，我目前设为了500

#### Usage (raw images)
python antispoofing/mcnns/scripts/mcnnsantispoofing.py \
    --dataset 7 \
    --augmentation 0 \
    --dataset_path $DATASET_PATH \
    --ground_truth_path $GT_PATH \
    --iris_location $IRIS_LOCATION \
    --output_path $OUTPUT_PATH \
    --n_jobs 6 \
    --classification \
    --operation segment \
    --max_axis 260 \
    --bs 32 \
    --epochs $EPOCHS \
    --lr 0.001 \
    --decay 0.0 \
    --last_layer softmax \
    --loss_function 2 \
    --optimizer 1 \
    --reg 0.1 \
    --device_number $CUDA_VISIBLE_DEVICES

#### Usage (bsif images)
python antispoofing/mcnns/scripts/mcnnsantispoofing.py \
    --dataset 7 \
    --augmentation 0 \
    --dataset_path $DATASET_PATH \
    --ground_truth_path $GT_PATH \
    --iris_location $IRIS_LOCATION \
    --output_path $OUTPUT_PATH \
    --n_jobs 6 \
    --feature_extraction \
    --descriptor bsif \
    --desc_params "[3x3x8]" \
    --classification \
    --operation segment \
    --max_axis 260 \
    --bs 32 \
    --epochs $EPOCHS \
    --lr 0.001 \
    --decay 0.0 \
    --last_layer softmax \
    --loss_function 2 \
    --optimizer 1 \
    --reg 0.1 \
    --device_number $CUDA_VISIBLE_DEVICES

#### Usage (weighted voting)
python antispoofing/mcnns/scripts/mcnnsantispoofing_fusion.py \
    "${SCRIPT_PATH}/weighted_votingconfig.json" \
    --augmentation 0 \
    --weighttype acc \
    --device_number ${CUDA_VISIBLE_DEVICES}
