data_path='./datasets/COCO2014/'
batch_size=64

bit=16
python main_multi_gpu.py \
--bit $bit \
--data-path $data_path \
--seed 0 \
--batch-size $batch_size \
--learning_rate 0.001

bit=32
python main_multi_gpu.py \
--bit $bit \
--data-path $data_path \
--seed 2000 \
--batch-size $batch_size \
--learning_rate 0.001

bit=48
python main_multi_gpu.py \
--bit $bit \
--data-path $data_path \
--seed 200 \
--batch-size $batch_size \
--learning_rate 0.001

bit=64
python main_multi_gpu.py \
--bit $bit \
--data-path $data_path \
--seed 2000 \
--batch-size $batch_size \
--learning_rate 0.001
