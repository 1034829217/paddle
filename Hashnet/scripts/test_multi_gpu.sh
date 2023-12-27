data_path='./datasets/COCO2014/'
batch_size=64

bit=16
python main_multi_gpu.py \
--bit $bit \
--eval \
--pretrained output/weights_$bit \
--data-path $data_path \
--batch-size $batch_size

bit=32
python main_multi_gpu.py \
--bit $bit \
--eval \
--pretrained output/weights_$bit \
--data-path $data_path \
--batch-size $batch_size

bit=48
python main_multi_gpu.py \
--bit $bit \
--eval \
--pretrained output/weights_$bit \
--data-path $data_path \
--batch-size $batch_size

bit=64
python main_multi_gpu.py \
--bit $bit \
--eval \
--pretrained output/weights_$bit \
--data-path $data_path \
--batch-size $batch_size
