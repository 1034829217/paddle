bit=16
bash test_tipc/test_train_inference_python.sh \
test_tipc/configs/hashnet_$bit/train_infer_python.txt \
lite_train_lite_infer

bit=32
bash test_tipc/test_train_inference_python.sh \
test_tipc/configs/hashnet_$bit/train_infer_python.txt \
lite_train_lite_infer

bit=48
bash test_tipc/test_train_inference_python.sh \
test_tipc/configs/hashnet_$bit/train_infer_python.txt \
lite_train_lite_infer

bit=64
bash test_tipc/test_train_inference_python.sh \
test_tipc/configs/hashnet_$bit/train_infer_python.txt \
lite_train_lite_infer
