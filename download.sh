#wget https://cloud.tsinghua.edu.cn/f/578e3d1d8851483090ef/?dl=1 -O ./checkpoints/'ACRLSD_2D(hemi+fib25)_Best_in_val.model'
#wget https://cloud.tsinghua.edu.cn/f/7792384dd8524fcda6a2/?dl=1 -O ./checkpoints/'ACRLSD_3D(hemi+fib25)_Best_in_val.model'
#wget https://cloud.tsinghua.edu.cn/f/3732bea32b1046e487ca/?dl=1 -O ./checkpoints/'segEM2d(hemi+fib25)_Best_in_val.model'
#wget https://cloud.tsinghua.edu.cn/f/3269eaab609244009ea2/?dl=1 -O ./checkpoints/'segEM2d(hemi+fib25)wloss-1_Best_in_val.model'
#wget https://cloud.tsinghua.edu.cn/f/6259a1b96da54d4d9096/?dl=1 -O ./checkpoints/'segEM3d_trace(hemi+fib25)_Best_in_val.model'


wget https://cloud.tsinghua.edu.cn/f/5ec8cb9dc86b4d1f95ba/?dl=1 -O ./data/hemibrain_test_roi_1.zip
unzip ./data/hemibrain_test_roi_1.zip -d ./data/
rm ./data/hemibrain_test_roi_1.zip
