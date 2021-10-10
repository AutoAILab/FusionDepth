cd sparsify
python sparsify.py --W 1024 --H 64 --random_sample 100 --split_file ../splits/eigen_zhou/train_files.txt
python sparsify.py --W 1024 --H 64 --random_sample 100 --split_file ../splits/eigen_zhou/val_files.txt
python sparsify.py --W 1024 --H 64 --random_sample 100 --split_file ../splits/eigen_full/train_files.txt
python sparsify.py --W 1024 --H 64 --random_sample 100 --split_file ../splits/eigen_full/val_files.txt
python sparsify.py --W 1024 --H 64 --random_sample 100 --split_file ../splits/eigen/test_files.txt
cd ..
python export_gt_depth.py --split r100
python gen2channel.py gen r100