cd sparsify
python sparsify.py --W 1024 --H 64 --line_spec 7 9 11 --nbeams 3 --split_file ../splits/eigen_zhou/train_files.txt
python sparsify.py --W 1024 --H 64 --line_spec 7 9 11 --nbeams 3 --split_file ../splits/eigen_zhou/val_files.txt
python sparsify.py --W 1024 --H 64 --line_spec 7 9 11 --nbeams 3 --split_file ../splits/eigen_full/train_files.txt
python sparsify.py --W 1024 --H 64 --line_spec 7 9 11 --nbeams 3 --split_file ../splits/eigen_full/val_files.txt
python sparsify.py --W 1024 --H 64 --line_spec 7 9 11 --nbeams 3 --split_file ../splits/eigen/test_files.txt
cd ..
python export_gt_depth.py --split eigen
python export_gt_depth.py --split 3beam
python gen2channel.py gen 3beam