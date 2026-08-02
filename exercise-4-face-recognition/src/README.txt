
How to Run the Scripts
--------------------------

python cvproj_exc/test.py --mode ident --video ../data/test_data/Alan_Ball/%04d.jpg

python cvproj_exc/test.py --mode cluster --video ../data/test_data/Alan_Ball/%04d.jpg

To enroll a new person to the recognition gallery:
python cvproj_exc/training.py --mode ident --video ../data/train_data/NAME/%04d.jpg --label NAME

To test recognition for a person:
python cvproj_exc/test.py --mode ident --video ../data/test_data/NAME/%04d.jpg
python cvproj_exc/test.py --mode ident --video ../data/test_data/Maria_Shriver/%04d.jpg

Not Needed.
To test clustering for a person:
python cvproj_exc/test.py --mode cluster --video ../data/test_data/NAME/%04d.jpg

For evaluation curves:
python cvproj_exc/dir_curve.py

To test the open-set recognition learning module:
python cvproj_exc/test_osr_learning.py


Final chosen parameters after running misc.py:
------------------------------------------------

tm_window_size = 25

tm_threshold = 0.5

k = 3

max_distance = 1.0

min_prob = 0.5
