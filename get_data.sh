# Download data from kaggle
# train.csv
kg download -u [kaggle_username] -p [kaggle_password] -c 'planet-understanding-the-amazon-from-space' -f 'train_v2.csv.zip'
unzip train_v2.csv.zip
rm train_v2.csv.zip
mv train_v2.csv train.csv

# test.csv
kg download -u [kaggle_username] -p [kaggle_password] -c 'planet-understanding-the-amazon-from-space' -f 'sample_submission_v2.csv.zip'
unzip sample_submission_v2.csv.zip
rm sample_submission_v2.csv.zip
mv sample_submission_v2.csv test.csv

# train-jpg
kg download -u [kaggle_username] -p [kaggle_password] -c 'planet-understanding-the-amazon-from-space' -f 'train-jpg.tar.7z'
7z x train-jpg.tar.7z
rm train-jpg.tar.7z
tar xf train-jpg.tar
rm  train-jpg.tar

# test-jpg
kg download -u [kaggle_username] -p [kaggle_password] -c 'planet-understanding-the-amazon-from-space' -f 'test-jpg.tar.7z'
7z x test-jpg.tar.7z
rm test-jpg.tar.7z
tar xf test-jpg.tar
rm  test-jpg.tar

# test-jpg-additional
kg download -u [kaggle_username] -p [kaggle_password] -c 'planet-understanding-the-amazon-from-space' -f 'test-jpg-additional.tar.7z'
7z x test-jpg-additional.tar.7z
rm test-jpg-additional.tar.7z
tar xf test-jpg-additional.tar
rm  test-jpg-additional.tar

# train-tif
kg download -u [kaggle_username] -p [kaggle_password] -c 'planet-understanding-the-amazon-from-space' -f 'train-tif-v2.tar.7z'
7z x train-tif-v2.tar.7z
rm train-tif-v2.tar.7z
tar xf train-tif-v2.tar
rm  train-tif-v2.tar
mv train-tif-v2 train-tif

# test-tif
kg download -u [kaggle_username] -p [kaggle_password] -c 'planet-understanding-the-amazon-from-space' -f     'test-tif-v2.tar.7z'
7z x test-tif-v2.tar.7z
rm test-tif-v2.tar.7z
tar xf test-tif-v2.tar
rm test-tif-v2.tar
mv test-tif-v2 test-tif

