# Deep-Learning-Assignment
ML4Science Assignment

*Problem statement: Given an image (and its label value) predict the sum of the digits in the image. 

![problem_statement](https://user-images.githubusercontent.com/24211231/212101787-250f8516-9ccb-4262-8abe-ead5b249b2cc.png)

Steps to run this project

1. Create **`/dataset`** dir in root dir
2. Extract data.rar in `/dataset` dir
3. Also extract **mnist.zip** dataset in **/dataset** dir
4. Create **`/customDataset`** dir in root dir to store new generated data
5. Run `$ python3 customDatasetGenerator.py` to generate new dataset. (It will add 4 files to customDataset Dir)
6. Run **train.ipynb** notebook to train model.

Once Done above 5 step, Root dir should look like this.

-------------------------------------------------------
root
  |-> customDatasetGenerator.py
  |-> dataset
          |-> data0.npy
          |-> lab0.npy
          |-> data1.npy
          |-> lab1.npy
          |-> data2.npy
          |-> lab2.npy
          |-> train.csv
          |-> test.csv
  |-> customDataset
          |-> trainX.npy
          |-> trainY.npy
          |-> testX.npy
          |-> testY.npy
  |-> train.ipynb
  |-> data.rar
  |-> mnist.zip

-------------------------------------------------------
