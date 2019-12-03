# Projeto AM 2019-2

1. [Description](https://www.cin.ufpe.br/~fatc/AM/Projeto-AM-2019-2.pdf)
2. [Paper](https://www.sciencedirect.com/science/article/pii/S0165011413002054)
3. [Dataset](http://archive.ics.uci.edu/ml/datasets/image+segmentation)

## Running
### Creating conda environment
```
conda create -n am python=3.6
conda activate am
pip intall -r requirements.txt
```
### Running question 1 code
```
python question1.py
```
While the code is running, the terminal will print every iteration the current error (J) value, the ajusted rand index and number of points in each cluster.

Results will be saved at `results/`  folder, with a folder for each table and a folder for each best error (J) and best rand. The folder with the highest number index will be the best value for every epoch. 
### Running question 2 code
```
python question2.py
```
### Running question 1 result_analyser code
```
python result_analyser [dir name or gt] [shape, rgb or view]
e.g.
python result_analyser.py results/rgb/rgb_J/rgb_1 view
```

## Files
- `labels.py` 
  - This program was used to convert the labels in the data to a number index
- `pca.py` 
  - outputs a 2D data and clusters visualization using principal component analysis
- `question1.py` 
  - outputs results for project's first question
- `question2.py` 
  - outputs results for project's second question
- `rand.py` 
  - outputs the rand index from the dataset labels and question1 output folder
- `result_analyser.py` 
  - outputs the rand index from the dataset labels and question1 output folder