# Projeto AM 2019-2

1. [Description](Projeto-AM-2019-2.pdf)
2. [Paper](https://www.sciencedirect.com/science/article/pii/S0165011413002054)
3. [Dataset](http://archive.ics.uci.edu/ml/datasets/image+segmentation)

Group 1

Students
- [Felipe Bezerra Martins](https://github.com/felipemartins96)
- [George Carvalho](https://github.com/geocarvalho)

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

Results used in the report are available at the `report_results`
### Running question 2 code
```
python question2.py
```
Results will be printed at the terminal (Wilcoxon Results, Mean Accuracy and Confidence Interval) and an accuracy histogram graph will be created named `hist_sgb.png`
### Running question 1 result_analyser code
```
python result_analyser [dir name or gt] [shape, rgb or view]
e.g.
python result_analyser.py results/rgb/rgb_J/rgb_1 view
```
Will output for the selected results folder the error (J), the adjusted rand index and the number of points in each cluster. It will create a PCA visualization image at the results folder

[dir name or gt]: input the directory for the results folder you want to analyse or use `gt` to output the ground truth results.

[shape, rgb or view]: to select which table features to be used for the pca visualization image, view being all features.

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
<<<<<<< HEAD
  - outputs the rand index from the dataset labels and question1 output folder
=======
  - outputs results data and a 2D cluster visualization using principal component analysis
>>>>>>> de139b6e70ae51e4a8d25f7a31f264d5b1fa5983
