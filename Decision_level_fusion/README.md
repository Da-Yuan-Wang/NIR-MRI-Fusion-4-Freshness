# Decision Level Fusion Project

This project implements decision-level fusion of four base classifiers, including three methods: voting fusion, averaging fusion, and weighted fusion.

## Project Structure

```
decision_level_fusion/
├── data/                     # Base classifier prediction results
├── src/                      # Source code
│   └── decision_fusion.py    # Decision fusion main program
├── results/                  # Fusion results output
└── README.md                 # Documentation
```

## Base Classifiers and Accuracy

1. NIR-Stacking: 0.9792
2. MRI-Stacking: 0.9557
3. NIR-MRI-GLCM-Stacking: 0.9896
4. NIR-MRI-Conv-Stacking: 0.9583

## Fusion Methods

### 1. Voting Fusion
Majority voting of predictions from four base classifiers, with the class receiving the most votes as the final prediction result.

### 2. Averaging Fusion
Averaging the probabilities of each class output by the four base classifiers, with the class with the highest probability as the final prediction result.

### 3. Weighted Fusion
Weighted fusion based on the accuracy of base classifiers, where classifiers with higher accuracy have greater weights.

Weight calculation formula:
```
Weight = Classifier Accuracy / Sum of All Classifier Accuracies
```

## Usage

1. Ensure the prediction results of the four base classifiers are placed in the `data/` directory
2. Run the fusion program:
   ```
   cd src
   python decision_fusion.py
   ```
3. Fusion results will be saved in the `results/` directory

## Output Files

1. `decision_fusion_voting.csv` - Voting fusion results
2. `decision_fusion_averaging.csv` - averaging fusion results
3. `decision_fusion_weighted.csv` - Weighted fusion results

Each result file contains the following fields:
- true_label: Ground truth label
- true_label_name: Ground truth label name
- fusion_prediction: Fusion prediction result
- fusion_prediction_name: Fusion prediction result name