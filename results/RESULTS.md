# Results

results of the VinDr-Mammo Object Detection project. The results include confusion matrices, sample detections, and performance metrics for different tasks such as detection, BI-RADS classification, and breast density classification.



## BIRADS Classification Task
BIRADS classification task is handled by additional head on top of the backbone of the detection model. There are no BIRADS-1 findings in the pruned dataset, so the model is not trained to predict BIRADS-1. Evaluation results for the BI-RADS classification task is shown below:

### BIRADS Classification Report
![BI-RADS Classification Report](images/birads_report.png)


### BIRADS Confusion Matrix
![BI-RADS Confusion Matrix](images/birads_matrix.png)

## Density Classification Task
Density classification task is handled by additional head on top of the backbone of the detection model as well. Since density classes are more balanced in the pruned dataset, the model is trained to predict all density classes. Evaluation results for the breast density classification task is shown below:

### Density Classification Report
![Density Classification Report](images/density_report.png)

Below is another classification report for which the model is evaluated on the dataset that includes all density classes.

![Density Classification Report](images/density_report_whole.png)

### Density Confusion Matrix
![Density Confusion Matrix](images/density_matrix.png)

Below is another confusion matrix on the dataset that includes all density classes.
![Density Confusion Matrix](images/density_matrix_whole.png)



## Detection Task
RetinaNet with ResNet101 backbone is used for the detection task. The model is trained on the pruned dataset and evaluated on the test set. The model is evaluated using the following metrics:

### Detection Classification Report
![Detection Classification Report](images/findings_report.png)

As we can see from the support numbers the dataset is heavily imbalanced. And model is struggling to detect the minority classes. The model is able to detect the majority class with high precision and recall.

Below is another classification report evaluated on the same dataset however this time detection is considered valid if ground truth is included in top 5 predictions.
![Detection Classification Report](images/top_5_findings_report.png)

### Detection Confusion Matrix
![Detection Confusion Matrix](images/findings_matrix.png)

Below is the AUROC curve for the detection task. Works well for Suspicious Calcification and moderately for Mass. However, the model is not able to detect the other classes well. Likely due to the class imbalance in the dataset.

![AUROC Curve](images/AUROC_findings.png)

## Sample Detections

### Detection Task
Below are some detections from the test set:

![Sample Detection 1](images/250.png)
![Sample Detection 1](images/250_gt.png)


![Sample Detection 2](images/120.png)
![Sample Detection 2](images/120_gt.png)


![Sample Detection 3](images/60_3.png)
![Sample Detection 3](images/60_gt.png)


![Sample Detection 4](images/15.png)
![Sample Detection 4](images/15_gt.png)


![Sample Detection 5](images/8.png)
![Sample Detection 5](images/8_gt.png)
