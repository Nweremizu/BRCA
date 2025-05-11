# Breast Cancer Prediction System

This project implements a machine learning-based breast cancer prediction system using the Wisconsin Breast Cancer dataset. The system includes both analysis notebooks and a web application for making predictions.

## Project Structure

```
.
├── breast_cancer_classification.ipynb    # Classification models analysis
├── breast_cancer_clustering.ipynb       # Clustering analysis
├── web_app/                             # Web application
│   ├── app.py                          # Flask application
│   ├── requirements.txt                # Python dependencies
│   ├── models/                         # Trained model storage
│   └── templates/                      # HTML templates
│       └── index.html                  # Main web interface
└── README.md                           # This file
```

## Setup and Installation

1. Create a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. Install dependencies:
   ```bash
   cd web_app
   pip install -r requirements.txt
   ```

3. Run the Jupyter notebooks to train and save the best model:
   - Open and run `breast_cancer_classification.ipynb` to train classification models
   - Open and run `breast_cancer_clustering.ipynb` to perform clustering analysis
   - The best model will be saved in `web_app/models/`

4. Start the web application:
   ```bash
   cd web_app
   python app.py
   ```

5. Open your web browser and navigate to `http://localhost:5000`

## Features

### Classification Analysis
- Implements multiple classification algorithms:
  - Logistic Regression
  - Support Vector Machine (SVM)
  - Decision Tree Classifier
  - Random Forest Classifier
- Includes model comparison and evaluation
- Feature importance analysis

### Clustering Analysis
- Implements multiple clustering algorithms:
  - K-means Clustering
  - Agglomerative Clustering
- Includes methods for finding optimal number of clusters:
  - Elbow Method
  - Silhouette Analysis
  - Gap Statistic
  - Hierarchical Clustering (Dendrogram)

### Web Application
- User-friendly interface for inputting tumor characteristics
- Real-time prediction of tumor classification
- Probability scores and interpretation of results
- Responsive design for desktop and mobile devices

## Usage

1. Open the web application in your browser
2. Enter the tumor characteristics in the form
3. Click "Predict" to get the classification result
4. View the prediction, confidence score, and interpretation

## Notes

- The web application requires a trained model to be present in the `web_app/models/` directory
- All predictions should be verified by healthcare professionals
- The system is for educational and research purposes only

## License

This project is licensed under the MIT License - see the LICENSE file for details. 