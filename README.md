# Iris Flower Species Classifier

![Iris Flower](https://www.edureka.co/blog/wp-content/uploads/2018/05/Iris-species.png)

Welcome to the Iris Flower Species Classifier! This interactive tool employs a Random Forest model to categorize Iris flowers into three distinct species: Setosa, Versicolor, and Virginica. The dataset includes four key features: sepal length, sepal width, petal length, and petal width.

## Table of Contents

- [About](#about)
- [Model Accuracy](#model-accuracy)
- [Graphs](#graphs)
- [Prediction](#prediction)
- [Getting Started](#getting-started)
- [Dependencies](#dependencies)
- [Installation](#installation)
- [Usage](#usage)

## About

The Iris dataset is a foundational dataset in machine learning and statistics. It was introduced by the renowned British statistician and biologist, Ronald A. Fisher, in 1936. The dataset consists of 150 Iris flower samples, equally divided into three species categories:

- Setosa (0)
- Versicolor (1)
- Virginica (2)

## Model Accuracy

The accuracy of the Random Forest classifier on the test data is displayed. The model's accuracy is calculated and shown using the `accuracy_score` function from scikit-learn.

## Graphs

This section displays two scatter plots side by side:

- The left plot shows the actual class labels (species) of the test data samples.
- The right plot shows the predicted class labels produced by the Random Forest classifier for the same test data samples. This visual comparison helps assess the model's performance.

## Prediction

In the prediction section, you can enter measurements for an Iris flower:

- Sepal Length
- Sepal Width
- Petal Length
- Petal Width

When you click the "Predict" button, the application:

- Scales the input data using the same Min-Max scaler used for training.
- Utilizes the trained Random Forest classifier to predict the species of the Iris flower based on the input measurements.
- Displays the predicted species and a bar chart showing the probability distribution of all three species.

## Getting Started

To run this Streamlit app locally, follow these steps:

### Dependencies

Ensure you have the following Python packages installed:

- pandas
- numpy
- matplotlib
- streamlit
- scikit-learn

You can install these packages using pip:

```bash
pip install pandas numpy matplotlib streamlit scikit-learn
```

### Installation

1. Clone this repository to your local machine:

```bash
git clone https://github.com/yourusername/iris-flower-classifier.git
```

2. Change to the project directory:

```bash
cd iris-flower-classifier
```

### Usage

Run the Streamlit app using the following command:

```bash
streamlit run app.py
```

Access the app in your web browser by navigating to the URL provided by Streamlit (usually http://localhost:8501). Interact with the app to predict Iris flower species and explore model accuracy.

---

Enjoy classifying Iris flowers with this handy tool! If you have any questions or feedback, feel free to reach out. Happy coding!
