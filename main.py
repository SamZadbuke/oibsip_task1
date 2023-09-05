import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.datasets import load_iris
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Analysis Section
iris = load_iris()
X = iris.data
y = iris.target

iris_df = pd.DataFrame(X, columns=iris.feature_names)
iris_df["species"] = y

species_names = {0: "Setosa", 1: "Versicolor", 2: "Virginica"}
iris_df["species"] = iris_df["species"].map(species_names)

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)

rf_classifier.fit(X_train, y_train)

# Streamlit app section
def main():
    # Information Section
    st.title("Iris Flower Species Classifier")
    st.write(
        "Welcome to the Iris Flower Species Classifier! This tool employs a Random Forest model to categorize Iris flowers into three distinct species: Setosa, Versicolor, and Virginica. The dataset includes four key features: sepal length, sepal width, petal length, and petal width."
    )
    st.write(
        "The Iris dataset is a foundational dataset in machine learning and statistics. It was introduced by the renowned British statistician and biologist, Ronald A. Fisher, in 1936. The dataset consists of 150 Iris flower samples, equally divided into three species categories:"
    )
    st.write("- Setosa (0)")
    st.write("- Versicolor (1)")
    st.write("- Virginica (2)")
    st.header("Model Accuracy")
    accuracy = accuracy_score(y_test, rf_classifier.predict(X_test))
    st.write(f"Model Accuracy: {accuracy:.2f}")

    # Graph Section
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.scatter(range(len(y_test)), y_test, label="Actual", color="blue", marker="o")
    plt.xlabel("Samples")
    plt.ylabel("Class")
    plt.title("Actual")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.scatter(
        range(len(y_test)),
        rf_classifier.predict(X_test),
        label="Predicted",
        color="orange",
        marker="x",
    )
    plt.xlabel("Samples")
    plt.ylabel("Class")
    plt.title("Predicted")
    plt.legend()

    plt.tight_layout()
    st.pyplot(plt)

    # Prediction Section
    st.header("Enter Sepal and Petal Measurements")
    sepal_length = st.number_input(
        "Sepal Length", min_value=0.1, max_value=10.0, value=5.4, step=0.1
    )
    sepal_width = st.number_input(
        "Sepal Width", min_value=0.1, max_value=10.0, value=3.4, step=0.1
    )
    petal_length = st.number_input(
        "Petal Length", min_value=0.1, max_value=10.0, value=1.7, step=0.1
    )
    petal_width = st.number_input(
        "Petal Width", min_value=0.1, max_value=10.0, value=0.2, step=0.1
    )

    if st.button("Predict"):
        input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
        input_data_scaled = scaler.transform(input_data)

        predicted_label = rf_classifier.predict(input_data_scaled)[0]
        predicted_species = species_names[predicted_label]

        st.header("Predicted Species:")
        st.write(f"*{predicted_species}*")

        proba_predictions = rf_classifier.predict_proba(input_data_scaled)[0]
        proba_species = {
            species_names[i]: proba for i, proba in enumerate(proba_predictions)
        }

        st.bar_chart(proba_species)

if __name__ == "__main__":
    main()
