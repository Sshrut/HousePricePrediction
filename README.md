# House Price Prediction

This project is a comprehensive analysis and modeling of house prices based on various features, including both numerical and image data of houses. The project utilizes several machine learning and deep learning techniques to predict house prices.

## Table of Contents
- [Introduction](#introduction)
- [Installation](#installation)
- [Usage](#usage)
- [Dataset Description](#dataset-description)
- [Modeling Techniques](#modeling-techniques)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Introduction
This project aims to predict house prices using a dataset containing features like the number of bedrooms, bathrooms, area, and images of various parts of the house. The goal is to apply different machine learning and deep learning methodologies to build robust predictive models.

## Installation
To run this project, you need to have the following libraries installed:
- python
- numpy
- pandas
- seaborn
- matplotlib
- sklearn
- tensorflow
- google.colab (for running on Google Colab)
- cv2

You can install these using pip:

```bash
pip install numpy pandas seaborn matplotlib scikit-learn tensorflow opencv-python
```

## Usage
1. Clone the repository:
```bash
git clone https://github.com/yourusername/House-Price-Prediction.git
```
2. Navigate to the project directory:
```bash
cd House-Price-Prediction
```
3. Open the Jupyter Notebook:
```bash
jupyter notebook "HOUSE PRICE PREDICTION.ipynb"
```
4. Run all the cells to perform data analysis, visualization, and modeling.

## Dataset Description
The dataset used for this project consists of the following features:
- **Bedrooms**: Number of bedrooms
- **Bathrooms**: Number of bathrooms
- **Area**: Area of the house
- **Zipcode**: The zipcode of the house location
- **Price**: The price of the house (target variable)
- **Images**: Images of the house (bathroom, bedroom, frontal view, kitchen)

## Modeling Techniques
The project employs the following modeling techniques:
1. **Data Cleaning and Preprocessing**:
   - Handling missing values and outliers.
   - Normalizing and encoding features.
   - Combining image data into a unified dataset.

2. **Machine Learning Models**:
   - Fully Connected Neural Networks for numerical data.
   - Convolutional Neural Networks (CNNs) for image data.
   - Combining both models using TensorFlow's Functional API for a comprehensive prediction model.

## Results
The models were evaluated using Root Mean Squared Error (RMSE). The project's results demonstrate the effectiveness of combining numerical and image data for predicting house prices. Detailed visualizations and error metrics are provided in the notebook.

## Contributing
Contributions are welcome! Please fork the repository and create a pull request with your changes.

## License
This project is licensed under the MIT License.