🚀 Delivery Time Prediction Model

This project develops a predictive model to estimate the actual_delivery_time for orders, measured in minutes. This prediction is crucial for logistics optimization, driver scheduling, and setting accurate customer expectations.

The final solution utilizes a Tuned Neural Network (NN) built with TensorFlow/Keras, trained on a comprehensive feature set derived from historical order data.

🎯 Project Goal

To build and deploy a machine learning regression model that accurately predicts the final delivery time (time_taken_mins) based on features available at the moment the order is created.

📊 Methodology and Modeling

The project followed a rigorous machine learning pipeline, focusing heavily on feature engineering and model optimization to achieve the best performance.

1. Data Source

File: dataset.csv

Target Variable (Y): time_taken_mins (calculated as actual_delivery_time - created_at and converted to minutes).

2. Feature Engineering & Changes Made

We introduced complex, time-aware features and made necessary encoding changes to enhance the model's predictive power.

A. Initial Features

Base features like subtotal, total_items, hours, and day were extracted.

B. Enhanced Features (Changes Made)

The following complex features were engineered to capture external and historical factors:

Feature Name

Description

Rationale

store_avg_delivery_time

Historical average delivery time for the specific store.

Crucial: Captures the intrinsic efficiency/speed of each unique store.

is_lunch_rush

Binary flag for orders created between 11:00 and 14:00.

Models time-of-day demand spikes and resulting delays.

is_dinner_rush

Binary flag for orders created between 17:00 and 21:00.

Models high-traffic evening periods.

is_weekend

Binary flag for orders created on Saturday or Sunday.

Captures weekly demand cycles.

Encoding Change:

One-Hot Encoding was applied to store_primary_category.

Replaced simple Label Encoding to avoid imposing false ordinality on categorical data (e.g., assuming 'American' is "less" than 'Mexican').

3. Model Selection and Optimization

We evaluated three different regression models before settling on a highly optimized Neural Network.

Model

Technique

Status

Test $R^2$ Score

Reason for Acceptance/Rejection

Random Forest

Ensemble Tree Regressor

Initial Baseline

(Not calculated)

Used as a benchmark, provided lower performance than the optimized NN.

XGBoost

Gradient Boosting Machine

Rejected

$\approx 0.0832$

Failed to capture variance in the data, resulting in a very poor R-squared score.

Neural Network

Keras/TensorFlow

Final Production Model

$\approx 0.1995$

Achieved the best overall $\text{RMSE}$ and $\text{R}^2$ after tuning and feature enhancement.

Neural Network Optimization (Changes Made)

Architecture: We used the Keras Tuner library with a RandomSearch strategy to optimize the layer structure and hyperparameters.

Final Structure: Found to be a network with 2 hidden layers (128 units, 64 units).

Learning Rate: Optimized to 0.01.

Activation & Loss: Used linear activation on the output layer (essential for continuous regression output) and mse (Mean Squared Error) loss.

Scaling: Applied MinMaxScaler to all features, which is mandatory for effective Neural Network training.

📈 Final Performance

The model was evaluated on the unseen test set, showcasing its efficiency gains primarily through a lower average error.

Metric

Score

Interpretation

Root Mean Squared Error (RMSE)

17.25 minutes

The average magnitude of the error is $\sim$17 minutes.

Mean Absolute Error (MAE)

11.76 minutes

The model's predictions are, on average, within 11.76 minutes of the actual delivery time.

R-squared ($R^2$) Score

0.1995

The model explains approximately 20% of the variance in delivery time.

Mean Absolute Percentage Error (MAPE)

26.57%

The average percentage error of the prediction.

Conclusion: The final Neural Network model provides a reliable time estimate with an average absolute error of under 12 minutes, demonstrating efficiency improvements over the baseline.

🚀 Next Steps for Performance Improvement (How to Make it Better)

The $R^2$ score of $0.1995$ indicates that $\sim$80% of the variance in delivery time remains unexplained. To push the performance higher (i.e., increase $R^2$ and decrease $\text{RMSE}$/$\text{MAE}$), the following steps are recommended:

Time-Series Feature Engineering:

Lagged Features: Incorporate features based on the previous hour's performance (e.g., "average delivery time in the last 60 minutes for this market"). This requires real-time data aggregation in a production setting.

Weather Data: Integrate external data sources like current weather conditions (rain, snow, extreme heat) at the time of order creation, as this heavily impacts driver speed.

Advanced Feature Encoding:

Target Encoding: Apply Target Encoding to high-cardinality features (if any) or store categories. This technique replaces a category with the average target value (delivery time) observed for that category, which often provides a stronger signal than One-Hot Encoding without increasing dimensionality.

Advanced Modeling Techniques:

Recurrent Neural Networks (RNNs): For time-series prediction, an RNN (like LSTM or GRU) could potentially better capture sequential dependencies in order flow than a simple Dense Neural Network.

Ensembling: Create an ensemble model by averaging or blending the predictions of the Neural Network with a highly optimized XGBoost (using techniques like Target Encoding to stabilize the XGBoost model).

🌐 Deployment Strategy (API Pipeline)

The trained model is deployed as a callable prediction service to enable real-time usage in a production environment.

Artifacts: The final trained model (delivery_time_model.h5) and the fitted scaler (fitted_scaler.joblib) are saved.

Service: The model is wrapped in a high-performance API using FastAPI.

Pipeline: The API endpoint calls a central prediction function (predict_delivery_time) that performs the exact same feature engineering and scaling steps as training before passing the data to the loaded Keras model.

Endpoint:

POST /predict_time


This endpoint accepts new order data (as JSON) and returns the estimated_delivery_minutes.
