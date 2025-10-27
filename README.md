# Porter NN Report

This repository contains an analysis of the dataset provided in the report and outlines recommendations for improving data quality and insights for modeling delivery times.

## Dataset Overview

The dataset contains 176,248 records with numerical and categorical features related to delivery time and store properties. Below are some key statistics:

- **`time_taken_mins`:**
  - Average: 47.76 minutes
  - Standard Deviation: 27.65 minutes
  - Range: ~1.68 minutes to 6,231.31 minutes (possible outliers)
  
- **`subtotal`:**
  - Average: ₹2,696.50
  - Range: ₹0 to ₹26,800

- **`total_items`:**
  - Average: 3.2 items/order
  - Range: 1 to 411 items (possible extreme values)

### Anomalies in Data
- **Negative Values:** Found in `min_item_price` and `total_onshift_partners`.
- **Potential Outliers:** Noted in `time_taken_mins` and `total_items`.

---

## Correlation Analysis

### Strongest Relationships
1. **`subtotal` and `total_items`**: Moderate positive correlation (0.555), indicating larger orders have higher subtotals.
2. **`num_distinct_items` and `total_items`**: High correlation (0.758), showing a strong relationship between order variety and quantity.

### Weak Relationships
- Most features (e.g., `store_primary_category_encoded`) have weak correlations with `time_taken_mins`.
- **`subtotal` and `time_taken_mins`**: Low positive correlation (0.144).

### Negative Correlations
- **`hours` and `total_onshift_partners`**: Moderate negative correlation (-0.375), suggesting fewer on-shift partners during off-peak hours.

---

## Recommendations for Next Steps

### Data Cleaning
1. Investigate and address:
   - Negative values in `min_item_price` and `total_onshift_partners`.
   - Outliers in `time_taken_mins` and `total_items`.

### Feature Importance
2. Focus on:
   - Correlated features such as `subtotal` and `num_distinct_items` for model training.
   - External data (e.g., traffic, weather) for enhanced modeling.

### Visualization Insights
3. Analyze:
   - Delivery time trends by hour (e.g., mornings, evenings) and day (weekends vs. weekdays).
   - Patterns using heatmaps and scatterplots.

### Modeling Strategy
4. Explore:
   - Hyperparameter tuning for Random Forest.
   - Feature selection techniques.

---

## Contact
For further questions or collaboration, please open an issue or contact the repository maintainers.

---
