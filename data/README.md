# Data instructions

This repository does not include any proprietary datasets. Place your dataset in the data/ directory.

Expected CSV format (example columns):
- order_id (optional)
- customer_id (optional)
- product_id (optional)
- features... (one or more numeric feature columns, e.g., price, qty, time_on_site)
- review_text (optional)  if you plan to use text features, preprocess separately
- purchase_date (optional)
- satisfaction_score (required)  target column. Numeric score or rating (e.g., 1-5)

If your dataset contains raw text, add a preprocessing step to produce a numeric feature matrix or embeddings before training the baseline MLP.

Preprocessing tips:
- Impute or drop missing values
- Normalize numeric features (standard scaling)
- Convert categorical variables with one-hot encoding or embeddings
