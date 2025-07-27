
from sklearn.metrics import precision_score
import numpy as np

# Example binary labels for anomaly detection
y_true = np.array([0, 0, 1, 1, 0, 1, 0, 0, 1, 0])
y_pred = np.array([0, 0, 1, 1, 0, 1, 0, 1, 1, 0])

precision_at_n = precision_score(y_true, y_pred)

print(f"Precision@N: {precision_at_n:.2f}")
