# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Load dataset
df = pd.read_csv("custom_boston_dataset.csv")
X = df.drop('PRICE', axis=1)
y = df['PRICE']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define scalers
scaler_minmax = MinMaxScaler()
scaler_standard = StandardScaler()

# Apply MinMaxScaler
X_train_mm = scaler_minmax.fit_transform(X_train)
X_test_mm = scaler_minmax.transform(X_test)

# Apply StandardScaler
X_train_std = scaler_standard.fit_transform(X_train)
X_test_std = scaler_standard.transform(X_test)

# Define model training function
def train_model(X_train, y_train, X_test, y_test, desc=""):
    model = Sequential([
        Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
        Dense(64, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_train, y_train, epochs=100, batch_size=32, verbose=0)
    loss = model.evaluate(X_test, y_test, verbose=0)
    print(f"{desc} - MSE on test set: {loss:.4f}")

# Train and compare
train_model(X_train_std, y_train, X_test_std, y_test, "StandardScaler")
train_model(X_train_mm, y_train, X_test_mm, y_test, "MinMaxScaler")
