import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_curve, auc
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import warnings

# Try to import TensorFlow for RNN functionality
try:
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, LSTM, Dropout
    from tensorflow.keras.callbacks import EarlyStopping

    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    st.warning("TensorFlow not available. RNN model will be disabled. Install with: pip install tensorflow")

warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #2c3e50;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .metric-container {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
        padding: 1rem;
        border-radius: 0.25rem;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        color: #856404;
        padding: 1rem;
        border-radius: 0.25rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)


class StreamlitChurnPredictor:
    def __init__(self):
        self.df = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.X_train_scaled = None
        self.X_test_scaled = None
        self.scaler = None
        self.models = {}
        self.results = {}
        self.feature_names = None
        self.target_column = None

    def load_data(self, uploaded_file):
        """Load data from uploaded file."""
        try:
            if uploaded_file.name.endswith('.csv'):
                self.df = pd.read_csv(uploaded_file)
            elif uploaded_file.name.endswith(('.xlsx', '.xls')):
                self.df = pd.read_excel(uploaded_file)
            else:
                st.error("Please upload a CSV or Excel file.")
                return None

            return self.df
        except Exception as e:
            st.error(f"Error loading file: {str(e)}")
            return None

    def preprocess_data(self):
        """Preprocess data for modeling."""
        if self.df is None:
            return False

        try:
            # Handle missing values
            if self.df.isnull().values.any():
                # For numeric columns, fill with mean
                numeric_cols = self.df.select_dtypes(include=[np.number]).columns
                for col in numeric_cols:
                    if self.df[col].isnull().sum() > 0:
                        self.df[col].fillna(self.df[col].mean(), inplace=True)

                # For categorical columns, fill with mode
                cat_cols = self.df.select_dtypes(exclude=[np.number]).columns
                for col in cat_cols:
                    if self.df[col].isnull().sum() > 0:
                        self.df[col].fillna(self.df[col].mode()[0], inplace=True)

            # Identify the target column
            target_columns = [col for col in self.df.columns if 'churn' in col.lower()]

            if not target_columns:
                return False

            self.target_column = target_columns[0]

            # Ensure target is numeric
            if self.df[self.target_column].dtype == 'object':
                le = LabelEncoder()
                self.df[self.target_column] = le.fit_transform(self.df[self.target_column])

            # Separate features and target
            X = self.df.drop(columns=[self.target_column])
            y = self.df[self.target_column]

            # Convert categorical features to dummy variables
            X = pd.get_dummies(X, drop_first=True)

            # Split the data
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                X, y, test_size=0.3, random_state=42
            )

            # Standardize features
            self.scaler = StandardScaler()
            self.X_train_scaled = self.scaler.fit_transform(self.X_train)
            self.X_test_scaled = self.scaler.transform(self.X_test)

            # Save feature names
            self.feature_names = X.columns.tolist()

            return True
        except Exception as e:
            st.error(f"Error preprocessing data: {str(e)}")
            return False

    def build_and_train_models(self, selected_models):
        """Build and train selected models."""
        if self.X_train_scaled is None:
            return False

        try:
            model_dict = {
                'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
                'Decision Tree': DecisionTreeClassifier(random_state=42),
                'Naive Bayes': GaussianNB(),
                'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
                'Gradient Boosting': GradientBoostingClassifier(random_state=42)
            }

            self.models = {name: model_dict[name] for name in selected_models}
            self.results = {}

            progress_bar = st.progress(0)
            status_text = st.empty()

            for i, (name, model) in enumerate(self.models.items()):
                status_text.text(f'Training {name}...')

                start_time = time.time()
                model.fit(self.X_train_scaled, self.y_train)

                y_pred = model.predict(self.X_test_scaled)
                y_pred_proba = model.predict_proba(self.X_test_scaled)[:, 1] if hasattr(model,
                                                                                        'predict_proba') else None

                training_time = time.time() - start_time

                self.results[name] = {
                    'model': model,
                    'accuracy': accuracy_score(self.y_test, y_pred),
                    'classification_report': classification_report(self.y_test, y_pred, output_dict=True),
                    'confusion_matrix': confusion_matrix(self.y_test, y_pred),
                    'training_time': training_time,
                    'y_pred': y_pred,
                    'y_pred_proba': y_pred_proba
                }

                progress_bar.progress((i + 1) / len(self.models))

            status_text.text('Training completed!')
            time.sleep(1)
            status_text.empty()
            progress_bar.empty()
        except Exception as e:
            print("RNN did not train properly")

    def build_rnn_model(self):
        """Build and train RNN model."""
        if not TENSORFLOW_AVAILABLE:
            st.error("TensorFlow is not available. Cannot build RNN model.")
            return False

        if self.X_train_scaled is None:
            return False

        try:
            st.info("Building RNN model... This may take a few minutes.")

            # Reshape input for RNN [samples, time steps, features]
            X_train_rnn = self.X_train_scaled.reshape(self.X_train_scaled.shape[0], 1, self.X_train_scaled.shape[1])
            X_test_rnn = self.X_test_scaled.reshape(self.X_test_scaled.shape[0], 1, self.X_test_scaled.shape[1])

            # Build RNN model
            model = Sequential([
                LSTM(64, input_shape=(1, self.X_train_scaled.shape[1]), return_sequences=True),
                Dropout(0.2),
                LSTM(32),
                Dropout(0.2),
                Dense(16, activation='relu'),
                Dense(1, activation='sigmoid')
            ])

            # Compile model
            model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

            # Define early stopping
            early_stopping = EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True
            )

            # Create progress bar for RNN training
            progress_text = st.empty()
            progress_bar = st.progress(0)

            class StreamlitCallback:
                def __init__(self, progress_bar, progress_text):
                    self.progress_bar = progress_bar
                    self.progress_text = progress_text
                    self.epoch = 0
                    self.total_epochs = 50

                def on_epoch_end(self, epoch, logs=None):
                    self.epoch = epoch + 1
                    progress = self.epoch / self.total_epochs
                    self.progress_bar.progress(progress)
                    self.progress_text.text(
                        f'RNN Training: Epoch {self.epoch}/{self.total_epochs} - Loss: {logs.get("loss", 0):.4f} - Accuracy: {logs.get("accuracy", 0):.4f}')

            # Train the model
            start_time = time.time()

            # Custom callback for Streamlit progress
            streamlit_callback = StreamlitCallback(progress_bar, progress_text)

            history = model.fit(
                X_train_rnn, self.y_train,
                epochs=50,
                batch_size=32,
                validation_split=0.2,
                callbacks=[early_stopping],
                verbose=0  # Set to 0 to avoid console output
            )

            training_time = time.time() - start_time

            # Evaluate model
            y_pred_proba = model.predict(X_test_rnn, verbose=0)
            y_pred = (y_pred_proba > 0.5).astype(int).flatten()

            # Store results
            self.results['RNN'] = {
                'model': model,
                'accuracy': accuracy_score(self.y_test, y_pred),
                'classification_report': classification_report(self.y_test, y_pred, output_dict=True),
                'confusion_matrix': confusion_matrix(self.y_test, y_pred),
                'training_time': training_time,
                'y_pred': y_pred,
                'y_pred_proba': y_pred_proba.flatten(),
                'history': history,
                'X_train_rnn': X_train_rnn,
                'X_test_rnn': X_test_rnn
            }

            # Clear progress indicators
            progress_bar.empty()
            progress_text.empty()

            st.success(f"RNN model trained successfully! Accuracy: {self.results['RNN']['accuracy']:.4f}")

            # Plot training history
            self.plot_rnn_history(history)

            return True

        except Exception as e:
            st.error(f"Error training RNN model: {str(e)}")
            return False

    def plot_rnn_history(self, history):
        """Plot RNN training history."""
        if history is None:
            return

        st.subheader("RNN Training History")

        # Create subplots
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Model Accuracy', 'Model Loss')
        )

        # Plot accuracy
        fig.add_trace(
            go.Scatter(y=history.history['accuracy'], name='Train Accuracy', line=dict(color='blue')),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(y=history.history['val_accuracy'], name='Validation Accuracy', line=dict(color='red')),
            row=1, col=1
        )

        # Plot loss
        fig.add_trace(
            go.Scatter(y=history.history['loss'], name='Train Loss', line=dict(color='blue'), showlegend=False),
            row=1, col=2
        )
        fig.add_trace(
            go.Scatter(y=history.history['val_loss'], name='Validation Loss', line=dict(color='red'), showlegend=False),
            row=1, col=2
        )

        fig.update_xaxes(title_text="Epoch", row=1, col=1)
        fig.update_xaxes(title_text="Epoch", row=1, col=2)
        fig.update_yaxes(title_text="Accuracy", row=1, col=1)
        fig.update_yaxes(title_text="Loss", row=1, col=2)

        fig.update_layout(height=400, showlegend=True)
        st.plotly_chart(fig, use_container_width=True)


def main():
    st.markdown('<h1 class="main-header">📊 Customer Churn Prediction Dashboard</h1>', unsafe_allow_html=True)

    # Initialize predictor
    if 'predictor' not in st.session_state:
        st.session_state.predictor = StreamlitChurnPredictor()

    predictor = st.session_state.predictor

    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox("Choose a page", [
        "Data Upload & Overview",
        "Model Training",
        "Model Comparison",
        "Feature Analysis",
        "Make Predictions"
    ])

    if page == "Data Upload & Overview":
        st.markdown('<h2 class="sub-header">📁 Data Upload & Overview</h2>', unsafe_allow_html=True)

        uploaded_file = st.file_uploader(
            "Upload your customer data file",
            type=['csv', 'xlsx', 'xls'],
            help="Upload a CSV or Excel file containing customer data with a 'churn' column"
        )

        if uploaded_file is not None:
            with st.spinner('Loading data...'):
                df = predictor.load_data(uploaded_file)

            if df is not None:
                st.markdown('<div class="success-box">✅ Data loaded successfully!</div>', unsafe_allow_html=True)

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Records", df.shape[0])
                with col2:
                    st.metric("Total Features", df.shape[1])
                with col3:
                    st.metric("Missing Values", df.isnull().sum().sum())

                # Data preview
                st.subheader("Data Preview")
                st.dataframe(df.head())

                # Data info
                st.subheader("Data Information")
                col1, col2 = st.columns(2)

                with col1:
                    st.write("**Data Types:**")
                    dtype_df = pd.DataFrame({
                        'Column': df.dtypes.index,
                        'Data Type': df.dtypes.values
                    })
                    st.dataframe(dtype_df)

                with col2:
                    st.write("**Missing Values:**")
                    missing_df = pd.DataFrame({
                        'Column': df.columns,
                        'Missing Count': df.isnull().sum().values,
                        'Missing %': (df.isnull().sum().values / len(df) * 100).round(2)
                    })
                    missing_df = missing_df[missing_df['Missing Count'] > 0]
                    if len(missing_df) > 0:
                        st.dataframe(missing_df)
                    else:
                        st.write("No missing values found!")

                # Target distribution
                target_cols = [col for col in df.columns if 'churn' in col.lower()]
                if target_cols:
                    st.subheader("Target Variable Distribution")
                    target_col = target_cols[0]

                    fig = px.pie(
                        values=df[target_col].value_counts().values,
                        names=df[target_col].value_counts().index,
                        title=f"Distribution of {target_col}"
                    )
                    st.plotly_chart(fig, use_container_width=True)

    elif page == "Model Training":
        st.markdown('<h2 class="sub-header">🤖 Model Training</h2>', unsafe_allow_html=True)

        if predictor.df is None:
            st.warning("Please upload data first!")
            return

        # Preprocessing
        if st.button("Preprocess Data"):
            with st.spinner('Preprocessing data...'):
                success = predictor.preprocess_data()

            if success:
                st.success("Data preprocessed successfully!")

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Training Samples", predictor.X_train.shape[0])
                with col2:
                    st.metric("Test Samples", predictor.X_test.shape[0])
                with col3:
                    st.metric("Features", predictor.X_train.shape[1])

        # Model selection and training
        if predictor.X_train_scaled is not None:
            st.subheader("Select Models to Train")

            available_models = [
                'Logistic Regression',
                'Decision Tree',
                'Naive Bayes',
                'Random Forest',
                'Gradient Boosting'
            ]

            # Add RNN if TensorFlow is available
            if TENSORFLOW_AVAILABLE:
                available_models.append('RNN (Neural Network)')

            selected_models = st.multiselect(
                "Choose models to train:",
                available_models,
                default=available_models[:3]
            )

            col1, col2 = st.columns(2)

            with col1:
                if st.button("Train Traditional Models") and selected_models:
                    # Filter out RNN for traditional training
                    traditional_models = [m for m in selected_models if m != 'RNN (Neural Network)']
                    if traditional_models:
                        with st.spinner('Training traditional models...'):
                            success = predictor.build_and_train_models(traditional_models)

                        if success:
                            st.success("Traditional models trained successfully!")

                            # Display results
                            results_df = pd.DataFrame({
                                'Model': list(predictor.results.keys()),
                                'Accuracy': [predictor.results[name]['accuracy'] for name in predictor.results.keys()],
                                'Training Time (s)': [predictor.results[name]['training_time'] for name in
                                                      predictor.results.keys()]
                            }).sort_values('Accuracy', ascending=False)

                            st.subheader("Training Results")
                            st.dataframe(results_df)
                    else:
                        st.warning("Please select at least one traditional model.")

            with col2:
                if TENSORFLOW_AVAILABLE and 'RNN (Neural Network)' in selected_models:
                    if st.button("Train RNN Model"):
                        success = predictor.build_rnn_model()

                        if success:
                            # Update results display
                            if predictor.results:
                                results_df = pd.DataFrame({
                                    'Model': list(predictor.results.keys()),
                                    'Accuracy': [predictor.results[name]['accuracy'] for name in
                                                 predictor.results.keys()],
                                    'Training Time (s)': [predictor.results[name].get('training_time', 'N/A') for name
                                                          in predictor.results.keys()]
                                }).sort_values('Accuracy', ascending=False)

                                st.subheader("Updated Training Results")
                                st.dataframe(results_df)

    elif page == "Model Comparison":
        st.markdown('<h2 class="sub-header">📈 Model Comparison</h2>', unsafe_allow_html=True)

        if not predictor.results:
            st.warning("Please train models first!")
            return

        # Accuracy comparison
        st.subheader("Model Accuracy Comparison")

        model_names = list(predictor.results.keys())
        accuracies = [predictor.results[name]['accuracy'] for name in model_names]

        fig = px.bar(
            x=model_names,
            y=accuracies,
            title="Model Accuracy Comparison",
            labels={'x': 'Model', 'y': 'Accuracy'}
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

        # ROC Curves
        st.subheader("ROC Curves")

        fig = go.Figure()

        for name in predictor.results:
            if predictor.results[name]['y_pred_proba'] is not None:
                fpr, tpr, _ = roc_curve(predictor.y_test, predictor.results[name]['y_pred_proba'])
                roc_auc = auc(fpr, tpr)

                fig.add_trace(go.Scatter(
                    x=fpr, y=tpr,
                    mode='lines',
                    name=f'{name} (AUC = {roc_auc:.3f})'
                ))

        # Add diagonal line
        fig.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1],
            mode='lines',
            line=dict(dash='dash', color='gray'),
            name='No Skill',
            showlegend=False
        ))

        fig.update_layout(
            title='ROC Curves',
            xaxis_title='False Positive Rate',
            yaxis_title='True Positive Rate'
        )

        st.plotly_chart(fig, use_container_width=True)

        # Detailed metrics
        st.subheader("Detailed Model Metrics")

        selected_model = st.selectbox("Select model for detailed metrics:", model_names)

        if selected_model:
            col1, col2 = st.columns(2)

            with col1:
                st.write("**Classification Report:**")
                report = predictor.results[selected_model]['classification_report']
                report_df = pd.DataFrame(report).transpose()
                st.dataframe(report_df)

            with col2:
                st.write("**Confusion Matrix:**")
                cm = predictor.results[selected_model]['confusion_matrix']

                fig = px.imshow(
                    cm,
                    text_auto=True,
                    aspect="auto",
                    title=f"Confusion Matrix - {selected_model}",
                    labels=dict(x="Predicted", y="Actual")
                )
                st.plotly_chart(fig, use_container_width=True)

    elif page == "Feature Analysis":
        st.markdown('<h2 class="sub-header">🔍 Feature Analysis</h2>', unsafe_allow_html=True)

        if not predictor.results:
            st.warning("Please train models first!")
            return

        # Feature importance
        st.subheader("Feature Importance")

        # Select model for feature importance
        models_with_importance = ['Logistic Regression', 'Decision Tree', 'Random Forest', 'Gradient Boosting']
        available_models = [m for m in models_with_importance if m in predictor.results]

        if available_models:
            selected_model = st.selectbox("Select model for feature importance:", available_models)

            model = predictor.results[selected_model]['model']

            if selected_model == 'Logistic Regression':
                importance = np.abs(model.coef_[0])
            else:
                importance = model.feature_importances_

            # Create feature importance dataframe
            feature_importance_df = pd.DataFrame({
                'Feature': predictor.feature_names,
                'Importance': importance
            }).sort_values('Importance', ascending=False)

            # Plot feature importance
            top_n = st.slider("Number of top features to display:", 5, min(20, len(feature_importance_df)), 10)

            fig = px.bar(
                feature_importance_df.head(top_n),
                x='Importance',
                y='Feature',
                orientation='h',
                title=f'Top {top_n} Feature Importance - {selected_model}'
            )
            fig.update_layout(yaxis={'categoryorder': 'total ascending'})
            st.plotly_chart(fig, use_container_width=True)

            # Feature importance table
            st.write("**Feature Importance Table:**")
            st.dataframe(feature_importance_df.head(top_n))
        else:
            st.info(
                "Train models with feature importance capability (Logistic Regression, Decision Tree, Random Forest, or Gradient Boosting) to see feature analysis.")

        # RNN specific analysis
        if 'RNN' in predictor.results:
            st.subheader("RNN Model Analysis")

            # Show RNN architecture
            with st.expander("RNN Model Architecture"):
                st.text("Model: Sequential")
                st.text("├── LSTM(64, return_sequences=True)")
                st.text("├── Dropout(0.2)")
                st.text("├── LSTM(32)")
                st.text("├── Dropout(0.2)")
                st.text("├── Dense(16, activation='relu')")
                st.text("└── Dense(1, activation='sigmoid')")

            # Show training history if available
            if 'history' in predictor.results['RNN']:
                predictor.plot_rnn_history(predictor.results['RNN']['history'])

    elif page == "Make Predictions":
        st.markdown('<h2 class="sub-header">🎯 Make Predictions</h2>', unsafe_allow_html=True)

        if not predictor.results:
            st.warning("Please train models first!")
            return

        st.subheader("Single Customer Prediction")

        # Select model for prediction
        best_model = max(predictor.results, key=lambda x: predictor.results[x]['accuracy'])
        selected_model = st.selectbox(
            "Select model for prediction:",
            list(predictor.results.keys()),
            index=list(predictor.results.keys()).index(best_model)
        )

        st.write(
            f"**Selected Model:** {selected_model} (Accuracy: {predictor.results[selected_model]['accuracy']:.4f})")

        # Create input form
        st.subheader("Enter Customer Information")

        # Sample data for demonstration
        sample_data = {
            'customerID': '7559-EXAMPLE',
            'gender': 'Male',
            'SeniorCitizen': 0,
            'Partner': 'No',
            'Dependents': 'No',
            'tenure': 12,
            'PhoneService': 'Yes',
            'MultipleLines': 'No',
            'InternetService': 'DSL',
            'OnlineSecurity': 'No',
            'OnlineBackup': 'No',
            'DeviceProtection': 'No',
            'TechSupport': 'No',
            'StreamingTV': 'No',
            'StreamingMovies': 'No',
            'Contract': 'Month-to-month',
            'PaperlessBilling': 'Yes',
            'PaymentMethod': 'Electronic check',
            'MonthlyCharges': 50.0,
            'TotalCharges': 600.0
        }

        # Create input form based on original data columns
        if predictor.df is not None:
            input_data = {}

            # Create form columns
            col1, col2 = st.columns(2)

            # Get original columns (excluding target)
            original_cols = [col for col in predictor.df.columns if col != predictor.target_column]

            for i, col in enumerate(original_cols):
                column = col1 if i % 2 == 0 else col2

                with column:
                    if predictor.df[col].dtype == 'object':
                        unique_values = predictor.df[col].unique().tolist()
                        input_data[col] = st.selectbox(f"{col}:", unique_values, key=col)
                    else:
                        min_val = float(predictor.df[col].min())
                        max_val = float(predictor.df[col].max())
                        default_val = float(predictor.df[col].mean())
                        input_data[col] = st.number_input(
                            f"{col}:",
                            min_value=min_val,
                            max_value=max_val,
                            value=default_val,
                            key=col
                        )

            if st.button("Make Prediction", type="primary"):
                try:
                    # Create DataFrame from input
                    input_df = pd.DataFrame([input_data])

                    # Apply same preprocessing as training data
                    input_df = pd.get_dummies(input_df, drop_first=True)

                    # Ensure all columns are present
                    missing_cols = set(predictor.feature_names) - set(input_df.columns)
                    for col in missing_cols:
                        input_df[col] = 0

                    # Ensure column order matches
                    input_df = input_df[predictor.feature_names]

                    # Scale the data
                    input_scaled = predictor.scaler.transform(input_df)

                    # Make prediction
                    model = predictor.results[selected_model]['model']

                    if selected_model == 'RNN':
                        # Special handling for RNN
                        input_rnn = input_scaled.reshape(input_scaled.shape[0], 1, input_scaled.shape[1])
                        probability = model.predict(input_rnn, verbose=0)[0, 0]
                        prediction = 1 if probability > 0.5 else 0
                    else:
                        # Traditional models
                        prediction = model.predict(input_scaled)[0]
                        if hasattr(model, 'predict_proba'):
                            probability = model.predict_proba(input_scaled)[0, 1]
                        else:
                            probability = None

                    # Display results
                    st.subheader("Prediction Results")

                    col1, col2 = st.columns(2)

                    with col1:
                        if prediction == 1:
                            st.error("🚨 **CHURN RISK: HIGH** 🚨")
                            st.write("This customer is likely to churn.")
                        else:
                            st.success("✅ **CHURN RISK: LOW** ✅")
                            st.write("This customer is likely to stay.")

                    with col2:
                        if probability is not None:
                            st.metric("Churn Probability", f"{probability:.2%}")

                            # Probability gauge
                            fig = go.Figure(go.Indicator(
                                mode="gauge+number",
                                value=probability * 100,
                                domain={'x': [0, 1], 'y': [0, 1]},
                                title={'text': "Churn Probability (%)"},
                                gauge={
                                    'axis': {'range': [None, 100]},
                                    'bar': {'color': "darkblue"},
                                    'steps': [
                                        {'range': [0, 50], 'color': "lightgray"},
                                        {'range': [50, 80], 'color': "yellow"},
                                        {'range': [80, 100], 'color': "red"}
                                    ],
                                    'threshold': {
                                        'line': {'color': "red", 'width': 4},
                                        'thickness': 0.75,
                                        'value': 90
                                    }
                                }
                            ))
                            fig.update_layout(height=300)
                            st.plotly_chart(fig, use_container_width=True)

                except Exception as e:
                    st.error(f"Error making prediction: {str(e)}")

    # Footer
    st.markdown("---")
    st.markdown("Customer Churn Prediction Dashboard")


if __name__ == "__main__":
    main()