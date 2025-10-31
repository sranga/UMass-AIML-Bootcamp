import pandas as pd
import numpy as np
from sklearn.model_selection import (
    train_test_split, cross_val_score, GridSearchCV, 
    StratifiedKFold, learning_curve
)
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier, 
    VotingClassifier, BaggingClassifier
)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    classification_report, confusion_matrix, roc_auc_score,
    log_loss, mean_squared_error
)
from sklearn.preprocessing import StandardScaler, LabelEncoder

import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from typing import Dict, List, Tuple, Any
import time
import json

class FakeNewsDetectionPipeline:
    """
    Comprehensive ML Pipeline for Fake News Detection
    """
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.results = {}
        self.best_models = {}
        self.ensemble_model = None
        self.cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
        
        # Define models to test
        self.models = {
            'Logistic_Regression': LogisticRegression(random_state=random_state, max_iter=1000),
            'Random_Forest': RandomForestClassifier(random_state=random_state),
            'Gradient_Boosting': GradientBoostingClassifier(random_state=random_state),
            'SVM': SVC(random_state=random_state, probability=True),
            'Naive_Bayes': MultinomialNB(),
            'MLP': MLPClassifier(random_state=random_state, max_iter=500),
            'Decision_Tree': DecisionTreeClassifier(random_state=random_state),
            'KNN': KNeighborsClassifier()
        }
        
        # Define hyperparameter grids for tuning
        self.param_grids = {
            'Logistic_Regression': {
                'C': [0.1, 1, 10],
                'penalty': ['l1', 'l2'],
                'solver': ['liblinear']
            },
            'Random_Forest': {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 10, 20],
                'min_samples_split': [2, 5, 10]
            },
            'Gradient_Boosting': {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7]
            },
            'SVM': {
                'C': [0.1, 1, 10],
                'kernel': ['rbf', 'linear'],
                'gamma': ['scale', 'auto']
            }
        }
        
        # Define loss functions for neural networks
        self.loss_functions = ['log_loss', 'hinge', 'squared_hinge']
        
    def load_and_preprocess_data(self, data_path: str = None, text_column: str = 'text', 
                                label_column: str = 'label') -> Tuple[np.ndarray, np.ndarray]:
        """
        Load and preprocess the data
        """
        if data_path:
            self.data = pd.read_csv(data_path)
        else:
            # Create sample data for demonstration
            np.random.seed(self.random_state)
            fake_texts = [
                "Breaking: Shocking revelation that will change everything you know!",
                "Scientists HATE this one simple trick that doctors don't want you to know",
                "URGENT: Government hiding the truth from citizens",
                "Celebrity dies in mysterious circumstances - cover up suspected",
                "Miracle cure discovered but pharmaceutical companies suppress it"
            ] * 200
            
            real_texts = [
                "According to recent studies published in peer-reviewed journals",
                "Government officials announced new policy measures yesterday",
                "Research conducted by university scientists shows evidence",
                "Economic indicators suggest moderate growth in the sector",
                "Local authorities report routine maintenance on infrastructure"
            ] * 200
            
            texts = fake_texts + real_texts
            labels = [1] * len(fake_texts) + [0] * len(real_texts)
            
            # Add some noise and shuffle
            combined = list(zip(texts, labels))
            np.random.shuffle(combined)
            texts, labels = zip(*combined)
            
            self.data = pd.DataFrame({
                text_column: texts,
                label_column: labels
            })
        
        # Text preprocessing and vectorization
        print("Preprocessing text data...")
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words='english',
            ngram_range=(1, 2),
            lowercase=True
        )
        
        X = self.vectorizer.fit_transform(self.data[text_column]).toarray()
        y = self.data[label_column].values
        
        # Ensure binary classification
        if len(np.unique(y)) > 2:
            self.label_encoder = LabelEncoder()
            y = self.label_encoder.fit_transform(y)
        
        print(f"Data shape: {X.shape}")
        print(f"Class distribution: {np.bincount(y)}")
        
        return X, y
    
    def define_performance_metrics(self) -> Dict[str, callable]:
        """
        Define performance metrics suitable for fake news detection
        """
        return {
            'accuracy': accuracy_score,
            'f1_macro': lambda y_true, y_pred: f1_score(y_true, y_pred, average='macro'),
            'f1_weighted': lambda y_true, y_pred: f1_score(y_true, y_pred, average='weighted'),
            'precision_macro': lambda y_true, y_pred: precision_score(y_true, y_pred, average='macro'),
            'recall_macro': lambda y_true, y_pred: recall_score(y_true, y_pred, average='macro'),
            'roc_auc': lambda y_true, y_pred: roc_auc_score(y_true, y_pred)
        }
    
    def automated_model_testing(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Dict]:
        """
        Automated testing of multiple ML algorithms
        """
        print("Starting automated model testing...")
        metrics = self.define_performance_metrics()
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=self.random_state, stratify=y
        )
        
        results = {}
        
        for model_name, model in self.models.items():
            print(f"\nTesting {model_name}...")
            start_time = time.time()
            
            try:
                # Fit the model
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                y_pred_proba = None
                
                if hasattr(model, 'predict_proba'):
                    y_pred_proba = model.predict_proba(X_test)[:, 1]
                
                # Calculate metrics
                model_results = {}
                for metric_name, metric_func in metrics.items():
                    try:
                        if metric_name == 'roc_auc' and y_pred_proba is not None:
                            score = metric_func(y_test, y_pred_proba)
                        else:
                            score = metric_func(y_test, y_pred)
                        model_results[metric_name] = score
                    except Exception as e:
                        model_results[metric_name] = np.nan
                
                # Cross-validation scores
                cv_scores = cross_val_score(
                    model, X_train, y_train, cv=self.cv_strategy, 
                    scoring='f1_macro', n_jobs=-1
                )
                
                model_results['cv_mean'] = cv_scores.mean()
                model_results['cv_std'] = cv_scores.std()
                model_results['training_time'] = time.time() - start_time
                
                results[model_name] = model_results
                
            except Exception as e:
                print(f"Error testing {model_name}: {str(e)}")
                results[model_name] = {'error': str(e)}
        
        self.results['automated_testing'] = results
        return results
    
    def test_loss_functions(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """
        Test various loss functions with neural network models
        """
        print("\nTesting different loss functions...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=self.random_state, stratify=y
        )
        
        # Scale data for neural networks
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        loss_results = {}
        
        # Test different configurations
        configurations = [
            {'loss': 'log_loss', 'activation': 'logistic'},
            {'loss': 'log_loss', 'activation': 'relu'},
            {'loss': 'log_loss', 'activation': 'tanh'}
        ]
        
        for i, config in enumerate(configurations):
            try:
                model = MLPClassifier(
                    hidden_layer_sizes=(100, 50),
                    max_iter=500,
                    random_state=self.random_state,
                    **config
                )
                
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
                y_pred_proba = model.predict_proba(X_test_scaled)
                
                # Calculate log loss
                loss_value = log_loss(y_test, y_pred_proba)
                f1 = f1_score(y_test, y_pred, average='macro')
                
                config_name = f"MLP_{config['loss']}_{config['activation']}"
                loss_results[config_name] = {
                    'log_loss': loss_value,
                    'f1_score': f1,
                    'config': config
                }
                
            except Exception as e:
                print(f"Error with configuration {config}: {str(e)}")
        
        self.results['loss_functions'] = loss_results
        return loss_results
    
    def hyperparameter_tuning(self, X: np.ndarray, y: np.ndarray, 
                            models_to_tune: List[str] = None) -> Dict[str, Any]:
        """
        Perform hyperparameter tuning on selected models
        """
        if models_to_tune is None:
            models_to_tune = ['Random_Forest', 'Gradient_Boosting', 'Logistic_Regression', 'SVM']
        
        print("\nPerforming hyperparameter tuning...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=self.random_state, stratify=y
        )
        
        tuning_results = {}
        
        for model_name in models_to_tune:
            if model_name not in self.param_grids:
                continue
                
            print(f"Tuning {model_name}...")
            
            try:
                model = self.models[model_name]
                param_grid = self.param_grids[model_name]
                
                # Perform grid search
                grid_search = GridSearchCV(
                    model, param_grid, cv=self.cv_strategy,
                    scoring='f1_macro', n_jobs=-1, verbose=1
                )
                
                grid_search.fit(X_train, y_train)
                
                # Test best model
                best_model = grid_search.best_estimator_
                y_pred = best_model.predict(X_test)
                
                tuning_results[model_name] = {
                    'best_params': grid_search.best_params_,
                    'best_cv_score': grid_search.best_score_,
                    'test_f1': f1_score(y_test, y_pred, average='macro'),
                    'test_accuracy': accuracy_score(y_test, y_pred),
                    'best_model': best_model
                }
                
                # Store best model
                self.best_models[model_name] = best_model
                
            except Exception as e:
                print(f"Error tuning {model_name}: {str(e)}")
                tuning_results[model_name] = {'error': str(e)}
        
        self.results['hyperparameter_tuning'] = tuning_results
        return tuning_results
    
    def robust_cross_validation(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """
        Implement robust cross-validation with multiple strategies
        """
        print("\nPerforming robust cross-validation...")
        
        from sklearn.model_selection import (
            StratifiedKFold, RepeatedStratifiedKFold, 
            StratifiedShuffleSplit
        )
        
        cv_strategies = {
            'StratifiedKFold_5': StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state),
            'StratifiedKFold_10': StratifiedKFold(n_splits=10, shuffle=True, random_state=self.random_state),
            'RepeatedStratifiedKFold': RepeatedStratifiedKFold(
                n_splits=5, n_repeats=3, random_state=self.random_state
            ),
            'StratifiedShuffleSplit': StratifiedShuffleSplit(
                n_splits=10, test_size=0.2, random_state=self.random_state
            )
        }
        
        cv_results = {}
        
        # Test with best performing models from previous steps
        models_to_test = {}
        if self.best_models:
            models_to_test.update(self.best_models)
        else:
            # Use subset of original models
            models_to_test = {
                'Random_Forest': self.models['Random_Forest'],
                'Gradient_Boosting': self.models['Gradient_Boosting'],
                'Logistic_Regression': self.models['Logistic_Regression']
            }
        
        for cv_name, cv_strategy in cv_strategies.items():
            cv_results[cv_name] = {}
            
            for model_name, model in models_to_test.items():
                try:
                    scores = cross_val_score(
                        model, X, y, cv=cv_strategy, 
                        scoring='f1_macro', n_jobs=-1
                    )
                    
                    cv_results[cv_name][model_name] = {
                        'scores': scores.tolist(),
                        'mean': scores.mean(),
                        'std': scores.std(),
                        'min': scores.min(),
                        'max': scores.max()
                    }
                    
                except Exception as e:
                    cv_results[cv_name][model_name] = {'error': str(e)}
        
        self.results['cross_validation'] = cv_results
        return cv_results
    
    def build_ensemble_models(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """
        Build and test ensemble models
        """
        print("\nBuilding ensemble models...")
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=self.random_state, stratify=y
        )
        
        # Prepare base models for ensemble
        if self.best_models:
            base_models = [(name, model) for name, model in self.best_models.items()]
        else:
            base_models = [
                ('rf', RandomForestClassifier(n_estimators=100, random_state=self.random_state)),
                ('gb', GradientBoostingClassifier(n_estimators=100, random_state=self.random_state)),
                ('lr', LogisticRegression(random_state=self.random_state, max_iter=1000))
            ]
        
        ensemble_results = {}
        
        # Voting Classifier (Hard voting)
        try:
            voting_hard = VotingClassifier(estimators=base_models, voting='hard')
            voting_hard.fit(X_train, y_train)
            y_pred_hard = voting_hard.predict(X_test)
            
            ensemble_results['Voting_Hard'] = {
                'accuracy': accuracy_score(y_test, y_pred_hard),
                'f1_macro': f1_score(y_test, y_pred_hard, average='macro'),
                'f1_weighted': f1_score(y_test, y_pred_hard, average='weighted')
            }
        except Exception as e:
            ensemble_results['Voting_Hard'] = {'error': str(e)}
        
        # Voting Classifier (Soft voting)
        try:
            voting_soft = VotingClassifier(estimators=base_models, voting='soft')
            voting_soft.fit(X_train, y_train)
            y_pred_soft = voting_soft.predict(X_test)
            
            ensemble_results['Voting_Soft'] = {
                'accuracy': accuracy_score(y_test, y_pred_soft),
                'f1_macro': f1_score(y_test, y_pred_soft, average='macro'),
                'f1_weighted': f1_score(y_test, y_pred_soft, average='weighted')
            }
            
            # Store best ensemble model
            self.ensemble_model = voting_soft
            
        except Exception as e:
            ensemble_results['Voting_Soft'] = {'error': str(e)}
        
        # Bagging Ensemble
        try:
            bagging = BaggingClassifier(
                base_estimator=RandomForestClassifier(random_state=self.random_state),
                n_estimators=10,
                random_state=self.random_state
            )
            bagging.fit(X_train, y_train)
            y_pred_bag = bagging.predict(X_test)
            
            ensemble_results['Bagging'] = {
                'accuracy': accuracy_score(y_test, y_pred_bag),
                'f1_macro': f1_score(y_test, y_pred_bag, average='macro'),
                'f1_weighted': f1_score(y_test, y_pred_bag, average='weighted')
            }
        except Exception as e:
            ensemble_results['Bagging'] = {'error': str(e)}
        
        self.results['ensemble_models'] = ensemble_results
        return ensemble_results
    
    def analyze_generalization_and_overfitting(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """
        Analyze model generalization and detect overfitting
        """
        print("\nAnalyzing generalization and overfitting...")
        
        analysis_results = {}
        
        # Models to analyze
        models_to_analyze = {}
        if self.best_models:
            models_to_analyze.update(self.best_models)
        models_to_analyze['Ensemble'] = self.ensemble_model if self.ensemble_model else None
        
        for model_name, model in models_to_analyze.items():
            if model is None:
                continue
                
            try:
                # Learning curves
                train_sizes, train_scores, val_scores = learning_curve(
                    model, X, y, cv=5, n_jobs=-1,
                    train_sizes=np.linspace(0.1, 1.0, 10),
                    scoring='f1_macro'
                )
                
                train_mean = np.mean(train_scores, axis=1)
                train_std = np.std(train_scores, axis=1)
                val_mean = np.mean(val_scores, axis=1)
                val_std = np.std(val_scores, axis=1)
                
                # Calculate overfitting indicators
                final_train_score = train_mean[-1]
                final_val_score = val_mean[-1]
                overfitting_gap = final_train_score - final_val_score
                
                # Stability analysis
                stability = 1 - (val_std[-1] / val_mean[-1]) if val_mean[-1] != 0 else 0
                
                analysis_results[model_name] = {
                    'train_sizes': train_sizes.tolist(),
                    'train_scores_mean': train_mean.tolist(),
                    'train_scores_std': train_std.tolist(),
                    'val_scores_mean': val_mean.tolist(),
                    'val_scores_std': val_std.tolist(),
                    'final_train_score': final_train_score,
                    'final_val_score': final_val_score,
                    'overfitting_gap': overfitting_gap,
                    'stability': stability,
                    'is_overfitting': overfitting_gap > 0.1,  # Threshold for overfitting
                    'is_stable': stability > 0.8  # Threshold for stability
                }
                
            except Exception as e:
                analysis_results[model_name] = {'error': str(e)}
        
        self.results['generalization_analysis'] = analysis_results
        return analysis_results
    
    def generate_comprehensive_report(self) -> str:
        """
        Generate a comprehensive report of all results
        """
        report = "=" * 80 + "\n"
        report += "FAKE NEWS DETECTION - COMPREHENSIVE ML PIPELINE REPORT\n"
        report += "=" * 80 + "\n\n"
        
        # Automated Testing Results
        if 'automated_testing' in self.results:
            report += "1. AUTOMATED MODEL TESTING RESULTS\n"
            report += "-" * 40 + "\n"
            
            results_df = pd.DataFrame(self.results['automated_testing']).T
            if 'f1_macro' in results_df.columns:
                best_model = results_df['f1_macro'].idxmax()
                report += f"Best performing model: {best_model}\n"
                report += f"Best F1 Score: {results_df.loc[best_model, 'f1_macro']:.4f}\n\n"
            
            report += "Top 3 Models by F1 Score:\n"
            if 'f1_macro' in results_df.columns:
                top_models = results_df.nlargest(3, 'f1_macro')[['f1_macro', 'accuracy', 'cv_mean']]
                report += str(top_models) + "\n\n"
        
        # Loss Function Results
        if 'loss_functions' in self.results:
            report += "2. LOSS FUNCTION TESTING RESULTS\n"
            report += "-" * 40 + "\n"
            loss_results = self.results['loss_functions']
            for config, metrics in loss_results.items():
                if 'error' not in metrics:
                    report += f"{config}: F1={metrics['f1_score']:.4f}, Log Loss={metrics['log_loss']:.4f}\n"
            report += "\n"
        
        # Hyperparameter Tuning Results
        if 'hyperparameter_tuning' in self.results:
            report += "3. HYPERPARAMETER TUNING RESULTS\n"
            report += "-" * 40 + "\n"
            for model, results in self.results['hyperparameter_tuning'].items():
                if 'error' not in results:
                    report += f"{model}:\n"
                    report += f"  Best CV Score: {results['best_cv_score']:.4f}\n"
                    report += f"  Test F1 Score: {results['test_f1']:.4f}\n"
                    report += f"  Best Parameters: {results['best_params']}\n\n"
        
        # Cross-Validation Results
        if 'cross_validation' in self.results:
            report += "4. CROSS-VALIDATION ANALYSIS\n"
            report += "-" * 40 + "\n"
            cv_results = self.results['cross_validation']
            for cv_strategy in cv_results:
                report += f"{cv_strategy}:\n"
                for model, scores in cv_results[cv_strategy].items():
                    if 'error' not in scores:
                        report += f"  {model}: {scores['mean']:.4f} (±{scores['std']:.4f})\n"
                report += "\n"
        
        # Ensemble Results
        if 'ensemble_models' in self.results:
            report += "5. ENSEMBLE MODEL RESULTS\n"
            report += "-" * 40 + "\n"
            for ensemble, metrics in self.results['ensemble_models'].items():
                if 'error' not in metrics:
                    report += f"{ensemble}:\n"
                    report += f"  Accuracy: {metrics['accuracy']:.4f}\n"
                    report += f"  F1 Macro: {metrics['f1_macro']:.4f}\n"
                    report += f"  F1 Weighted: {metrics['f1_weighted']:.4f}\n\n"
        
        # Generalization Analysis
        if 'generalization_analysis' in self.results:
            report += "6. GENERALIZATION AND OVERFITTING ANALYSIS\n"
            report += "-" * 40 + "\n"
            for model, analysis in self.results['generalization_analysis'].items():
                if 'error' not in analysis:
                    report += f"{model}:\n"
                    report += f"  Final Validation Score: {analysis['final_val_score']:.4f}\n"
                    report += f"  Overfitting Gap: {analysis['overfitting_gap']:.4f}\n"
                    report += f"  Is Overfitting: {analysis['is_overfitting']}\n"
                    report += f"  Stability: {analysis['stability']:.4f}\n\n"
        
        report += "=" * 80 + "\n"
        report += "RECOMMENDATIONS:\n"
        report += "=" * 80 + "\n"
        
        # Add recommendations based on results
        if 'ensemble_models' in self.results:
            best_ensemble = max(self.results['ensemble_models'].items(), 
                              key=lambda x: x[1].get('f1_macro', 0) if 'error' not in x[1] else 0)
            report += f"1. Use {best_ensemble[0]} ensemble model for best performance\n"
        
        if 'generalization_analysis' in self.results:
            stable_models = [model for model, analysis in self.results['generalization_analysis'].items()
                           if 'error' not in analysis and analysis.get('is_stable', False)]
            if stable_models:
                report += f"2. Most stable models: {', '.join(stable_models)}\n"
        
        report += "3. Consider additional feature engineering for improved performance\n"
        report += "4. Collect more diverse training data to reduce overfitting\n"
        report += "5. Implement online learning for real-time fake news detection\n"
        
        return report
    
    def run_complete_pipeline(self, data_path: str = None, 
                            text_column: str = 'text', 
                            label_column: str = 'label') -> Dict[str, Any]:
        """
        Run the complete ML pipeline
        """
        print("Starting comprehensive ML pipeline for Fake News Detection...")
        
        # Step 1: Load and preprocess data
        X, y = self.load_and_preprocess_data(data_path, text_column, label_column)
        
        # Step 2: Automated model testing
        self.automated_model_testing(X, y)
        
        # Step 3: Test loss functions
        self.test_loss_functions(X, y)
        
        # Step 4: Hyperparameter tuning
        self.hyperparameter_tuning(X, y)
        
        # Step 5: Robust cross-validation
        self.robust_cross_validation(X, y)
        
        # Step 6: Build ensemble models
        self.build_ensemble_models(X, y)
        
        # Step 7: Analyze generalization and overfitting
        self.analyze_generalization_and_overfitting(X, y)
        
        # Step 8: Generate comprehensive report
        report = self.generate_comprehensive_report()
        
        return {
            'results': self.results,
            'report': report,
            'best_models': self.best_models,
            'ensemble_model': self.ensemble_model
        }

# Example usage
if __name__ == "__main__":
    # Initialize the pipeline
    pipeline = FakeNewsDetectionPipeline(random_state=42)
    
    # Run the complete pipeline
    # For your actual data, replace None with the path to your CSV file
    # and specify the correct column names
    results = pipeline.run_complete_pipeline(
        data_path=None,  # Replace with your data path
        text_column='text',  # Replace with your text column name
        label_column='label'  # Replace with your label column name
    )
    
    # Print the comprehensive report
    print(results['report'])
    
    # Save results to file
    with open('fake_news_detection_results.json', 'w') as f:
        # Convert numpy arrays to lists for JSON serialization
        import json
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            return obj
        
        serializable_results = json.loads(json.dumps(results['results'], default=convert_numpy))
        json.dump(serializable_results, f, indent=2)
    
    print("\nResults saved to 'fake_news_detection_results.json'")
    print("Report saved to console output")
