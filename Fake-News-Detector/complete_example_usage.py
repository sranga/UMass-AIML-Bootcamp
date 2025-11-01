"""
Complete Example Usage Script for Fake News Detection ML Pipeline

This script demonstrates how to use the comprehensive ML pipeline with your own data
and includes examples with both sample data and real data loading.
"""

import pandas as pd
import numpy as np
import os
import sys
from datetime import datetime
from fake_news_ml_pipeline import FakeNewsDetectionPipeline
from visualization_analysis_module import FakeNewsVisualizationModule

import warnings
warnings.filterwarnings('ignore')

# Import our custom modules (assuming they're in the same directory)
# from fake_news_pipeline import FakeNewsDetectionPipeline
# from visualization_module import FakeNewsVisualizationModule

def create_sample_fake_news_dataset(n_samples: int = 2000) -> pd.DataFrame:
    """
    Create a more realistic sample dataset for demonstration
    """
    np.random.seed(42)
    
    # Fake news patterns and keywords
    fake_patterns = [
        "BREAKING: Shocking truth revealed about",
        "Scientists HATE this one simple trick",
        "Government doesn't want you to know",
        "Doctors are baffled by this",
        "This will change everything you thought you knew",
        "URGENT: Hidden conspiracy exposed",
        "Miracle cure that pharmaceutical companies hide",
        "Celebrity secret that will amaze you",
        "Unbelievable discovery that mainstream media ignores",
        "Ancient secret finally revealed"
    ]
    
    fake_topics = [
        "vaccines", "5G towers", "climate change hoax", "election fraud",
        "celebrity death", "miracle diet", "government surveillance", 
        "alien encounter", "financial crash", "health supplement"
    ]
    
    # Real news patterns
    real_patterns = [
        "According to recent studies published in",
        "Research conducted by scientists at",
        "Government officials announced new policy",
        "Economic indicators suggest",
        "Local authorities report",
        "Medical experts recommend",
        "Analysis of data shows",
        "Recent survey indicates",
        "Official statistics reveal",
        "Peer-reviewed research demonstrates"
    ]
    
    real_topics = [
        "university research", "policy changes", "economic growth",
        "public health measures", "infrastructure development",
        "educational initiatives", "environmental protection",
        "technology advancement", "healthcare improvement", "scientific discovery"
    ]
    
    # Generate fake news
    fake_texts = []
    for _ in range(n_samples // 2):
        pattern = np.random.choice(fake_patterns)
        topic = np.random.choice(fake_topics)
        additional_text = np.random.choice([
            "Click here to learn more!", "You won't believe what happens next!",
            "Share this before it's deleted!", "The truth they don't want you to see!",
            "This changes everything!", "Don't let them fool you!"
        ])
        text = f"{pattern} {topic}. {additional_text}"
        fake_texts.append(text)
    
    # Generate real news
    real_texts = []
    for _ in range(n_samples // 2):
        pattern = np.random.choice(real_patterns)
        topic = np.random.choice(real_topics)
        additional_text = np.random.choice([
            "The findings were published in a peer-reviewed journal.",
            "Further research is needed to confirm these results.",
            "The study involved a large sample size over multiple years.",
            "Experts recommend cautious interpretation of the data.",
            "The research follows established scientific methodology.",
            "Multiple institutions collaborated on this study."
        ])
        text = f"{pattern} {topic}. {additional_text}"
        real_texts.append(text)
    
    # Combine and create DataFrame
    texts = fake_texts + real_texts
    labels = [1] * len(fake_texts) + [0] * len(real_texts)  # 1 for fake, 0 for real
    
    # Shuffle the data
    combined = list(zip(texts, labels))
    np.random.shuffle(combined)
    texts, labels = zip(*combined)
    
    # Create additional features that might be useful
    df = pd.DataFrame({
        'text': texts,
        'label': labels,
        'text_length': [len(text) for text in texts],
        'word_count': [len(text.split()) for text in texts],
        'exclamation_count': [text.count('!') for text in texts],
        'caps_count': [sum(1 for c in text if c.isupper()) for text in texts]
    })
    
    return df

def load_your_data(file_path: str) -> pd.DataFrame:
    """
    Load your actual fake news dataset
    Modify this function according to your data format
    """
    try:
        # Assuming CSV format - modify as needed for your data
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        elif file_path.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(file_path)
        elif file_path.endswith('.json'):
            df = pd.read_json(file_path)
        else:
            raise ValueError("Unsupported file format. Use CSV, Excel, or JSON.")
        
        print(f"Loaded dataset with {len(df)} samples")
        print(f"Columns: {list(df.columns)}")
        
        # Basic data validation
        if df.empty:
            raise ValueError("Dataset is empty")
        
        return df
        
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        print("Using sample dataset instead...")
        return create_sample_fake_news_dataset()

def run_comprehensive_analysis(data_path: str = None, 
                             text_column: str = 'text', 
                             label_column: str = 'label',
                             output_dir: str = None) -> dict:
    """
    Run the complete fake news detection analysis
    """
    
    # Create output directory with timestamp
    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"fake_news_analysis_{timestamp}"
    
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 80)
    print("FAKE NEWS DETECTION - COMPREHENSIVE ML ANALYSIS")
    print("=" * 80)
    
    # Step 1: Initialize the pipeline
    print("\n1. Initializing ML Pipeline...")
    pipeline = FakeNewsDetectionPipeline(random_state=42)
    
    # Step 2: Load and validate data
    print("\n2. Loading and validating data...")
    if data_path:
        if os.path.exists(data_path):
            print(f"Loading data from: {data_path}")
            data = load_your_data(data_path)
        else:
            raise FileNotFoundError(f"Dataset file not found: {data_path}")
    else:
        print("No dataset path provided. Using sample dataset for demonstration...")
        data = create_sample_fake_news_dataset(2000)
    
    # Validate required columns
    if text_column not in data.columns:
        available_text_cols = [col for col in data.columns if data[col].dtype == 'object']
        if available_text_cols:
            text_column = available_text_cols[0]
            print(f"Text column '{text_column}' not found. Using '{text_column}' instead.")
        else:
            raise ValueError(f"No text column found. Available columns: {list(data.columns)}")
    
    if label_column not in data.columns:
        available_label_cols = [col for col in data.columns if data[col].dtype in ['int64', 'int32', 'float64']]
        if available_label_cols:
            label_column = available_label_cols[0]
            print(f"Label column '{label_column}' not found. Using '{label_column}' instead.")
        else:
            raise ValueError(f"No label column found. Available columns: {list(data.columns)}")
    
    # Step 3: Run the complete pipeline
    print("\n3. Running comprehensive ML pipeline...")
    print("This may take several minutes depending on dataset size...")
    
    try:
        # Save the data to a temporary file for the pipeline
        temp_data_path = os.path.join(output_dir, "temp_data.csv")
        data[[text_column, label_column]].to_csv(temp_data_path, index=False)
        
        # Run the pipeline
        results = pipeline.run_complete_pipeline(
            data_path=temp_data_path,
            text_column=text_column,
            label_column=label_column
        )
        
        # Clean up temp file
        if os.path.exists(temp_data_path):
            os.remove(temp_data_path)
        
        print("\n4. Pipeline completed successfully!")
        
        # Step 4: Save detailed results
        print("\n5. Saving detailed results...")
        
        # Save the comprehensive report
        report_path = os.path.join(output_dir, "comprehensive_report.txt")
        with open(report_path, 'w') as f:
            f.write(results['report'])
        print(f"Comprehensive report saved to: {report_path}")

        # Save JSON results (exclude model objects)
        results_path = os.path.join(output_dir, "detailed_results.json")
        import json

        def is_serializable(obj):
            """Check if object can be JSON serialized"""
            # Check for sklearn models and other non-serializable objects
            if hasattr(obj, 'fit') or hasattr(obj, 'predict'):
                return False
            # Check for specific sklearn types
            obj_type = type(obj).__name__
            if 'Classifier' in obj_type or 'Regressor' in obj_type or 'Estimator' in obj_type:
                return False
            return True

        def make_serializable(obj):
            """Recursively convert object to JSON-serializable format"""
            # Handle numpy types
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.integer, np.int32, np.int64)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float32, np.float64)):
                return float(obj)
            elif isinstance(obj, (np.bool_, bool)):
                return bool(obj)

            # Handle dictionaries
            elif isinstance(obj, dict):
                result = {}
                for key, value in obj.items():
                    if is_serializable(value):
                        result[key] = make_serializable(value)
                    else:
                        result[key] = f"<{type(value).__name__} object - not serialized>"
                return result

            # Handle lists
            elif isinstance(obj, (list, tuple)):
                return [make_serializable(item) if is_serializable(item)
                        else f"<{type(item).__name__} object>"
                        for item in obj]

            # Handle primitives
            elif isinstance(obj, (str, int, float, bool, type(None))):
                return obj

            # Non-serializable objects
            else:
                return f"<{type(obj).__name__} object - not serialized>"

        # Create serializable copy of results
        serializable_results = make_serializable(results['results'])

        try:
            with open(results_path, 'w') as f:
                json.dump(serializable_results, f, indent=2)
            print(f"Detailed results saved to: {results_path}")
        except Exception as json_error:
            print(f"Warning: Could not save JSON results: {str(json_error)}")
            print("All other results (report, visualizations) were saved successfully.")

        print(f"Detailed results saved to: {results_path}")

        # Step 5: Generate visualizations
        print("\n6. Generating visualizations...")
        try:
            viz = FakeNewsVisualizationModule(results['results'])
            viz_dir = os.path.join(output_dir, "visualizations")
            viz.save_all_visualizations(viz_dir)
            
            # Generate performance summary
            summary = viz.generate_performance_summary()
            if not summary.empty:
                summary_path = os.path.join(output_dir, "performance_summary.csv")
                summary.to_csv(summary_path, index=False)
                print(f"Performance summary saved to: {summary_path}")
                
                # Display top 5 models
                print("\nTop 5 Models by F1 Score:")
                print("-" * 40)
                print(summary.head().to_string(index=False))
        
        except Exception as viz_error:
            print(f"Warning: Visualization generation failed: {str(viz_error)}")
        
        # Step 6: Generate recommendations
        print("\n7. Generating recommendations...")
        recommendations = generate_recommendations(results['results'])
        
        recommendations_path = os.path.join(output_dir, "recommendations.txt")
        with open(recommendations_path, 'w') as f:
            f.write(recommendations)
        print(f"Recommendations saved to: {recommendations_path}")
        
        print(f"\n{'='*80}")
        print(f"ANALYSIS COMPLETE!")
        print(f"All results saved to: {output_dir}")
        print(f"{'='*80}")
        
        return {
            'results': results,
            'output_dir': output_dir,
            'data_info': {
                'samples': len(data),
                'text_column': text_column,
                'label_column': label_column
            }
        }
        
    except Exception as e:
        print(f"Error during pipeline execution: {str(e)}")
        raise

def generate_recommendations(results: dict) -> str:
    """
    Generate specific recommendations based on results
    """
    recommendations = []
    recommendations.append("FAKE NEWS DETECTION - SPECIFIC RECOMMENDATIONS")
    recommendations.append("=" * 60)
    recommendations.append("")
    
    # Model Selection Recommendations
    if 'automated_testing' in results:
        recommendations.append("1. MODEL SELECTION:")
        recommendations.append("-" * 20)
        
        # Find best performing models
        model_results = results['automated_testing']
        valid_models = {k: v for k, v in model_results.items() if 'error' not in v and 'f1_macro' in v}
        
        if valid_models:
            best_model = max(valid_models.items(), key=lambda x: x[1].get('f1_macro', 0))
            recommendations.append(f"• Primary recommendation: {best_model[0]} (F1: {best_model[1]['f1_macro']:.4f})")
            
            # Find models good for different use cases
            fastest_model = min(valid_models.items(), key=lambda x: x[1].get('training_time', float('inf')))
            most_stable = min(valid_models.items(), key=lambda x: x[1].get('cv_std', float('inf')))
            
            recommendations.append(f"• For speed: {fastest_model[0]} (Training time: {fastest_model[1]['training_time']:.2f}s)")
            recommendations.append(f"• For stability: {most_stable[0]} (CV std: {most_stable[1]['cv_std']:.4f})")
    
    # Ensemble Recommendations
    if 'ensemble_models' in results:
        recommendations.append("")
        recommendations.append("2. ENSEMBLE MODEL USAGE:")
        recommendations.append("-" * 25)
        
        ensemble_results = results['ensemble_models']
        valid_ensembles = {k: v for k, v in ensemble_results.items() if 'error' not in v}
        
        if valid_ensembles:
            best_ensemble = max(valid_ensembles.items(), key=lambda x: x[1].get('f1_macro', 0))
            recommendations.append(f"• Best ensemble: {best_ensemble[0]} (F1: {best_ensemble[1]['f1_macro']:.4f})")
            recommendations.append("• Use ensemble models for production deployment")
            recommendations.append("• Ensemble models provide better generalization")
    
    # Overfitting Analysis
    if 'generalization_analysis' in results:
        recommendations.append("")
        recommendations.append("3. OVERFITTING ANALYSIS:")
        recommendations.append("-" * 25)
        
        analysis = results['generalization_analysis']
        overfitted_models = [model for model, res in analysis.items() 
                           if 'error' not in res and res.get('is_overfitting', False)]
        
        if overfitted_models:
            recommendations.append(f"• Models showing overfitting: {', '.join(overfitted_models)}")
            recommendations.append("• Consider regularization for overfitted models")
            recommendations.append("• Collect more diverse training data")
        else:
            recommendations.append("• No significant overfitting detected")
            recommendations.append("• Models show good generalization")
    
    # Data-specific recommendations
    recommendations.append("")
    recommendations.append("4. DATA AND FEATURE ENGINEERING:")
    recommendations.append("-" * 35)
    recommendations.append("• Consider adding linguistic features (sentiment, readability)")
    recommendations.append("• Extract source credibility features")
    recommendations.append("• Implement real-time data collection")
    recommendations.append("• Add temporal features (publication time, trending topics)")
    recommendations.append("• Consider user engagement metrics")
    
    # Deployment recommendations
    recommendations.append("")
    recommendations.append("5. DEPLOYMENT CONSIDERATIONS:")
    recommendations.append("-" * 30)
    recommendations.append("• Implement model versioning and A/B testing")
    recommendations.append("• Set up monitoring for model drift")
    recommendations.append("• Create feedback loop for continuous learning")
    recommendations.append("• Implement explanation/interpretability features")
    recommendations.append("• Consider edge cases and bias detection")
    
    # Performance optimization
    recommendations.append("")
    recommendations.append("6. PERFORMANCE OPTIMIZATION:")
    recommendations.append("-" * 30)
    recommendations.append("• Use feature selection to reduce dimensionality")
    recommendations.append("• Implement caching for repeated predictions")
    recommendations.append("• Consider model quantization for faster inference")
    recommendations.append("• Batch processing for high-volume scenarios")
    
    return "\n".join(recommendations)

def quick_test_with_custom_data(texts: list, labels: list, test_name: str = "Custom Test"):
    """
    Quick test function for custom data
    """
    print(f"\n{test_name}")
    print("-" * len(test_name))
    
    # Create DataFrame
    df = pd.DataFrame({
        'text': texts,
        'label': labels
    })
    
    # Save to temp file
    temp_file = f"temp_{test_name.lower().replace(' ', '_')}.csv"
    df.to_csv(temp_file, index=False)
    
    try:
        # Run analysis
        results = run_comprehensive_analysis(
            data_path=temp_file,
            output_dir=f"results_{test_name.lower().replace(' ', '_')}"
        )
        
        print(f"Analysis completed for {test_name}")
        
    finally:
        # Cleanup
        if os.path.exists(temp_file):
            os.remove(temp_file)

# Additional utility functions
def validate_dataset_format(file_path: str) -> dict:
    """
    Validate if dataset is in correct format for the pipeline
    """
    try:
        df = pd.read_csv(file_path)
        
        validation_results = {
            'is_valid': True,
            'errors': [],
            'warnings': [],
            'suggestions': []
        }
        
        # Check if file is empty
        if df.empty:
            validation_results['is_valid'] = False
            validation_results['errors'].append("Dataset is empty")
            return validation_results
        
        # Check for text columns
        text_columns = df.select_dtypes(include=['object']).columns.tolist()
        if not text_columns:
            validation_results['is_valid'] = False
            validation_results['errors'].append("No text columns found")
        else:
            validation_results['suggestions'].append(f"Potential text columns: {text_columns}")
        
        # Check for numeric columns (potential labels)
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        if not numeric_columns:
            validation_results['warnings'].append("No numeric columns found for labels")
        else:
            validation_results['suggestions'].append(f"Potential label columns: {numeric_columns}")
        
        # Check data quality
        if df.isnull().sum().sum() > 0:
            validation_results['warnings'].append("Dataset contains missing values")
        
        # Check for reasonable text length
        if text_columns:
            text_col = text_columns[0]
            avg_length = df[text_col].str.len().mean()
            if avg_length < 10:
                validation_results['warnings'].append("Text entries seem very short")
            elif avg_length > 5000:
                validation_results['warnings'].append("Text entries are very long - may need truncation")
        
        validation_results['dataset_info'] = {
            'shape': df.shape,
            'columns': df.columns.tolist(),
            'dtypes': df.dtypes.to_dict()
        }
        
        return validation_results
        
    except Exception as e:
        return {
            'is_valid': False,
            'errors': [f"Error reading file: {str(e)}"],
            'warnings': [],
            'suggestions': []
        }

def create_config_file(output_path: str = "pipeline_config.json") -> None:
    """
    Create a configuration file for the pipeline
    """
    config = {
        "data_settings": {
            "text_column": "text",
            "label_column": "label",
            "test_size": 0.2,
            "random_state": 42
        },
        "preprocessing": {
            "max_features": 5000,
            "ngram_range": [1, 2],
            "use_stopwords": True
        },
        "models_to_test": [
            "Logistic_Regression",
            "Random_Forest", 
            "Gradient_Boosting",
            "SVM",
            "Naive_Bayes",
            "MLP"
        ],
        "hyperparameter_tuning": {
            "cv_folds": 5,
            "scoring_metric": "f1_macro",
            "n_jobs": -1
        },
        "ensemble_settings": {
            "voting_methods": ["hard", "soft"],
            "use_bagging": True
        },
        "output_settings": {
            "save_visualizations": True,
            "save_detailed_report": True,
            "create_summary": True
        }
    }
    
    import json
    with open(output_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"Configuration file created: {output_path}")
    print("Modify this file to customize pipeline behavior")


if __name__ == "__main__":
    print("Fake News Detection Pipeline - LIAR Dataset Analysis")
    print("=" * 50)

    # Run with LIAR dataset
    print("\nRunning analysis with LIAR dataset...")
    try:
        results = run_comprehensive_analysis(
            data_path='data/liar_binary_combined.csv',
            text_column='text',
            label_column='label',
            output_dir='liar_analysis_results'
        )
        print("\n✅ LIAR dataset analysis completed successfully!")
        print(f"Results saved to: liar_analysis_results/")
    except Exception as e:
        print(f"❌ Error in LIAR analysis: {str(e)}")
        import traceback

        traceback.print_exc()