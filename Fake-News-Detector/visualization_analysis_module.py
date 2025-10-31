import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from sklearn.model_selection import validation_curve
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

class FakeNewsVisualizationModule:
    """
    Comprehensive visualization and analysis module for the Fake News Detection Pipeline
    """
    
    def __init__(self, pipeline_results: dict):
        """
        Initialize with results from FakeNewsDetectionPipeline
        """
        self.results = pipeline_results
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
    
    def plot_model_comparison(self, save_path: str = None) -> None:
        """
        Create comprehensive model comparison visualizations
        """
        if 'automated_testing' not in self.results:
            print("No automated testing results found.")
            return
        
        results_df = pd.DataFrame(self.results['automated_testing']).T
        results_df = results_df.dropna()
        
        # Remove error entries
        results_df = results_df[~results_df.index.str.contains('error', na=False)]
        
        if results_df.empty:
            print("No valid results to plot.")
            return
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Model Performance Comparison', fontsize=16, fontweight='bold')
        
        # 1. F1 Score Comparison
        if 'f1_macro' in results_df.columns:
            ax1 = axes[0, 0]
            f1_scores = results_df['f1_macro'].astype(float)
            bars = ax1.bar(range(len(f1_scores)), f1_scores.values, 
                          color=sns.color_palette("viridis", len(f1_scores)))
            ax1.set_title('F1 Score (Macro) Comparison', fontweight='bold')
            ax1.set_xlabel('Models')
            ax1.set_ylabel('F1 Score')
            ax1.set_xticks(range(len(f1_scores)))
            ax1.set_xticklabels(f1_scores.index, rotation=45, ha='right')
            
            # Add value labels on bars
            for i, bar in enumerate(bars):
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{height:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # 2. Accuracy vs F1 Score Scatter
        if 'accuracy' in results_df.columns and 'f1_macro' in results_df.columns:
            ax2 = axes[0, 1]
            scatter = ax2.scatter(results_df['accuracy'].astype(float), 
                                results_df['f1_macro'].astype(float),
                                s=100, c=range(len(results_df)), 
                                cmap='viridis', alpha=0.7)
            ax2.set_title('Accuracy vs F1 Score', fontweight='bold')
            ax2.set_xlabel('Accuracy')
            ax2.set_ylabel('F1 Score (Macro)')
            ax2.grid(True, alpha=0.3)
            
            # Add model labels
            for i, model in enumerate(results_df.index):
                ax2.annotate(model[:8], 
                           (results_df['accuracy'].iloc[i], results_df['f1_macro'].iloc[i]),
                           xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        # 3. Cross-validation scores with error bars
        if 'cv_mean' in results_df.columns and 'cv_std' in results_df.columns:
            ax3 = axes[1, 0]
            cv_means = results_df['cv_mean'].astype(float)
            cv_stds = results_df['cv_std'].astype(float)
            
            bars = ax3.bar(range(len(cv_means)), cv_means.values, 
                          yerr=cv_stds.values, capsize=5,
                          color=sns.color_palette("plasma", len(cv_means)))
            ax3.set_title('Cross-Validation Scores', fontweight='bold')
            ax3.set_xlabel('Models')
            ax3.set_ylabel('CV F1 Score')
            ax3.set_xticks(range(len(cv_means)))
            ax3.set_xticklabels(cv_means.index, rotation=45, ha='right')
        
        # 4. Training time comparison
        if 'training_time' in results_df.columns:
            ax4 = axes[1, 1]
            training_times = results_df['training_time'].astype(float)
            bars = ax4.bar(range(len(training_times)), training_times.values,
                          color=sns.color_palette("coolwarm", len(training_times)))
            ax4.set_title('Training Time Comparison', fontweight='bold')
            ax4.set_xlabel('Models')
            ax4.set_ylabel('Training Time (seconds)')
            ax4.set_xticks(range(len(training_times)))
            ax4.set_xticklabels(training_times.index, rotation=45, ha='right')
            ax4.set_yscale('log')  # Log scale for better visualization
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_hyperparameter_analysis(self, save_path: str = None) -> None:
        """
        Visualize hyperparameter tuning results
        """
        if 'hyperparameter_tuning' not in self.results:
            print("No hyperparameter tuning results found.")
            return
        
        tuning_results = self.results['hyperparameter_tuning']
        valid_results = {k: v for k, v in tuning_results.items() if 'error' not in v}
        
        if not valid_results:
            print("No valid hyperparameter tuning results to plot.")
            return
        
        # Create comparison plot
        models = list(valid_results.keys())
        cv_scores = [results['best_cv_score'] for results in valid_results.values()]
        test_scores = [results['test_f1'] for results in valid_results.values()]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle('Hyperparameter Tuning Results', fontsize=16, fontweight='bold')
        
        # CV vs Test scores
        x = np.arange(len(models))
        width = 0.35
        
        ax1.bar(x - width/2, cv_scores, width, label='CV Score', alpha=0.8)
        ax1.bar(x + width/2, test_scores, width, label='Test Score', alpha=0.8)
        ax1.set_title('CV vs Test Performance')
        ax1.set_xlabel('Models')
        ax1.set_ylabel('F1 Score')
        ax1.set_xticks(x)
        ax1.set_xticklabels(models, rotation=45, ha='right')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Improvement analysis
        if 'automated_testing' in self.results:
            original_results = self.results['automated_testing']
            improvements = []
            original_scores = []
            
            for model in models:
                if model in original_results and 'f1_macro' in original_results[model]:
                    orig_score = float(original_results[model]['f1_macro'])
                    tuned_score = valid_results[model]['test_f1']
                    improvement = tuned_score - orig_score
                    improvements.append(improvement)
                    original_scores.append(orig_score)
                else:
                    improvements.append(0)
                    original_scores.append(0)
            
            colors = ['green' if imp > 0 else 'red' for imp in improvements]
            bars = ax2.bar(models, improvements, color=colors, alpha=0.7)
            ax2.set_title('Performance Improvement After Tuning')
            ax2.set_xlabel('Models')
            ax2.set_ylabel('F1 Score Improvement')
            ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            ax2.tick_params(axis='x', rotation=45)
            ax2.grid(True, alpha=0.3)
            
            # Add value labels
            for bar, imp in zip(bars, improvements):
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height + (0.01 if height >= 0 else -0.01),
                        f'{imp:.3f}', ha='center', va='bottom' if height >= 0 else 'top', 
                        fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_cross_validation_analysis(self, save_path: str = None) -> None:
        """
        Comprehensive cross-validation analysis visualization
        """
        if 'cross_validation' not in self.results:
            print("No cross-validation results found.")
            return
        
        cv_results = self.results['cross_validation']
        
        # Prepare data for visualization
        all_data = []
        for cv_strategy, models in cv_results.items():
            for model, scores in models.items():
                if 'error' not in scores:
                    all_data.extend([{
                        'CV_Strategy': cv_strategy,
                        'Model': model,
                        'Score': score
                    } for score in scores['scores']])
        
        if not all_data:
            print("No valid cross-validation data to plot.")
            return
        
        df = pd.DataFrame(all_data)
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Cross-Validation Analysis', fontsize=16, fontweight='bold')
        
        # 1. Box plot by CV strategy
        ax1 = axes[0, 0]
        sns.boxplot(data=df, x='CV_Strategy', y='Score', ax=ax1)
        ax1.set_title('Score Distribution by CV Strategy')
        ax1.tick_params(axis='x', rotation=45)
        
        # 2. Box plot by model
        ax2 = axes[0, 1]
        sns.boxplot(data=df, x='Model', y='Score', ax=ax2)
        ax2.set_title('Score Distribution by Model')
        ax2.tick_params(axis='x', rotation=45)
        
        # 3. Heatmap of mean scores
        ax3 = axes[1, 0]
        pivot_mean = df.groupby(['CV_Strategy', 'Model'])['Score'].mean().unstack()
        sns.heatmap(pivot_mean, annot=True, fmt='.3f', cmap='viridis', ax=ax3)
        ax3.set_title('Mean Scores Heatmap')
        
        # 4. Stability analysis (coefficient of variation)
        ax4 = axes[1, 1]
        stability_data = df.groupby(['CV_Strategy', 'Model'])['Score'].agg(['mean', 'std']).reset_index()
        stability_data['CV'] = stability_data['std'] / stability_data['mean']
        
        for cv_strategy in stability_data['CV_Strategy'].unique():
            strategy_data = stability_data[stability_data['CV_Strategy'] == cv_strategy]
            ax4.scatter(strategy_data['mean'], strategy_data['CV'], 
                       label=cv_strategy, s=60, alpha=0.7)
        
        ax4.set_xlabel('Mean Score')
        ax4.set_ylabel('Coefficient of Variation')
        ax4.set_title('Model Stability Analysis')
        ax4.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_ensemble_comparison(self, save_path: str = None) -> None:
        """
        Visualize ensemble model performance
        """
        if 'ensemble_models' not in self.results:
            print("No ensemble model results found.")
            return
        
        ensemble_results = self.results['ensemble_models']
        valid_results = {k: v for k, v in ensemble_results.items() if 'error' not in v}
        
        if not valid_results:
            print("No valid ensemble results to plot.")
            return
        
        # Prepare data
        models = list(valid_results.keys())
        metrics = ['accuracy', 'f1_macro', 'f1_weighted']
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle('Ensemble Model Performance', fontsize=16, fontweight='bold')
        
        # 1. Grouped bar chart
        ax1 = axes[0]
        x = np.arange(len(models))
        width = 0.25
        
        for i, metric in enumerate(metrics):
            values = [valid_results[model][metric] for model in models]
            ax1.bar(x + i*width, values, width, label=metric.replace('_', ' ').title(), alpha=0.8)
        
        ax1.set_title('Ensemble Performance Metrics')
        ax1.set_xlabel('Ensemble Methods')
        ax1.set_ylabel('Score')
        ax1.set_xticks(x + width)
        ax1.set_xticklabels(models)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Radar chart for comprehensive comparison
        ax2 = axes[1]
        
        # Create radar chart
        angles = np.linspace(0, 2*np.pi, len(metrics), endpoint=False).tolist()
        angles += angles[:1]  # Complete the circle
        
        for model in models:
            values = [valid_results[model][metric] for metric in metrics]
            values += values[:1]  # Complete the circle
            
            ax2.plot(angles, values, 'o-', linewidth=2, label=model)
            ax2.fill(angles, values, alpha=0.25)
        
        ax2.set_xticks(angles[:-1])
        ax2.set_xticklabels([m.replace('_', ' ').title() for m in metrics])
        ax2.set_ylim(0, 1)
        ax2.set_title('Ensemble Performance Radar')
        ax2.legend(bbox_to_anchor=(1.3, 1), loc='upper left')
        ax2.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_learning_curves(self, save_path: str = None) -> None:
        """
        Plot learning curves for generalization analysis
        """
        if 'generalization_analysis' not in self.results:
            print("No generalization analysis results found.")
            return
        
        analysis_results = self.results['generalization_analysis']
        valid_results = {k: v for k, v in analysis_results.items() if 'error' not in v}
        
        if not valid_results:
            print("No valid learning curve data to plot.")
            return
        
        n_models = len(valid_results)
        cols = 2
        rows = (n_models + 1) // 2
        
        fig, axes = plt.subplots(rows, cols, figsize=(15, 5*rows))
        fig.suptitle('Learning Curves Analysis', fontsize=16, fontweight='bold')
        
        if n_models == 1:
            axes = [axes]
        elif rows == 1:
            axes = axes.reshape(1, -1)
        
        for idx, (model_name, results) in enumerate(valid_results.items()):
            row = idx // cols
            col = idx % cols
            ax = axes[row, col] if rows > 1 else axes[col]
            
            train_sizes = results['train_sizes']
            train_mean = results['train_scores_mean']
            train_std = results['train_scores_std']
            val_mean = results['val_scores_mean']
            val_std = results['val_scores_std']
            
            # Plot training and validation curves
            ax.plot(train_sizes, train_mean, 'o-', color='blue', label='Training Score')
            ax.fill_between(train_sizes, 
                           np.array(train_mean) - np.array(train_std),
                           np.array(train_mean) + np.array(train_std),
                           alpha=0.3, color='blue')
            
            ax.plot(train_sizes, val_mean, 'o-', color='red', label='Validation Score')
            ax.fill_between(train_sizes,
                           np.array(val_mean) - np.array(val_std),
                           np.array(val_mean) + np.array(val_std),
                           alpha=0.3, color='red')
            
            ax.set_title(f'{model_name} Learning Curve')
            ax.set_xlabel('Training Set Size')
            ax.set_ylabel('F1 Score')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Add overfitting indicator
            if results['is_overfitting']:
                ax.text(0.05, 0.95, 'OVERFITTING DETECTED', 
                       transform=ax.transAxes, fontsize=10, 
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="red", alpha=0.7),
                       verticalalignment='top', color='white', fontweight='bold')
        
        # Hide empty subplots
        if n_models % 2 == 1 and rows > 1:
            axes[rows-1, cols-1].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def create_interactive_dashboard(self) -> None:
        """
        Create an interactive Plotly dashboard
        """
        try:
            # Model comparison data
            if 'automated_testing' in self.results:
                results_df = pd.DataFrame(self.results['automated_testing']).T
                results_df = results_df.dropna()
                results_df = results_df[~results_df.index.str.contains('error', na=False)]
                
                if not results_df.empty:
                    # Interactive scatter plot
                    fig = go.Figure()
                    
                    if 'accuracy' in results_df.columns and 'f1_macro' in results_df.columns:
                        fig.add_trace(go.Scatter(
                            x=results_df['accuracy'].astype(float),
                            y=results_df['f1_macro'].astype(float),
                            mode='markers+text',
                            text=results_df.index,
                            textposition="top center",
                            marker=dict(size=12, opacity=0.7),
                            name='Models'
                        ))
                        
                        fig.update_layout(
                            title='Interactive Model Performance Comparison',
                            xaxis_title='Accuracy',
                            yaxis_title='F1 Score (Macro)',
                            hovermode='closest'
                        )
                        
                        fig.show()
            
            # Ensemble comparison
            if 'ensemble_models' in self.results:
                ensemble_results = self.results['ensemble_models']
                valid_results = {k: v for k, v in ensemble_results.items() if 'error' not in v}
                
                if valid_results:
                    models = list(valid_results.keys())
                    metrics = ['accuracy', 'f1_macro', 'f1_weighted']
                    
                    fig = go.Figure()
                    
                    for metric in metrics:
                        values = [valid_results[model][metric] for model in models]
                        fig.add_trace(go.Bar(
                            name=metric.replace('_', ' ').title(),
                            x=models,
                            y=values
                        ))
                    
                    fig.update_layout(
                        title='Ensemble Model Performance Comparison',
                        xaxis_title='Ensemble Methods',
                        yaxis_title='Score',
                        barmode='group'
                    )
                    
                    fig.show()
        
        except ImportError:
            print("Plotly not available. Install with: pip install plotly")
    
    def generate_performance_summary(self) -> pd.DataFrame:
        """
        Generate a comprehensive performance summary table
        """
        summary_data = []
        
        # Add individual model results
        if 'automated_testing' in self.results:
            for model, metrics in self.results['automated_testing'].items():
                if 'error' not in metrics:
                    summary_data.append({
                        'Model': model,
                        'Type': 'Individual',
                        'F1_Macro': metrics.get('f1_macro', np.nan),
                        'Accuracy': metrics.get('accuracy', np.nan),
                        'CV_Mean': metrics.get('cv_mean', np.nan),
                        'CV_Std': metrics.get('cv_std', np.nan),
                        'Training_Time': metrics.get('training_time', np.nan)
                    })
        
        # Add ensemble results
        if 'ensemble_models' in self.results:
            for model, metrics in self.results['ensemble_models'].items():
                if 'error' not in metrics:
                    summary_data.append({
                        'Model': model,
                        'Type': 'Ensemble',
                        'F1_Macro': metrics.get('f1_macro', np.nan),
                        'Accuracy': metrics.get('accuracy', np.nan),
                        'CV_Mean': np.nan,
                        'CV_Std': np.nan,
                        'Training_Time': np.nan
                    })
        
        # Add tuned model results
        if 'hyperparameter_tuning' in self.results:
            for model, metrics in self.results['hyperparameter_tuning'].items():
                if 'error' not in metrics:
                    summary_data.append({
                        'Model': f"{model}_Tuned",
                        'Type': 'Tuned',
                        'F1_Macro': metrics.get('test_f1', np.nan),
                        'Accuracy': metrics.get('test_accuracy', np.nan),
                        'CV_Mean': metrics.get('best_cv_score', np.nan),
                        'CV_Std': np.nan,
                        'Training_Time': np.nan
                    })
        
        if summary_data:
            df = pd.DataFrame(summary_data)
            df = df.round(4)
            return df.sort_values('F1_Macro', ascending=False)
        else:
            return pd.DataFrame()
    
    def save_all_visualizations(self, output_dir: str = 'visualizations') -> None:
        """
        Save all visualizations to specified directory
        """
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"Saving visualizations to {output_dir}...")
        
        self.plot_model_comparison(f"{output_dir}/model_comparison.png")
        self.plot_hyperparameter_analysis(f"{output_dir}/hyperparameter_analysis.png")
        self.plot_cross_validation_analysis(f"{output_dir}/cross_validation_analysis.png")
        self.plot_ensemble_comparison(f"{output_dir}/ensemble_comparison.png")
        self.plot_learning_curves(f"{output_dir}/learning_curves.png")
        
        # Save summary table
        summary_df = self.generate_performance_summary()
        summary_df.to_csv(f"{output_dir}/performance_summary.csv", index=False)
        
        print("All visualizations saved successfully!")

# Example usage with the pipeline
if __name__ == "__main__":
    # Assuming you have results from the main pipeline
    # results = pipeline.run_complete_pipeline()
    
    # Initialize visualization module
    # viz = FakeNewsVisualizationModule(results['results'])
    
    # Generate all visualizations
    # viz.save_all_visualizations()
    
    # Create interactive dashboard
    # viz.create_interactive_dashboard()
    
    # Get performance summary
    # summary = viz.generate_performance_summary()
    # print(summary)
    
    print("Visualization module ready. Use with pipeline results.")
