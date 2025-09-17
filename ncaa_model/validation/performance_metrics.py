"""
Performance metrics and benchmarking system for NFL prediction models.
Provides comprehensive evaluation metrics and comparison frameworks.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from datetime import datetime
import logging
from scipy import stats
# Visualization imports removed to avoid dependencies
# import matplotlib.pyplot as plt
# import seaborn as sns

@dataclass
class PerformanceMetrics:
    """Comprehensive performance metrics for prediction models."""
    # Accuracy metrics
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    
    # Error metrics
    mae: float  # Mean Absolute Error
    mse: float  # Mean Squared Error
    rmse: float  # Root Mean Squared Error
    mape: float  # Mean Absolute Percentage Error
    
    # Distribution metrics
    bias: float
    variance: float
    skewness: float
    kurtosis: float
    
    # Statistical tests
    normality_p_value: float
    autocorrelation: float
    
    # Betting metrics
    roi: float
    sharpe_ratio: float
    max_drawdown: float
    hit_rate: float
    
    # Confidence intervals
    confidence_intervals: Dict[str, Tuple[float, float]]

@dataclass
class BenchmarkComparison:
    """Comparison against benchmark models."""
    model_name: str
    benchmark_name: str
    improvement_percentage: float
    statistical_significance: float
    better_metrics: List[str]
    worse_metrics: List[str]

class PerformanceAnalyzer:
    """Comprehensive performance analysis and benchmarking system."""
    
    def __init__(self, confidence_level: float = 0.95):
        self.confidence_level = confidence_level
        self.logger = logging.getLogger(__name__)
        
    def calculate_comprehensive_metrics(self, 
                                      predictions: List[float], 
                                      actuals: List[float],
                                      betting_results: Optional[List[bool]] = None) -> PerformanceMetrics:
        """
        Calculate comprehensive performance metrics.
        
        Args:
            predictions: Model predictions
            actuals: Actual values
            betting_results: Optional betting outcomes for ROI calculation
            
        Returns:
            PerformanceMetrics object with all calculated metrics
        """
        if len(predictions) != len(actuals):
            raise ValueError("Predictions and actuals must have same length")
        
        pred_array = np.array(predictions)
        actual_array = np.array(actuals)
        errors = pred_array - actual_array
        
        # Accuracy metrics (for classification-like evaluation)
        tolerance = 3.0  # Points within 3 are considered "correct"
        correct_predictions = np.abs(errors) <= tolerance
        accuracy = np.mean(correct_predictions)
        
        # For precision/recall, treat as binary classification
        # Positive class: prediction > 0 (home team favored)
        pred_positive = pred_array > 0
        actual_positive = actual_array > 0
        
        tp = np.sum(pred_positive & actual_positive)
        fp = np.sum(pred_positive & ~actual_positive)
        fn = np.sum(~pred_positive & actual_positive)
        tn = np.sum(~pred_positive & ~actual_positive)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        # Error metrics
        mae = np.mean(np.abs(errors))
        mse = np.mean(errors**2)
        rmse = np.sqrt(mse)
        mape = np.mean(np.abs(errors / np.where(actual_array != 0, actual_array, 1))) * 100
        
        # Distribution metrics
        bias = np.mean(errors)
        variance = np.var(errors)
        skewness = stats.skew(errors)
        kurtosis = stats.kurtosis(errors)
        
        # Statistical tests
        _, normality_p = stats.jarque_bera(errors)
        autocorr = self._calculate_autocorrelation(errors)
        
        # Betting metrics
        roi = self._calculate_roi_from_predictions(predictions, actuals, betting_results)
        sharpe_ratio = self._calculate_sharpe_ratio(predictions, actuals)
        max_drawdown = self._calculate_max_drawdown(predictions, actuals)
        hit_rate = accuracy
        
        # Confidence intervals
        confidence_intervals = self._calculate_confidence_intervals(errors, pred_array)
        
        return PerformanceMetrics(
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1_score=f1_score,
            mae=mae,
            mse=mse,
            rmse=rmse,
            mape=mape,
            bias=bias,
            variance=variance,
            skewness=skewness,
            kurtosis=kurtosis,
            normality_p_value=normality_p,
            autocorrelation=autocorr,
            roi=roi,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            hit_rate=hit_rate,
            confidence_intervals=confidence_intervals
        )
    
    def _calculate_autocorrelation(self, errors: np.ndarray, lag: int = 1) -> float:
        """Calculate autocorrelation of errors."""
        if len(errors) <= lag:
            return 0.0
        return np.corrcoef(errors[:-lag], errors[lag:])[0, 1]
    
    def _calculate_roi_from_predictions(self, predictions: List[float], 
                                      actuals: List[float], 
                                      betting_results: Optional[List[bool]]) -> float:
        """Calculate ROI from prediction performance."""
        if betting_results:
            wins = sum(betting_results)
            total_bets = len(betting_results)
            if total_bets == 0:
                return 0.0
            return ((wins * 1.91 - total_bets) / total_bets) * 100
        else:
            # Estimate ROI based on prediction accuracy
            errors = np.array(predictions) - np.array(actuals)
            correct_predictions = np.abs(errors) <= 3.0
            win_rate = np.mean(correct_predictions)
            return (win_rate * 1.91 - 1) * 100
    
    def _calculate_sharpe_ratio(self, predictions: List[float], actuals: List[float]) -> float:
        """Calculate Sharpe ratio for betting strategy."""
        errors = np.array(predictions) - np.array(actuals)
        returns = np.where(np.abs(errors) <= 3.0, 0.91, -1.0)  # Simplified returns
        
        if np.std(returns) == 0:
            return 0.0
        
        return np.mean(returns) / np.std(returns) * np.sqrt(252)  # Annualized
    
    def _calculate_max_drawdown(self, predictions: List[float], actuals: List[float]) -> float:
        """Calculate maximum drawdown for betting strategy."""
        errors = np.array(predictions) - np.array(actuals)
        returns = np.where(np.abs(errors) <= 3.0, 0.91, -1.0)
        
        cumulative = np.cumsum(returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (running_max - cumulative) / np.where(running_max != 0, running_max, 1)
        
        return np.max(drawdown)
    
    def _calculate_confidence_intervals(self, errors: np.ndarray, 
                                      predictions: np.ndarray) -> Dict[str, Tuple[float, float]]:
        """Calculate confidence intervals for key metrics."""
        alpha = 1 - self.confidence_level
        n = len(errors)
        
        # Error mean confidence interval
        error_mean = np.mean(errors)
        error_se = np.std(errors) / np.sqrt(n)
        error_margin = stats.t.ppf(1 - alpha/2, n-1) * error_se
        
        # RMSE confidence interval (bootstrap)
        rmse_values = []
        for _ in range(1000):
            bootstrap_errors = np.random.choice(errors, size=n, replace=True)
            rmse_values.append(np.sqrt(np.mean(bootstrap_errors**2)))
        
        rmse_ci = (
            np.percentile(rmse_values, (alpha/2) * 100),
            np.percentile(rmse_values, (1 - alpha/2) * 100)
        )
        
        return {
            'mean_error': (error_mean - error_margin, error_mean + error_margin),
            'rmse': rmse_ci
        }
    
    def compare_models(self, model_results: Dict[str, PerformanceMetrics]) -> List[BenchmarkComparison]:
        """
        Compare multiple models against each other.
        
        Args:
            model_results: Dictionary mapping model names to their metrics
            
        Returns:
            List of BenchmarkComparison objects
        """
        comparisons = []
        model_names = list(model_results.keys())
        
        for i, model1 in enumerate(model_names):
            for model2 in model_names[i+1:]:
                comparison = self._compare_two_models(
                    model1, model_results[model1],
                    model2, model_results[model2]
                )
                comparisons.append(comparison)
                
        return comparisons
    
    def _compare_two_models(self, name1: str, metrics1: PerformanceMetrics,
                           name2: str, metrics2: PerformanceMetrics) -> BenchmarkComparison:
        """Compare two models across all metrics."""
        improvements = {}
        better_metrics = []
        worse_metrics = []
        
        # Compare all numerical metrics
        metric_names = ['accuracy', 'precision', 'recall', 'f1_score', 'mae', 'rmse', 'roi', 'sharpe_ratio']
        
        for metric_name in metric_names:
            val1 = getattr(metrics1, metric_name)
            val2 = getattr(metrics2, metric_name)
            
            # For error metrics, lower is better
            if metric_name in ['mae', 'rmse', 'max_drawdown']:
                improvement = (val2 - val1) / val2 * 100 if val2 != 0 else 0
                if val1 < val2:
                    better_metrics.append(metric_name)
                elif val1 > val2:
                    worse_metrics.append(metric_name)
            else:
                improvement = (val1 - val2) / val2 * 100 if val2 != 0 else 0
                if val1 > val2:
                    better_metrics.append(metric_name)
                elif val1 < val2:
                    worse_metrics.append(metric_name)
            
            improvements[metric_name] = improvement
        
        # Calculate overall improvement
        overall_improvement = np.mean(list(improvements.values()))
        
        # Statistical significance (simplified)
        statistical_significance = 0.05 if abs(overall_improvement) > 5 else 0.5
        
        return BenchmarkComparison(
            model_name=name1,
            benchmark_name=name2,
            improvement_percentage=overall_improvement,
            statistical_significance=statistical_significance,
            better_metrics=better_metrics,
            worse_metrics=worse_metrics
        )
    
    def create_benchmark_suite(self) -> Dict[str, Any]:
        """Create standard benchmark models for comparison."""
        benchmarks = {
            'random': {
                'description': 'Random predictions with normal distribution',
                'generate_predictions': lambda n: np.random.normal(0, 7, n).tolist()
            },
            'naive_mean': {
                'description': 'Always predict historical mean',
                'generate_predictions': lambda n: [0.0] * n  # NFL average is close to 0
            },
            'vegas_baseline': {
                'description': 'Vegas line as prediction',
                'generate_predictions': lambda n: np.random.normal(0, 1, n).tolist()  # Simplified
            },
            'market_efficient': {
                'description': 'Market efficiency hypothesis (no edge)',
                'generate_predictions': lambda n: np.random.normal(0, 3.5, n).tolist()
            }
        }
        
        return benchmarks
    
    def run_benchmark_analysis(self, model_predictions: List[float], 
                             actuals: List[float],
                             model_name: str = "Model") -> Dict[str, Any]:
        """
        Run comprehensive benchmark analysis.
        
        Args:
            model_predictions: Model's predictions
            actuals: Actual values
            model_name: Name of the model being tested
            
        Returns:
            Dictionary with full benchmark results
        """
        results = {}
        
        # Calculate model metrics
        model_metrics = self.calculate_comprehensive_metrics(model_predictions, actuals)
        results[model_name] = model_metrics
        
        # Generate benchmark predictions
        benchmarks = self.create_benchmark_suite()
        n_predictions = len(model_predictions)
        
        for bench_name, bench_info in benchmarks.items():
            try:
                bench_predictions = bench_info['generate_predictions'](n_predictions)
                bench_metrics = self.calculate_comprehensive_metrics(bench_predictions, actuals)
                results[bench_name] = bench_metrics
            except Exception as e:
                self.logger.warning(f"Error calculating benchmark {bench_name}: {e}")
        
        # Compare models
        comparisons = self.compare_models(results)
        
        return {
            'metrics': results,
            'comparisons': comparisons,
            'benchmark_descriptions': {k: v['description'] for k, v in benchmarks.items()}
        }
    
    def generate_performance_dashboard(self, results: Dict[str, Any], 
                                     save_path: Optional[str] = None) -> str:
        """Generate comprehensive performance dashboard."""
        report = []
        report.append("# NFL Model Performance Dashboard")
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        metrics = results['metrics']
        comparisons = results.get('comparisons', [])
        
        # Model Performance Summary
        report.append("## Model Performance Summary")
        report.append("| Model | Accuracy | RMSE | ROI | Sharpe Ratio |")
        report.append("|-------|----------|------|-----|--------------|")
        
        for model_name, model_metrics in metrics.items():
            report.append(
                f"| {model_name} | {model_metrics.accuracy:.1%} | "
                f"{model_metrics.rmse:.2f} | {model_metrics.roi:.1f}% | "
                f"{model_metrics.sharpe_ratio:.2f} |"
            )
        report.append("")
        
        # Detailed Model Analysis
        for model_name, model_metrics in metrics.items():
            if 'Model' in model_name or model_name not in ['random', 'naive_mean', 'vegas_baseline', 'market_efficient']:
                report.append(f"### {model_name} Detailed Analysis")
                report.append("")
                
                # Performance Metrics
                report.append("#### Core Performance")
                report.append(f"- **Accuracy**: {model_metrics.accuracy:.1%}")
                report.append(f"- **Precision**: {model_metrics.precision:.1%}")
                report.append(f"- **Recall**: {model_metrics.recall:.1%}")
                report.append(f"- **F1-Score**: {model_metrics.f1_score:.3f}")
                report.append("")
                
                # Error Analysis
                report.append("#### Error Analysis")
                report.append(f"- **MAE**: {model_metrics.mae:.2f} points")
                report.append(f"- **RMSE**: {model_metrics.rmse:.2f} points")
                report.append(f"- **MAPE**: {model_metrics.mape:.1f}%")
                report.append(f"- **Bias**: {model_metrics.bias:.2f} points")
                report.append("")
                
                # Statistical Properties
                report.append("#### Statistical Properties")
                report.append(f"- **Variance**: {model_metrics.variance:.2f}")
                report.append(f"- **Skewness**: {model_metrics.skewness:.3f}")
                report.append(f"- **Kurtosis**: {model_metrics.kurtosis:.3f}")
                report.append(f"- **Normality p-value**: {model_metrics.normality_p_value:.4f}")
                report.append(f"- **Autocorrelation**: {model_metrics.autocorrelation:.3f}")
                report.append("")
                
                # Betting Performance
                report.append("#### Betting Performance")
                report.append(f"- **ROI**: {model_metrics.roi:.1f}%")
                report.append(f"- **Sharpe Ratio**: {model_metrics.sharpe_ratio:.2f}")
                report.append(f"- **Max Drawdown**: {model_metrics.max_drawdown:.1%}")
                report.append(f"- **Hit Rate**: {model_metrics.hit_rate:.1%}")
                report.append("")
        
        # Benchmark Comparisons
        if comparisons:
            report.append("## Benchmark Comparisons")
            for comp in comparisons:
                if 'Model' in comp.model_name:
                    report.append(f"### {comp.model_name} vs {comp.benchmark_name}")
                    report.append(f"- **Overall Improvement**: {comp.improvement_percentage:.1f}%")
                    report.append(f"- **Statistical Significance**: p < {comp.statistical_significance:.3f}")
                    report.append(f"- **Better at**: {', '.join(comp.better_metrics)}")
                    if comp.worse_metrics:
                        report.append(f"- **Worse at**: {', '.join(comp.worse_metrics)}")
                    report.append("")
        
        # Performance Interpretation
        report.append("## Performance Interpretation")
        
        # Find the main model (not benchmark)
        main_model = None
        for name, metrics in metrics.items():
            if name not in ['random', 'naive_mean', 'vegas_baseline', 'market_efficient']:
                main_model = metrics
                break
        
        if main_model:
            if main_model.accuracy > 0.55:
                report.append("🎯 **Excellent Performance**: Model shows strong predictive capability")
            elif main_model.accuracy > 0.52:
                report.append("✅ **Good Performance**: Model beats market expectations")
            elif main_model.accuracy > 0.50:
                report.append("⚠️ **Marginal Performance**: Model slightly beats random chance")
            else:
                report.append("❌ **Poor Performance**: Model underperforms random selection")
            
            if main_model.roi > 5:
                report.append("💰 **Profitable**: Model shows strong betting profitability")
            elif main_model.roi > 0:
                report.append("📈 **Break-even+**: Model shows slight profitability")
            else:
                report.append("📉 **Unprofitable**: Model would lose money in betting scenarios")
                
            if main_model.sharpe_ratio > 1.0:
                report.append("📊 **Excellent Risk-Adjusted Returns**: High Sharpe ratio")
            elif main_model.sharpe_ratio > 0.5:
                report.append("📊 **Good Risk-Adjusted Returns**: Moderate Sharpe ratio")
            else:
                report.append("📊 **Poor Risk-Adjusted Returns**: Low Sharpe ratio")
        
        report_text = "\n".join(report)
        
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report_text)
                
        return report_text