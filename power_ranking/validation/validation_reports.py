"""
Statistical validation report generation system.
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import json
import logging

from power_ranking.validation.backtesting import PowerRankingBacktester, BacktestResult, ValidationMetrics
from power_ranking.validation.vegas_comparison import VegasLinesComparator, VegasComparisonResult
from power_ranking.models.power_rankings import PowerRankModel, PowerRankingWithConfidence

logger = logging.getLogger(__name__)


@dataclass
class ValidationReport:
    """Comprehensive validation report for power ranking model."""
    report_id: str
    timestamp: str
    model_config: Dict[str, Any]
    data_summary: Dict[str, Any]
    historical_accuracy: Dict[str, Any]
    vegas_comparison: Dict[str, Any]
    confidence_analysis: Dict[str, Any]
    stability_analysis: Dict[str, Any]
    predictive_power: Dict[str, Any]
    recommendations: List[str]
    overall_score: float


@dataclass
class ModelBenchmark:
    """Benchmark results for model performance."""
    metric_name: str
    model_value: float
    benchmark_value: float
    industry_average: float
    percentile_rank: float
    status: str  # 'excellent', 'good', 'fair', 'poor'


class StatisticalValidationReporter:
    """Generates comprehensive statistical validation reports."""
    
    def __init__(self):
        """Initialize validation reporter."""
        self.benchmarks = self._load_industry_benchmarks()
        self.report_history = []
    
    def generate_comprehensive_report(self, 
                                    model: PowerRankModel,
                                    historical_data: Dict[str, Any] = None,
                                    vegas_data: pd.DataFrame = None,
                                    season: int = 2024) -> ValidationReport:
        """
        Generate comprehensive validation report.
        
        Args:
            model: PowerRankModel to validate
            historical_data: Historical game data
            vegas_data: Vegas lines data
            season: Season to analyze
            
        Returns:
            ValidationReport with complete analysis
        """
        report_id = f"validation_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        logger.info(f"Generating comprehensive validation report: {report_id}")
        
        # Initialize backtester and comparator
        backtester = PowerRankingBacktester(model)
        vegas_comparator = VegasLinesComparator()
        
        try:
            # 1. Model Configuration Summary
            model_config = {
                'weights': model.weights,
                'rolling_window': model.rolling_window,
                'week18_weight': model.week18_weight,
                'validation_season': season
            }
            
            # 2. Data Summary
            data_summary = self._analyze_data_quality(historical_data, vegas_data)
            
            # 3. Historical Accuracy Analysis
            logger.info("Running historical accuracy analysis...")
            historical_results = backtester.validate_historical_accuracy(season=season)
            historical_analysis = self._analyze_historical_performance(historical_results)
            
            # 4. Vegas Lines Comparison
            vegas_analysis = {}
            if vegas_data is not None and not vegas_data.empty:
                logger.info("Running Vegas lines comparison...")
                # This would need model predictions to compare
                vegas_analysis = self._analyze_vegas_performance(model, vegas_data, season)
            
            # 5. Confidence Analysis
            logger.info("Analyzing confidence intervals...")
            confidence_analysis = self._analyze_confidence_intervals(model, historical_data)
            
            # 6. Stability Analysis
            logger.info("Running stability analysis...")
            stability_results = backtester.validate_ranking_stability(season=season)
            stability_analysis = self._analyze_stability_performance(stability_results)
            
            # 7. Predictive Power Analysis
            logger.info("Analyzing predictive power...")
            predictive_metrics = backtester.validate_predictive_power(season=season)
            predictive_analysis = self._analyze_predictive_performance(predictive_metrics)
            
            # 8. Generate Recommendations
            recommendations = self._generate_recommendations(
                historical_analysis, vegas_analysis, confidence_analysis, 
                stability_analysis, predictive_analysis
            )
            
            # 9. Calculate Overall Score
            overall_score = self._calculate_overall_score(
                historical_analysis, vegas_analysis, confidence_analysis,
                stability_analysis, predictive_analysis
            )
            
            report = ValidationReport(
                report_id=report_id,
                timestamp=datetime.now().isoformat(),
                model_config=model_config,
                data_summary=data_summary,
                historical_accuracy=historical_analysis,
                vegas_comparison=vegas_analysis,
                confidence_analysis=confidence_analysis,
                stability_analysis=stability_analysis,
                predictive_power=predictive_analysis,
                recommendations=recommendations,
                overall_score=overall_score
            )
            
            self.report_history.append(report)
            logger.info(f"Validation report completed. Overall score: {overall_score:.2f}")
            
            return report
        
        except Exception as e:
            logger.error(f"Error generating validation report: {e}")
            raise
    
    def compare_models(self, 
                      models: Dict[str, PowerRankModel], 
                      historical_data: Dict[str, Any] = None,
                      season: int = 2024) -> Dict[str, Any]:
        """
        Compare multiple models against each other.
        
        Args:
            models: Dictionary of model_name -> PowerRankModel
            historical_data: Historical data for comparison
            season: Season to analyze
            
        Returns:
            Dictionary with comparison results
        """
        logger.info(f"Comparing {len(models)} models")
        
        comparison_results = {
            'comparison_timestamp': datetime.now().isoformat(),
            'season': season,
            'models': {},
            'rankings': {},
            'summary': {}
        }
        
        model_reports = {}
        
        # Generate reports for each model
        for model_name, model in models.items():
            logger.info(f"Validating model: {model_name}")
            report = self.generate_comprehensive_report(model, historical_data, season=season)
            model_reports[model_name] = report
            
            comparison_results['models'][model_name] = {
                'overall_score': report.overall_score,
                'historical_accuracy': report.historical_accuracy.get('accuracy', 0.0),
                'predictive_power': report.predictive_power.get('predictive_power', 0.0),
                'stability_score': report.stability_analysis.get('mean_stability', 0.0)
            }
        
        # Rank models by performance
        model_rankings = sorted(
            models.keys(), 
            key=lambda m: model_reports[m].overall_score, 
            reverse=True
        )
        
        comparison_results['rankings'] = {
            rank + 1: model_name for rank, model_name in enumerate(model_rankings)
        }
        
        # Generate comparison summary
        best_model = model_rankings[0]
        worst_model = model_rankings[-1]
        
        comparison_results['summary'] = {
            'best_model': best_model,
            'best_score': model_reports[best_model].overall_score,
            'worst_model': worst_model,
            'worst_score': model_reports[worst_model].overall_score,
            'score_spread': model_reports[best_model].overall_score - model_reports[worst_model].overall_score,
            'total_models_compared': len(models)
        }
        
        return comparison_results
    
    def generate_performance_dashboard(self, reports: List[ValidationReport] = None) -> Dict[str, Any]:
        """
        Generate performance dashboard data.
        
        Args:
            reports: List of validation reports (uses history if None)
            
        Returns:
            Dashboard data dictionary
        """
        if reports is None:
            reports = self.report_history
        
        if not reports:
            return {'error': 'No reports available for dashboard'}
        
        logger.info(f"Generating dashboard from {len(reports)} reports")
        
        # Time series data
        timestamps = [report.timestamp for report in reports]
        overall_scores = [report.overall_score for report in reports]
        
        # Latest performance metrics
        latest_report = reports[-1]
        
        dashboard = {
            'generated_at': datetime.now().isoformat(),
            'total_reports': len(reports),
            'time_series': {
                'timestamps': timestamps,
                'overall_scores': overall_scores,
                'historical_accuracy': [r.historical_accuracy.get('accuracy', 0) for r in reports],
                'stability_scores': [r.stability_analysis.get('mean_stability', 0) for r in reports]
            },
            'current_performance': {
                'overall_score': latest_report.overall_score,
                'accuracy': latest_report.historical_accuracy.get('accuracy', 0),
                'stability': latest_report.stability_analysis.get('mean_stability', 0),
                'predictive_power': latest_report.predictive_power.get('predictive_power', 0)
            },
            'benchmarks': self._get_benchmark_comparisons(latest_report),
            'trends': self._calculate_performance_trends(reports),
            'alerts': self._generate_performance_alerts(latest_report)
        }
        
        return dashboard
    
    def export_report(self, report: ValidationReport, 
                     format: str = 'json', 
                     output_path: str = None) -> str:
        """
        Export validation report to file.
        
        Args:
            report: ValidationReport to export
            format: Export format ('json', 'csv', 'html')
            output_path: Output file path
            
        Returns:
            Path to exported file
        """
        if output_path is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_path = f"validation_report_{timestamp}.{format}"
        
        logger.info(f"Exporting report to {output_path} in {format} format")
        
        try:
            if format.lower() == 'json':
                with open(output_path, 'w') as f:
                    json.dump(asdict(report), f, indent=2, default=str)
            
            elif format.lower() == 'csv':
                # Convert report to flat structure for CSV
                flat_data = self._flatten_report_for_csv(report)
                pd.DataFrame([flat_data]).to_csv(output_path, index=False)
            
            elif format.lower() == 'html':
                html_content = self._generate_html_report(report)
                with open(output_path, 'w') as f:
                    f.write(html_content)
            
            else:
                raise ValueError(f"Unsupported export format: {format}")
            
            logger.info(f"Report exported successfully to {output_path}")
            return output_path
        
        except Exception as e:
            logger.error(f"Error exporting report: {e}")
            raise
    
    def _analyze_data_quality(self, historical_data: Dict[str, Any] = None, 
                            vegas_data: pd.DataFrame = None) -> Dict[str, Any]:
        """Analyze quality of input data."""
        analysis = {
            'historical_data_available': historical_data is not None,
            'vegas_data_available': vegas_data is not None and not vegas_data.empty,
            'data_quality_score': 0.0,
            'completeness': {},
            'issues': []
        }
        
        quality_score = 0.0
        
        if historical_data:
            # Analyze historical data quality
            events = historical_data.get('events', [])
            analysis['completeness']['total_games'] = len(events)
            
            completed_games = [e for e in events if e.get('status', {}).get('type', {}).get('name') == 'STATUS_FINAL']
            analysis['completeness']['completed_games'] = len(completed_games)
            
            if len(events) > 0:
                completion_rate = len(completed_games) / len(events)
                quality_score += min(0.5, completion_rate)
            
            if len(completed_games) >= 200:  # Reasonable sample size
                quality_score += 0.3
        
        if vegas_data is not None and not vegas_data.empty:
            analysis['completeness']['vegas_games'] = len(vegas_data)
            quality_score += 0.2
        
        analysis['data_quality_score'] = min(1.0, quality_score)
        
        return analysis
    
    def _analyze_historical_performance(self, backtest_result: BacktestResult) -> Dict[str, Any]:
        """Analyze historical backtesting performance."""
        if backtest_result.total_predictions == 0:
            return {'error': 'No historical predictions available'}
        
        # Classify performance
        accuracy = backtest_result.accuracy
        if accuracy >= 0.60:
            accuracy_grade = 'excellent'
        elif accuracy >= 0.55:
            accuracy_grade = 'good' 
        elif accuracy >= 0.50:
            accuracy_grade = 'fair'
        else:
            accuracy_grade = 'poor'
        
        return {
            'total_predictions': backtest_result.total_predictions,
            'accuracy': accuracy,
            'accuracy_grade': accuracy_grade,
            'mean_absolute_error': backtest_result.mean_absolute_error,
            'rmse': backtest_result.root_mean_squared_error,
            'correlation': backtest_result.correlation_coefficient,
            'confidence_intervals': backtest_result.confidence_intervals,
            'period': backtest_result.period
        }
    
    def _analyze_vegas_performance(self, model: PowerRankModel, 
                                 vegas_data: pd.DataFrame, 
                                 season: int) -> Dict[str, Any]:
        """Analyze performance against Vegas lines."""
        # This would require generating model predictions for comparison
        # For now, return placeholder analysis
        return {
            'vegas_data_available': True,
            'total_games_compared': len(vegas_data) if vegas_data is not None else 0,
            'analysis_pending': 'Requires model predictions for comparison'
        }
    
    def _analyze_confidence_intervals(self, model: PowerRankModel, 
                                    historical_data: Dict[str, Any] = None) -> Dict[str, Any]:
        """Analyze confidence interval accuracy and calibration."""
        analysis = {
            'confidence_intervals_implemented': True,
            'calibration_analysis': 'Available with historical data',
            'interval_coverage': 'Not calculated - requires backtesting data',
            'average_interval_width': 'Not calculated - requires predictions'
        }
        
        # In a full implementation, this would:
        # 1. Generate predictions with confidence intervals
        # 2. Check if actual results fall within intervals at expected rates
        # 3. Calculate interval coverage rates
        # 4. Assess interval calibration
        
        return analysis
    
    def _analyze_stability_performance(self, stability_results: Dict[str, float]) -> Dict[str, Any]:
        """Analyze ranking stability performance."""
        mean_stability = stability_results.get('mean_stability', 0.0)
        
        # Classify stability
        if mean_stability >= 0.8:
            stability_grade = 'excellent'
        elif mean_stability >= 0.7:
            stability_grade = 'good'
        elif mean_stability >= 0.6:
            stability_grade = 'fair'
        else:
            stability_grade = 'poor'
        
        return {
            'mean_stability': mean_stability,
            'stability_grade': stability_grade,
            'std_stability': stability_results.get('std_stability', 0.0),
            'min_stability': stability_results.get('min_stability', 0.0),
            'max_stability': stability_results.get('max_stability', 1.0),
            'weeks_analyzed': stability_results.get('weeks_analyzed', 0)
        }
    
    def _analyze_predictive_performance(self, metrics: ValidationMetrics) -> Dict[str, Any]:
        """Analyze predictive power metrics."""
        predictive_power = metrics.predictive_power
        
        # Classify predictive power (R² interpretation)
        if predictive_power >= 0.25:
            prediction_grade = 'excellent'
        elif predictive_power >= 0.15:
            prediction_grade = 'good'
        elif predictive_power >= 0.10:
            prediction_grade = 'fair'
        else:
            prediction_grade = 'poor'
        
        return {
            'ranking_accuracy': metrics.ranking_accuracy,
            'spread_accuracy': metrics.spread_accuracy,
            'predictive_power': predictive_power,
            'prediction_grade': prediction_grade,
            'ranking_correlation': metrics.ranking_correlation,
            'calibration_score': metrics.calibration_score,
            'stability_score': metrics.stability_score
        }
    
    def _generate_recommendations(self, historical: Dict, vegas: Dict, confidence: Dict,
                                stability: Dict, predictive: Dict) -> List[str]:
        """Generate actionable recommendations."""
        recommendations = []
        
        # Historical accuracy recommendations
        accuracy = historical.get('accuracy', 0.0)
        if accuracy < 0.55:
            recommendations.append("Consider adjusting model weights - accuracy below industry average")
        
        # Stability recommendations  
        stability_score = stability.get('mean_stability', 0.0)
        if stability_score < 0.7:
            recommendations.append("Rankings show high volatility - consider increasing recency factor weight")
        
        # Predictive power recommendations
        pred_power = predictive.get('predictive_power', 0.0)
        if pred_power < 0.15:
            recommendations.append("Low predictive power - consider adding new features or adjusting methodology")
        
        # Correlation recommendations
        correlation = historical.get('correlation', 0.0)
        if correlation < 0.3:
            recommendations.append("Low correlation with actual outcomes - review statistical methodology")
        
        # Error rate recommendations
        mae = historical.get('mean_absolute_error', float('inf'))
        if mae > 10.0:
            recommendations.append("High prediction errors - consider improving margin prediction methodology")
        
        if not recommendations:
            recommendations.append("Model performance is within acceptable ranges")
        
        return recommendations
    
    def _calculate_overall_score(self, historical: Dict, vegas: Dict, confidence: Dict,
                               stability: Dict, predictive: Dict) -> float:
        """Calculate overall model performance score (0-100)."""
        score_components = []
        
        # Historical accuracy (30% weight)
        accuracy = historical.get('accuracy', 0.0)
        accuracy_score = min(100, accuracy * 200)  # Scale to 0-100 (60% accuracy = 100 points)
        score_components.append((accuracy_score, 0.30))
        
        # Stability (20% weight)
        stability_score = stability.get('mean_stability', 0.0) * 100
        score_components.append((stability_score, 0.20))
        
        # Predictive power (25% weight)
        pred_power = predictive.get('predictive_power', 0.0)
        pred_score = min(100, pred_power * 400)  # Scale to 0-100 (25% R² = 100 points)
        score_components.append((pred_score, 0.25))
        
        # Correlation (15% weight)
        correlation = historical.get('correlation', 0.0)
        corr_score = max(0, correlation * 100)  # Scale to 0-100
        score_components.append((corr_score, 0.15))
        
        # Error rate (10% weight) - inverse scoring
        mae = historical.get('mean_absolute_error', 10.0)
        error_score = max(0, 100 - (mae * 5))  # Lower error = higher score
        score_components.append((error_score, 0.10))
        
        # Calculate weighted average
        weighted_score = sum(score * weight for score, weight in score_components)
        
        return round(min(100.0, max(0.0, weighted_score)), 2)
    
    def _load_industry_benchmarks(self) -> Dict[str, Dict[str, float]]:
        """Load industry benchmark values for comparison."""
        return {
            'accuracy': {
                'excellent': 0.60,
                'good': 0.55,
                'fair': 0.50,
                'industry_average': 0.53
            },
            'correlation': {
                'excellent': 0.40,
                'good': 0.30,
                'fair': 0.20,
                'industry_average': 0.25
            },
            'stability': {
                'excellent': 0.80,
                'good': 0.70,
                'fair': 0.60,
                'industry_average': 0.65
            },
            'predictive_power': {
                'excellent': 0.25,
                'good': 0.15,
                'fair': 0.10,
                'industry_average': 0.12
            }
        }
    
    def _get_benchmark_comparisons(self, report: ValidationReport) -> List[ModelBenchmark]:
        """Get benchmark comparisons for the report."""
        benchmarks = []
        
        # Accuracy benchmark
        accuracy = report.historical_accuracy.get('accuracy', 0.0)
        benchmarks.append(ModelBenchmark(
            metric_name='Prediction Accuracy',
            model_value=accuracy,
            benchmark_value=self.benchmarks['accuracy']['good'],
            industry_average=self.benchmarks['accuracy']['industry_average'],
            percentile_rank=self._calculate_percentile(accuracy, 'accuracy'),
            status=self._get_performance_status(accuracy, 'accuracy')
        ))
        
        return benchmarks
    
    def _calculate_percentile(self, value: float, metric: str) -> float:
        """Calculate percentile rank for a metric value."""
        benchmarks = self.benchmarks.get(metric, {})
        industry_avg = benchmarks.get('industry_average', value)
        
        # Simple percentile calculation (in practice, would use historical data)
        if value >= benchmarks.get('excellent', value):
            return 90.0
        elif value >= benchmarks.get('good', value):
            return 70.0
        elif value >= industry_avg:
            return 50.0
        else:
            return 25.0
    
    def _get_performance_status(self, value: float, metric: str) -> str:
        """Get performance status for a metric value."""
        benchmarks = self.benchmarks.get(metric, {})
        
        if value >= benchmarks.get('excellent', float('inf')):
            return 'excellent'
        elif value >= benchmarks.get('good', float('inf')):
            return 'good'
        elif value >= benchmarks.get('fair', float('inf')):
            return 'fair'
        else:
            return 'poor'
    
    def _calculate_performance_trends(self, reports: List[ValidationReport]) -> Dict[str, Any]:
        """Calculate performance trends over time."""
        if len(reports) < 2:
            return {'insufficient_data': True}
        
        # Calculate trends for key metrics
        overall_scores = [r.overall_score for r in reports]
        accuracies = [r.historical_accuracy.get('accuracy', 0) for r in reports]
        
        return {
            'overall_score_trend': self._calculate_trend(overall_scores),
            'accuracy_trend': self._calculate_trend(accuracies),
            'trend_direction': 'improving' if overall_scores[-1] > overall_scores[0] else 'declining',
            'reports_analyzed': len(reports)
        }
    
    def _calculate_trend(self, values: List[float]) -> float:
        """Calculate linear trend slope."""
        if len(values) < 2:
            return 0.0
        
        x = np.arange(len(values))
        y = np.array(values)
        
        # Linear regression slope
        slope = np.polyfit(x, y, 1)[0]
        return float(slope)
    
    def _generate_performance_alerts(self, report: ValidationReport) -> List[Dict[str, str]]:
        """Generate performance alerts."""
        alerts = []
        
        # Critical alerts
        if report.overall_score < 40:
            alerts.append({
                'level': 'critical',
                'message': 'Overall model performance is critically low',
                'action': 'Immediate model review and retuning required'
            })
        
        # Warning alerts
        accuracy = report.historical_accuracy.get('accuracy', 0.0)
        if accuracy < 0.50:
            alerts.append({
                'level': 'warning',
                'message': 'Prediction accuracy below 50%',
                'action': 'Review model methodology and parameters'
            })
        
        return alerts
    
    def _flatten_report_for_csv(self, report: ValidationReport) -> Dict[str, Any]:
        """Flatten report structure for CSV export."""
        flat = {
            'report_id': report.report_id,
            'timestamp': report.timestamp,
            'overall_score': report.overall_score,
        }
        
        # Flatten nested dictionaries with prefixes
        for key, value in report.historical_accuracy.items():
            flat[f'historical_{key}'] = value
        
        for key, value in report.stability_analysis.items():
            flat[f'stability_{key}'] = value
        
        for key, value in report.predictive_power.items():
            flat[f'predictive_{key}'] = value
        
        return flat
    
    def _generate_html_report(self, report: ValidationReport) -> str:
        """Generate HTML report."""
        # Basic HTML template (in practice, would use a proper template engine)
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Validation Report - {report.report_id}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ background-color: #f0f0f0; padding: 20px; }}
                .section {{ margin: 20px 0; padding: 15px; border: 1px solid #ddd; }}
                .score {{ font-size: 24px; font-weight: bold; color: #333; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Statistical Validation Report</h1>
                <p>Report ID: {report.report_id}</p>
                <p>Generated: {report.timestamp}</p>
                <div class="score">Overall Score: {report.overall_score}/100</div>
            </div>
            
            <div class="section">
                <h2>Historical Accuracy</h2>
                <p>Accuracy: {report.historical_accuracy.get('accuracy', 0):.1%}</p>
                <p>Total Predictions: {report.historical_accuracy.get('total_predictions', 0)}</p>
            </div>
            
            <div class="section">
                <h2>Recommendations</h2>
                <ul>
                    {''.join(f'<li>{rec}</li>' for rec in report.recommendations)}
                </ul>
            </div>
        </body>
        </html>
        """
        return html


def generate_sample_validation_report() -> ValidationReport:
    """Generate a sample validation report for testing."""
    return ValidationReport(
        report_id="sample_report_001",
        timestamp=datetime.now().isoformat(),
        model_config={
            'weights': {'season_avg_margin': 0.5, 'rolling_avg_margin': 0.25, 'sos': 0.2, 'recency_factor': 0.05},
            'rolling_window': 5
        },
        data_summary={'data_quality_score': 0.85, 'total_games': 272},
        historical_accuracy={'accuracy': 0.58, 'total_predictions': 128, 'correlation': 0.32},
        vegas_comparison={'analysis_pending': True},
        confidence_analysis={'confidence_intervals_implemented': True},
        stability_analysis={'mean_stability': 0.72, 'stability_grade': 'good'},
        predictive_power={'predictive_power': 0.18, 'ranking_accuracy': 0.61},
        recommendations=['Model performance is within acceptable ranges'],
        overall_score=68.5
    )