# Data Quality Assessment Report
**Dataset:** Test NFL Dataset with Issues
**Generated:** 2025-09-04 11:51:26

## Executive Summary
- **Total Records:** 272
- **Valid Records:** 268
- **Invalid Records:** 4
- **Overall Quality Score:** 99.4%

## Quality Metrics
| Metric | Score | Status |
|--------|-------|--------|
| Completeness | 99.7% | ✅ |
| Accuracy | 99.6% | ✅ |
| Consistency | 99.0% | ✅ |
| **Overall** | **99.4%** | **✅** |

## Issues Identified
### Completeness Issues
- **🚨 CRITICAL:** Critical field 'away_team' has 98.9% completeness (required: 100%)
  - Records affected: 3
  - Details: {
  "field": "away_team",
  "completeness": 0.9889705882352942,
  "missing_count": "3"
}

### Accuracy Issues
- **⚠️ HIGH:** Invalid home_score values outside range [0, 70]
  - Records affected: 1
  - Details: {
  "field": "home_score",
  "range": [
    0,
    70
  ],
  "invalid_values": [
    85.0
  ]
}
- **⚠️ HIGH:** Invalid away_score values outside range [0, 70]
  - Records affected: 1
  - Details: {
  "field": "away_score",
  "range": [
    0,
    70
  ],
  "invalid_values": [
    -5
  ]
}
- **⚠️ HIGH:** Invalid margin values outside range [-50, 50]
  - Records affected: 1
  - Details: {
  "field": "margin",
  "range": [
    -50,
    50
  ],
  "invalid_values": [
    999
  ]
}
- **⚠️ HIGH:** Invalid week numbers (expected 1-18)
  - Records affected: 1
  - Details: {
  "invalid_weeks": [
    25
  ]
}
- **🔶 MEDIUM:** Date validation failed: Invalid comparison between dtype=datetime64[ns, UTC] and Timestamp
  - Records affected: 0
  - Details: {
  "error": "Invalid comparison between dtype=datetime64[ns, UTC] and Timestamp"
}

### Consistency Issues
- **⚠️ HIGH:** Margin values inconsistent with score difference
  - Records affected: 6
  - Details: {
  "sample_inconsistencies": [
    {
      "home_score": 85.0,
      "away_score": 20,
      "margin": 6
    },
    {
      "home_score": 17.0,
      "away_score": -5,
      "margin": -16
    },
    ...
- **⚠️ HIGH:** home_win values inconsistent with margin
  - Records affected: 2
  - Details: {
  "inconsistent_count": 2
}
- **⚠️ HIGH:** Duplicate game IDs detected
  - Records affected: 2
  - Details: {
  "duplicate_game_ids": [
    401671867
  ]
}
- **🚨 CRITICAL:** Teams playing against themselves detected
  - Records affected: 1
  - Details: {
  "self_games": [
    {
      "home_team": "San Francisco 49ers",
      "away_team": "San Francisco 49ers",
      "game_id": 401671645
    }
  ]
}

### Anomaly Issues
- **🔶 MEDIUM:** Statistical outliers detected in home_score
  - Records affected: 2
  - Details: {
  "field": "home_score",
  "outlier_values": [
    85.0,
    70.0
  ],
  "threshold": 3.0,
  "mean": 24.24907063197026,
  "std": 11.171043369083648
}
- **🔶 MEDIUM:** Statistical outliers detected in away_score
  - Records affected: 1
  - Details: {
  "field": "away_score",
  "outlier_values": [
    51
  ],
  "threshold": 3.0,
  "mean": 21.827205882352942,
  "std": 9.490892639083393
}
- **🔶 MEDIUM:** Statistical outliers detected in margin
  - Records affected: 1
  - Details: {
  "outlier_values": [
    999
  ],
  "threshold": 2.5
}

### Business Rule Issues
- **🔵 LOW:** Unusual number of games per week detected
  - Records affected: 0
  - Details: {
  "expected_range": [
    14,
    16
  ],
  "unusual_weeks": {
    "12": 13,
    "14": 13,
    "25": 1
  }
}

## Recommendations
- 🚨 CRITICAL: Address critical data quality issues immediately before using data for analysis
-    - Fix: Critical field 'away_team' has 98.9% completeness (required: 100%)
-    - Fix: Teams playing against themselves detected
- ⚠️ HIGH PRIORITY: Address these issues to improve data reliability
-    - Invalid home_score values outside range [0, 70]
-    - Invalid away_score values outside range [0, 70]
-    - Invalid margin values outside range [-50, 50]
-    - Invalid week numbers (expected 1-18)
-    - Margin values inconsistent with score difference
-    - home_win values inconsistent with margin
-    - Duplicate game IDs detected
- ✅ Excellent data quality! Consider this validation framework for ongoing monitoring
- 🔍 Improve data collection processes to reduce missing values
- 🎯 Review data source accuracy and implement range validations
- 🔄 Add data consistency checks in your ETL pipeline
- 📈 Investigate anomalous values - they may indicate data collection issues

## Summary
🎯 **Excellent Data Quality** - This dataset meets high quality standards and is suitable for analysis and modeling.