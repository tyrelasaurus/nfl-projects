import pandas as pd
import os
import logging
from typing import List, Tuple, Union

logger = logging.getLogger(__name__)


class CSVExporter:
    def __init__(self, output_dir: str = "./output"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def export_rankings(self, rankings: List[Tuple[str, str, float]], week: Union[int, str]) -> str:
        if not rankings:
            logger.warning("No rankings to export")
            return ""
        
        # Create DataFrame
        df = pd.DataFrame(rankings, columns=['team_id', 'team_name', 'power_score'])
        df['week'] = week
        df['rank'] = range(1, len(df) + 1)
        
        # Reorder columns to match PRD spec: week, rank, team_id, team_name, power_score
        df = df[['week', 'rank', 'team_id', 'team_name', 'power_score']]
        
        # Generate filename
        filename = f"power_rankings_week_{week}.csv"
        filepath = os.path.join(self.output_dir, filename)
        
        try:
            df.to_csv(filepath, index=False, float_format='%.3f')
            logger.info(f"Rankings exported to {filepath}")
            return filepath
        except Exception as e:
            logger.error(f"Failed to export rankings: {e}")
            raise