"""
Data analysis skill for AgentForge.

I've created this skill to handle common data analysis tasks using pandas.
It can transform data, compute statistics, and even generate visualizations.
Perfect for building data pipelines in your agents.

Author: Prof. Shahab Anbarjafari
"""

from __future__ import annotations

import base64
import io
import logging
from typing import Any, Dict, List, Optional, Union

import pandas as pd

from agentforge.core import Skill

logger = logging.getLogger("agentforge.skills.analysis")


class DataAnalysisSkill(Skill):
    """
    Analyze and transform data with pandas.
    
    This skill performs data analysis operations. It can work with DataFrames,
    dictionaries, or CSV/JSON data. Results can include statistics, 
    transformations, and visualizations.
    
    Input (one of these):
        - dataframe: A pandas DataFrame
        - data: A dictionary or list of dictionaries
        - csv_data: CSV string or path
        - json_data: JSON string or path
        
    Options:
        - operations: List of operations to perform (see below)
        - group_by: Column(s) to group by
        - aggregate: Aggregation functions to apply
        - filter: Filter condition string
        - sort_by: Column(s) to sort by
        - limit: Maximum rows to return
        - plot: Visualization config (type, x, y, title)
        
    Output:
        - result: The processed DataFrame (as dict)
        - statistics: Summary statistics
        - plot_base64: Base64-encoded plot image (if requested)
        
    Example:
        >>> analyzer = DataAnalysisSkill()
        >>> result = analyzer.execute({
        ...     "data": [{"name": "Alice", "score": 85}, {"name": "Bob", "score": 92}],
        ...     "operations": ["describe"],
        ...     "sort_by": "score",
        ... })
    """
    
    name = "data_analysis"
    description = "Analyze and transform data using pandas"
    requires_llm = False
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform data analysis operations.
        
        Args:
            input_data: Dictionary with data and operation configuration
            
        Returns:
            Dictionary with analysis results
        """
        # First, get or create a DataFrame
        df = self._get_dataframe(input_data)
        
        if df is None or df.empty:
            return {
                "result": {},
                "row_count": 0,
                "message": "No data to analyze",
            }
        
        logger.info(f"Analyzing DataFrame with {len(df)} rows, {len(df.columns)} columns")
        
        result = {
            "original_shape": {"rows": len(df), "columns": len(df.columns)},
            "columns": list(df.columns),
        }
        
        # Apply filter if specified
        filter_condition = input_data.get("filter")
        if filter_condition:
            df = self._apply_filter(df, filter_condition)
            result["filtered_rows"] = len(df)
        
        # Apply group by and aggregation
        group_by = input_data.get("group_by")
        aggregate = input_data.get("aggregate")
        if group_by:
            df = self._apply_groupby(df, group_by, aggregate)
        
        # Apply sorting
        sort_by = input_data.get("sort_by")
        if sort_by:
            ascending = input_data.get("ascending", True)
            df = df.sort_values(sort_by, ascending=ascending)
        
        # Apply limit
        limit = input_data.get("limit")
        if limit:
            df = df.head(limit)
        
        # Run requested operations
        operations = input_data.get("operations", [])
        for op in operations:
            if op == "describe":
                result["statistics"] = df.describe().to_dict()
            elif op == "info":
                result["dtypes"] = {col: str(dtype) for col, dtype in df.dtypes.items()}
            elif op == "unique":
                result["unique_counts"] = {col: df[col].nunique() for col in df.columns}
            elif op == "missing":
                result["missing_counts"] = df.isnull().sum().to_dict()
            elif op == "correlation":
                numeric_df = df.select_dtypes(include=["number"])
                if not numeric_df.empty:
                    result["correlation"] = numeric_df.corr().to_dict()
        
        # Generate plot if requested
        plot_config = input_data.get("plot")
        if plot_config:
            try:
                plot_base64 = self._generate_plot(df, plot_config)
                result["plot_base64"] = plot_base64
            except Exception as e:
                logger.warning(f"Failed to generate plot: {e}")
                result["plot_error"] = str(e)
        
        # Convert result DataFrame to dict
        result["result"] = df.to_dict(orient="records")
        result["row_count"] = len(df)
        
        return result
    
    def _get_dataframe(self, input_data: Dict[str, Any]) -> Optional[pd.DataFrame]:
        """Extract or create a DataFrame from input data."""
        # Direct DataFrame
        if "dataframe" in input_data:
            return input_data["dataframe"]
        
        # Dict or list of dicts
        if "data" in input_data:
            data = input_data["data"]
            if isinstance(data, list):
                return pd.DataFrame(data)
            elif isinstance(data, dict):
                return pd.DataFrame([data])
        
        # CSV data
        if "csv_data" in input_data:
            csv_data = input_data["csv_data"]
            return pd.read_csv(io.StringIO(csv_data))
        
        # JSON data
        if "json_data" in input_data:
            json_data = input_data["json_data"]
            return pd.read_json(io.StringIO(json_data))
        
        # CSV path
        if "csv_path" in input_data:
            return pd.read_csv(input_data["csv_path"])
        
        # JSON path
        if "json_path" in input_data:
            return pd.read_json(input_data["json_path"])
        
        # Try to find content from previous skills (like scraper)
        if "content" in input_data:
            # Try to parse as table data
            content = input_data["content"]
            # This is a simplified approach - you might want to use
            # an LLM to extract structured data from text
            return None
        
        return None
    
    def _apply_filter(self, df: pd.DataFrame, condition: str) -> pd.DataFrame:
        """
        Apply a filter condition to the DataFrame.
        
        The condition should be a pandas query string, e.g., "age > 25"
        """
        try:
            return df.query(condition)
        except Exception as e:
            logger.warning(f"Failed to apply filter '{condition}': {e}")
            return df
    
    def _apply_groupby(
        self,
        df: pd.DataFrame,
        group_by: Union[str, List[str]],
        aggregate: Optional[Dict[str, str]] = None,
    ) -> pd.DataFrame:
        """Apply group by with optional aggregation."""
        if aggregate is None:
            # Default to counting
            return df.groupby(group_by).size().reset_index(name="count")
        
        # Apply specified aggregations
        agg_result = df.groupby(group_by).agg(aggregate)
        agg_result.columns = ["_".join(col).strip() for col in agg_result.columns.values]
        return agg_result.reset_index()
    
    def _generate_plot(self, df: pd.DataFrame, config: Dict[str, Any]) -> str:
        """
        Generate a plot and return it as a base64-encoded image.
        
        Supports matplotlib and plotly (if installed).
        """
        plot_type = config.get("type", "bar")
        x = config.get("x")
        y = config.get("y")
        title = config.get("title", "")
        backend = config.get("backend", "matplotlib")
        
        if backend == "plotly":
            return self._generate_plotly(df, plot_type, x, y, title)
        else:
            return self._generate_matplotlib(df, plot_type, x, y, title)
    
    def _generate_matplotlib(
        self, df: pd.DataFrame, plot_type: str, x: str, y: str, title: str
    ) -> str:
        """Generate a matplotlib plot."""
        try:
            import matplotlib
            matplotlib.use("Agg")  # Non-interactive backend
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError(
                "Matplotlib not installed. Install with: pip install matplotlib"
            )
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        if plot_type == "bar":
            df.plot.bar(x=x, y=y, ax=ax)
        elif plot_type == "line":
            df.plot.line(x=x, y=y, ax=ax)
        elif plot_type == "scatter":
            df.plot.scatter(x=x, y=y, ax=ax)
        elif plot_type == "pie":
            df.set_index(x)[y].plot.pie(ax=ax)
        elif plot_type == "hist":
            df[y].plot.hist(ax=ax)
        elif plot_type == "box":
            df.boxplot(column=y, by=x, ax=ax)
        else:
            raise ValueError(f"Unknown plot type: {plot_type}")
        
        if title:
            ax.set_title(title)
        
        # Save to buffer
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        buf.seek(0)
        
        return base64.b64encode(buf.read()).decode("utf-8")
    
    def _generate_plotly(
        self, df: pd.DataFrame, plot_type: str, x: str, y: str, title: str
    ) -> str:
        """Generate a plotly plot."""
        try:
            import plotly.express as px
        except ImportError:
            raise ImportError(
                "Plotly not installed. Install with: pip install plotly"
            )
        
        if plot_type == "bar":
            fig = px.bar(df, x=x, y=y, title=title)
        elif plot_type == "line":
            fig = px.line(df, x=x, y=y, title=title)
        elif plot_type == "scatter":
            fig = px.scatter(df, x=x, y=y, title=title)
        elif plot_type == "pie":
            fig = px.pie(df, names=x, values=y, title=title)
        elif plot_type == "hist":
            fig = px.histogram(df, x=y, title=title)
        elif plot_type == "box":
            fig = px.box(df, x=x, y=y, title=title)
        else:
            raise ValueError(f"Unknown plot type: {plot_type}")
        
        # Convert to PNG
        img_bytes = fig.to_image(format="png")
        return base64.b64encode(img_bytes).decode("utf-8")


class TextToDataSkill(Skill):
    """
    Convert unstructured text to structured data using an LLM.
    
    This is super useful when you've scraped some text and want to
    extract specific fields from it.
    
    Input:
        - text: The text to extract data from
        - fields: List of fields to extract
        - examples: (Optional) Few-shot examples for better extraction
        
    Output:
        - data: Extracted data as a dictionary
    """
    
    name = "text_to_data"
    description = "Extract structured data from unstructured text using LLM"
    requires_llm = True
    
    def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract structured data from text."""
        text = input_data.get("text") or input_data.get("content", "")
        fields = input_data.get("fields", [])
        
        if not text:
            return {"data": {}, "error": "No text provided"}
        
        if not fields:
            return {"data": {}, "error": "No fields specified"}
        
        if not self._llm:
            return {"data": {}, "error": "No LLM configured for extraction"}
        
        # Use the LLM's extract template
        try:
            response = self._llm.generate_with_template(
                "extract",
                text=text,
                fields=", ".join(fields)
            )
            
            # Try to parse as JSON
            import json
            try:
                data = json.loads(response)
            except json.JSONDecodeError:
                # Try to find JSON in the response
                import re
                json_match = re.search(r"\{[^{}]*\}", response)
                if json_match:
                    data = json.loads(json_match.group())
                else:
                    data = {"raw_response": response}
            
            return {"data": data}
            
        except Exception as e:
            logger.error(f"Extraction failed: {e}")
            return {"data": {}, "error": str(e)}

