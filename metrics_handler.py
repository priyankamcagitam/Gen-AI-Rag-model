import os
import json
from datetime import datetime
from typing import Dict, Any, List
import logging
import numpy as np

logger = logging.getLogger(__name__)

class MetricsHandler:
    def __init__(self):
        self.metrics = {
            "system_config": {},
            "queries": [],
            "performance_metrics": {
                "total_queries": 0,
                "average_response_time": 0,
                "average_query_time": 0,
                "average_latency": 0,
                "success_rate": 0,
                "min_response_time": float('inf'),
                "max_response_time": 0,
                "min_query_time": float('inf'),
                "max_query_time": 0,
                "min_latency": float('inf'),
                "max_latency": 0
            },
            "quality_metrics": {
                "average_faithfulness_score": 0,
                "total_faithful_answers": 0,
                "min_faithfulness_score": 1.0,
                "max_faithfulness_score": 0
            }
        }
        self._load_latest_metrics()
    
    def _load_latest_metrics(self):
        """Load the latest metrics file if it exists"""
        try:
            metrics_dir = "metrics"
            if os.path.exists(metrics_dir):
                metrics_files = [f for f in os.listdir(metrics_dir) if f.startswith("metrics_")]
                if metrics_files:
                    latest_file = max(metrics_files)
                    with open(os.path.join(metrics_dir, latest_file), 'r') as f:
                        self.metrics = json.load(f)
                        logger.info(f"Loaded metrics from {latest_file}")
        except Exception as e:
            logger.error(f"Error loading metrics: {str(e)}")
    
    def update_system_config(self, config: Dict[str, Any]):
        """Update system configuration metrics"""
        self.metrics["system_config"].update(config)
        logger.info(f"Updated system config: {config}")
    
    def add_query_metrics(self, question: str, answer: str, sources: List[Any], 
                         response_time: float, query_time: float, latency: float,
                         faithfulness_score: float):
        """Add metrics for a single query"""
        query_metrics = {
            "timestamp": datetime.now().isoformat(),
            "question": question,
            "answer": answer,
            "sources_used": len(sources),
            "response_time": float(response_time),
            "query_time": float(query_time),
            "latency": float(latency),
            "faithfulness_score": float(faithfulness_score),
            "is_faithful": bool(faithfulness_score >= 0.7)
        }
        
        self.metrics["queries"].append(query_metrics)
        self._update_performance_metrics(
            response_time=response_time,
            query_time=query_time,
            latency=latency
        )
        self._update_quality_metrics(faithfulness_score)
        
        logger.info(f"Added query metrics: {query_metrics}")
    
    def _update_performance_metrics(self, response_time: float, query_time: float,
                                  latency: float):
        """Update performance metrics"""
        total_queries = len(self.metrics["queries"])
        self.metrics["performance_metrics"]["total_queries"] = total_queries
        
        # Update averages and min/max values
        if total_queries > 0:
            perf_metrics = self.metrics["performance_metrics"]
            
            # Response time (total time)
            perf_metrics["average_response_time"] = float(
                sum(q["response_time"] for q in self.metrics["queries"]) / total_queries
            )
            perf_metrics["min_response_time"] = float(min(perf_metrics["min_response_time"], response_time))
            perf_metrics["max_response_time"] = float(max(perf_metrics["max_response_time"], response_time))
            
            # Query time (time spent in answer generation)
            perf_metrics["average_query_time"] = float(
                sum(q["query_time"] for q in self.metrics["queries"]) / total_queries
            )
            perf_metrics["min_query_time"] = float(min(perf_metrics["min_query_time"], query_time))
            perf_metrics["max_query_time"] = float(max(perf_metrics["max_query_time"], query_time))
            
            # Latency
            perf_metrics["average_latency"] = float(
                sum(q["latency"] for q in self.metrics["queries"]) / total_queries
            )
            perf_metrics["min_latency"] = float(min(perf_metrics["min_latency"], latency))
            perf_metrics["max_latency"] = float(max(perf_metrics["max_latency"], latency))
            
            # Success rate
            perf_metrics["success_rate"] = float(
                sum(1 for q in self.metrics["queries"] if q["is_faithful"]) / total_queries
            )
            
            logger.info(f"Updated performance metrics: {perf_metrics}")
            
    def _update_quality_metrics(self, faithfulness_score: float):
        """Update quality metrics"""
        total_queries = len(self.metrics["queries"])
        if total_queries > 0:
            quality_metrics = self.metrics["quality_metrics"]
            
            quality_metrics["average_faithfulness_score"] = float(
                sum(q["faithfulness_score"] for q in self.metrics["queries"]) / total_queries
            )
            quality_metrics["total_faithful_answers"] = int(
                sum(1 for q in self.metrics["queries"] if q["is_faithful"])
            )
            quality_metrics["min_faithfulness_score"] = float(min(
                quality_metrics["min_faithfulness_score"], faithfulness_score
            ))
            quality_metrics["max_faithfulness_score"] = float(max(
                quality_metrics["max_faithfulness_score"], faithfulness_score
            ))
            
            logger.info(f"Updated quality metrics: {quality_metrics}")
        
    def save_metrics(self) -> str:
        """Save metrics to a JSON file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"metrics_{timestamp}.json"
        
        # Create metrics directory if it doesn't exist
        os.makedirs("metrics", exist_ok=True)
        filepath = os.path.join("metrics", filename)
        
        # Convert all values to native Python types
        metrics_copy = self._convert_to_native_types(self.metrics)
        
        with open(filepath, "w") as f:
            json.dump(metrics_copy, f, indent=2)
            
        logger.info(f"Saved metrics to {filepath}")
        return filepath
    
    def _convert_to_native_types(self, obj):
        """Convert NumPy types to native Python types"""
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: self._convert_to_native_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_to_native_types(item) for item in obj]
        return obj 