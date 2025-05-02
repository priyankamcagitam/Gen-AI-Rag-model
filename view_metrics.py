import os
import json
from datetime import datetime

def view_latest_metrics():
    """View the latest metrics from both metrics and output directories"""
    
    # Check metrics directory
    metrics_dir = "metrics"
    if os.path.exists(metrics_dir):
        print("\nMetrics Directory Contents:")
        print("-" * 50)
        metrics_files = [f for f in os.listdir(metrics_dir) if f.endswith('.json')]
        if metrics_files:
            # Get the latest metrics file
            latest_metrics = max(metrics_files, key=lambda x: os.path.getctime(os.path.join(metrics_dir, x)))
            metrics_path = os.path.join(metrics_dir, latest_metrics)
            
            print(f"\nLatest Metrics File: {latest_metrics}")
            print(f"Created: {datetime.fromtimestamp(os.path.getctime(metrics_path))}")
            
            # Read and display metrics
            with open(metrics_path, 'r') as f:
                metrics = json.load(f)
                
            print("\nSystem Configuration:")
            print(f"Document Count: {metrics['system_config'].get('document_count', 'N/A')}")
            print(f"Total Chunks: {metrics['system_config'].get('chunk_config', {}).get('total_chunks', 'N/A')}")
            
            print("\nPerformance Metrics:")
            perf_metrics = metrics.get('performance_metrics', {})
            print(f"Total Queries: {perf_metrics.get('total_queries', 0)}")
            print(f"Average Response Time: {perf_metrics.get('average_response_time', 0):.2f} seconds")
            print(f"Average Latency: {perf_metrics.get('average_latency', 0):.2f} seconds")
            
            print("\nQuality Metrics:")
            quality_metrics = metrics.get('quality_metrics', {})
            print(f"Average Faithfulness Score: {quality_metrics.get('average_faithfulness_score', 0):.2f}")
            print(f"Total Faithful Answers: {quality_metrics.get('total_faithful_answers', 0)}")
            
            print("\nRecent Queries:")
            queries = metrics.get('queries', [])
            for query in queries[-5:]:  # Show last 5 queries
                print(f"\nQuestion: {query.get('question', 'N/A')}")
                print(f"Answer: {query.get('answer', 'N/A')[:100]}...")  # Show first 100 chars of answer
                print(f"Sources Used: {query.get('sources_used', 0)}")
                if 'faithfulness_score' in query:
                    print(f"Faithfulness Score: {query['faithfulness_score']:.2f}")
        else:
            print("No metrics files found")
    else:
        print("Metrics directory not found")
    
    # Check output directory
    output_dir = "output"
    if os.path.exists(output_dir):
        print("\nOutput Directory Contents:")
        print("-" * 50)
        output_files = [f for f in os.listdir(output_dir) if f.endswith('.json')]
        if output_files:
            # Get the latest output file
            latest_output = max(output_files, key=lambda x: os.path.getctime(os.path.join(output_dir, x)))
            output_path = os.path.join(output_dir, latest_output)
            
            print(f"\nLatest Output File: {latest_output}")
            print(f"Created: {datetime.fromtimestamp(os.path.getctime(output_path))}")
            
            # Read and display output
            with open(output_path, 'r') as f:
                output = json.load(f)
                
            print("\nOverall Metrics:")
            overall = output.get('overall_metrics', {})
            for metric, value in overall.items():
                if not metric.endswith('_std'):
                    std = overall.get(f"{metric}_std", 0)
                    print(f"{metric}: {value:.2f} (±{std:.2f})")
            
            print("\nTest Cases:")
            test_cases = output.get('test_cases', [])
            for test_case in test_cases:
                print(f"\nQuestion: {test_case.get('question', 'N/A')}")
                print(f"Answer: {test_case.get('actual', 'N/A')[:100]}...")  # Show first 100 chars of answer
                print(f"Sources Used: {test_case.get('sources_used', 0)}")
                if 'faithfulness_score' in test_case:
                    print(f"Faithfulness Score: {test_case['faithfulness_score']:.2f}")
        else:
            print("No output files found")
    else:
        print("Output directory not found")

if __name__ == "__main__":
    view_latest_metrics() 