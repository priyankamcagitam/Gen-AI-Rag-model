"""
Script to evaluate RAG system responses against ground truth.
This script shows how to use ground_truth.py to evaluate the quality of responses.
"""

from ground_truth import get_ground_truth, evaluate_response
from rag_system import SOWRAGSystem
import json
from datetime import datetime

def evaluate_rag_system():
    # Initialize the RAG system
    print("Initializing RAG system...")
    rag_system = SOWRAGSystem()
    rag_system.initialize()
    
    # Get ground truth questions and answers
    ground_truth = get_ground_truth()
    
    # Store evaluation results
    evaluation_results = {
        "timestamp": datetime.now().isoformat(),
        "questions": []
    }
    
    # Evaluate each question
    for question in ground_truth.keys():
        print(f"\nEvaluating question: {question}")
        
        # Get RAG system's response
        response = rag_system.query(question)
        
        # Evaluate the response against ground truth
        evaluation = evaluate_response(question, response["answer"])
        
        # Add detailed results
        question_result = {
            "question": question,
            "rag_response": response["answer"],
            "ground_truth": ground_truth[question]["correct_answer"],
            "evaluation": evaluation,
            "metrics": {
                "response_time": response["response_time"],
                "query_time": response["query_time"],
                "faithfulness_score": response["faithfulness_score"]
            }
        }
        
        evaluation_results["questions"].append(question_result)
        
        # Print results
        print("\nEvaluation Results:")
        print(f"Exact Match: {evaluation['exact_match']}")
        print(f"Key Points Coverage: {evaluation['key_points_coverage']:.2%}")
        print(f"Key Points Matched: {evaluation['key_points_matched']}/{evaluation['total_key_points']}")
        print(f"Response Time: {response['response_time']:.2f} seconds")
        print(f"Faithfulness Score: {response['faithfulness_score']:.2f}")
    
    # Calculate overall metrics
    total_coverage = sum(q["evaluation"]["key_points_coverage"] for q in evaluation_results["questions"])
    avg_coverage = total_coverage / len(evaluation_results["questions"])
    
    total_faithfulness = sum(q["metrics"]["faithfulness_score"] for q in evaluation_results["questions"])
    avg_faithfulness = total_faithfulness / len(evaluation_results["questions"])
    
    evaluation_results["overall_metrics"] = {
        "average_key_points_coverage": avg_coverage,
        "average_faithfulness_score": avg_faithfulness,
        "total_questions": len(evaluation_results["questions"])
    }
    
    # Save results to file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"evaluation_results_{timestamp}.json"
    
    with open(filename, 'w') as f:
        json.dump(evaluation_results, f, indent=2)
    
    print(f"\nEvaluation complete! Results saved to {filename}")
    print("\nOverall Metrics:")
    print(f"Average Key Points Coverage: {avg_coverage:.2%}")
    print(f"Average Faithfulness Score: {avg_faithfulness:.2f}")
    
    return evaluation_results

if __name__ == "__main__":
    evaluate_rag_system() 