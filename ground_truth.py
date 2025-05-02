"""
Ground truth table for SOW RAG system evaluation.
This file contains the correct answers for example questions that can be used
to evaluate the model's responses.
"""

GROUND_TRUTH = {
    "What are the deliverables for this project?": {
        "correct_answer": """The deliverables for the project include:
1. Comprehensive documentation and training (Project Objectives - Software Development)
2. High-quality solution meeting all requirements (Project Objectives - Digital Transformation)
3. Completion of project kickoff phase
4. Completion of requirements analysis phase
5. Completion of design phase
6. Completion of development phase
7. Completion of testing phase (Digital Transformation)""",
        "source": "SOW_PRJ-8190_20250429.json",
        "key_points": [
            "documentation and training",
            "high-quality solution",
            "project kickoff",
            "requirements analysis",
            "design phase",
            "development phase",
            "testing phase"
        ]
    },
    
    "What is the project duration?": {
        "correct_answer": """The project duration can be calculated based on the timeline provided in the SOW documents. 
The project starts with the Requirements Analysis phase on 2025-05-24 and ends with the Project Closure phase on 2025-09-30. 
Therefore, the project duration is approximately 4 months and 6 days.""",
        "source": "SOW_PRJ-8190_20250429.json",
        "key_points": [
            "start date: 2025-05-24",
            "end date: 2025-09-30",
            "duration: 4 months and 6 days"
        ]
    },
    
    "What are the payment terms?": {
        "correct_answer": """The payment terms mentioned in the provided SOW documents are net 30 days from the invoice date. 
This information is consistent across multiple sections of the SOWs, indicating that payment is expected within 30 days of receiving the invoice.""",
        "source": "SOW_PRJ-8190_20250429.json",
        "key_points": [
            "net 30 days",
            "from invoice date",
            "consistent across SOWs"
        ]
    },
    
    "What are the key milestones?": {
        "correct_answer": """The key milestones for the project based on the provided SOW documents are as follows:

1. Project Kickoff: Completion of project kickoff phase on 2025-04-29
2. Requirements Analysis: Completion of requirements analysis phase on 2025-05-24
3. Design Phase: Completion of design phase phase on 2025-06-19
4. Development Phase: Completion of development phase phase on 2025-07-15
5. Testing Phase: Completion of testing phase phase on 2025-08-09
6. Deployment: Completion of deployment phase on 2025-09-04
7. Project Closure: Project closure on 2025-09-30""",
        "source": "SOW_PRJ-8190_20250429.json",
        "key_points": [
            "Project Kickoff: 2025-04-29",
            "Requirements Analysis: 2025-05-24",
            "Design Phase: 2025-06-19",
            "Development Phase: 2025-07-15",
            "Testing Phase: 2025-08-09",
            "Deployment: 2025-09-04",
            "Project Closure: 2025-09-30"
        ]
    },
    
    "What is included in the project scope?": {
        "correct_answer": """The project scope includes requirements gathering and analysis, system design and architecture, 
development and implementation, testing and quality assurance, deployment and training. It also involves ensuring 
effective communication and collaboration with stakeholders. Hardware procurement and third-party software licensing 
are explicitly mentioned as out of scope activities in the provided SOW documents.""",
        "source": "SOW_PRJ-8190_20250429.json",
        "key_points": [
            "requirements gathering and analysis",
            "system design and architecture",
            "development and implementation",
            "testing and quality assurance",
            "deployment and training",
            "effective communication and collaboration",
            "hardware procurement (out of scope)",
            "third-party software licensing (out of scope)"
        ]
    }
}

def get_ground_truth():
    """Get the ground truth table for evaluation."""
    return GROUND_TRUTH

def evaluate_response(question: str, response: str) -> dict:
    """
    Evaluate a model's response against the ground truth.
    
    Args:
        question (str): The question asked
        response (str): The model's response
        
    Returns:
        dict: Evaluation metrics including:
            - exact_match (bool): Whether the response exactly matches the ground truth
            - key_points_matched (int): Number of key points present in the response
            - total_key_points (int): Total number of key points in ground truth
            - key_points_coverage (float): Percentage of key points covered
    """
    if question not in GROUND_TRUTH:
        return {
            "error": "Question not found in ground truth"
        }
    
    ground_truth = GROUND_TRUTH[question]
    key_points = ground_truth["key_points"]
    
    # Count how many key points are present in the response
    matched_points = sum(1 for point in key_points if point.lower() in response.lower())
    
    return {
        "exact_match": response.strip() == ground_truth["correct_answer"].strip(),
        "key_points_matched": matched_points,
        "total_key_points": len(key_points),
        "key_points_coverage": matched_points / len(key_points) if key_points else 0
    } 