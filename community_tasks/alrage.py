from typing import List, Dict, Optional
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc
from lighteval.metrics.llm_as_judge import JudgeLM
from lighteval.metrics.metrics import MetricCategory  # Import MetricCategory

class JudgeMetricWrapper:
    def __init__(self, judge: JudgeLM):
        self.judge = judge
        self.metric_name = "llm_as_judge"  # Define a metric name
        self.category = MetricCategory.LLM_AS_JUDGE  # Add the category attribute

    def compute(self, responses: list[str], formatted_docs: list[Doc], **kwargs) -> dict[str, float]:
        """
        Compute the score using the judge's evaluate_answer method.
        
        Args:
            predictions (list[str]): The predicted answers.
            formatted_docs (list[Doc]): The formatted documents containing questions and gold answers.
        
        Returns:
            dict[str, float]: A dictionary containing the evaluation scores.
        """
        scores = []
        print("Starting computation of scores...")  # Debugging print

        for i, doc in enumerate(formatted_docs):
            question = doc.query
            gold = doc.choices[doc.gold_index[0]] if doc.gold_index else None
            answer = responses[i]

            print(f"Processing document {i}:")  # Debugging print
            print(f"  Question: {question}")  # Debugging print
            print(f"  Gold Answer: {gold}")  # Debugging print
            print(f"  Predicted Answer: {answer}")  # Debugging print

            # Directly use judge.evaluate_answer here
            score, _, _ = self.judge.evaluate_answer(question, answer, options=None, gold=gold)
            scores.append(score)

            print(f"  Score for document {i}: {score}")  # Debugging print

        # Return a dictionary with a key that can be accessed
        result = {"scores": scores}
        print("Computed scores:", result)  # Debugging print
        return result

def qa_prompt_arabic(line: Dict, task_name: str = None) -> Doc:
    """Format the prompt for question answering with candidates"""
    # Ensure all inputs are strings
    question = str(line["question"])
    
    # Convert candidates to string if it isn't already
    if isinstance(line["candidates"], list):
        candidates = [str(c) for c in line["candidates"]]
    else:
        candidates = str(line["candidates"]).split('\n')
    
    # Clean up candidates
    candidates = [c.strip() for c in candidates if c.strip()]

    instruction = "بناءً على السياقات المقترحة التالية، اجب عن السؤال التالي"
    
    # Format the query with proper string handling
    query = f"{instruction}\n\nالسؤال:\n{question}\n\nالسياقات المقترحة:\n{', '.join(candidates)}\n"
    
    # Ensure gold_answer is a string
    gold_answer = str(line.get("gold_answer", ""))
    
    # Create Doc with proper string types
    return Doc(
        task_name=task_name or "alrage",
        query=query,
        instruction=instruction,
        choices=[gold_answer],
        gold_index=0
    )
    
def judge_template(question: str, answer: str, gold: str, options: Optional[List[str]] = None) -> List[Dict[str, str]]:
    """Template for the judge prompt in Arabic"""
    messages = [
        {
            "role": "system", 
            "content": """أنت مقيّم محايد خبير. مهمتك هي:
1. تقييم دقة الإجابة مقارنة بالإجابة الصحيحة
2. التحقق من أن الإجابة مدعومة بالسياق المقدم
3. تقييم جودة وشمولية الإجابة

قم بتقييم الإجابة على مقياس من 0 إلى 10."""
        },
        {
            "role": "user", 
            "content": f"""{question}

الإجابة المقدمة: {answer}

الإجابة الصحيحة: {gold}

قيّم الإجابة على مقياس من 0 إلى 10، حيث:
- 0-2: إجابة خاطئة تماماً أو غير متعلقة
- 3-4: إجابة جزئية مع أخطاء كبيرة
- 5-6: إجابة متوسطة الدقة
- 7-8: إجابة جيدة مع بعض النقص
- 9-10: إجابة ممتازة ودقيقة

قدم تقييمك كرقم فقط."""
        }
    ]
    return messages

def process_judge_response(response) -> float:
    """Process the judge's response to extract the score"""
    # If response is a list, extract the content from the user role
    if isinstance(response, list):
        # Join the content from the user role into a single string
        response_content = ' '.join(item['content'] for item in response if item['role'] == 'user')
    else:
        response_content = response  # If it's not a list, use it directly

    try:
        # Extract the score from the response content
        score = float(next(num for num in response_content.split() if num.replace('.', '', 1).isdigit()))
        return min(max(score / 10.0, 0.0), 1.0)
    except (StopIteration, ValueError):
        return 0.0

# Initialize the judge metric
judge = JudgeLM(
    model="Qwen/Qwen2.5-7B-Instruct",  
    templates=judge_template,
    process_judge_response=process_judge_response,
    judge_backend="vllm" 
)

# Wrap the judge in the new wrapper class
wrapped_judge = JudgeMetricWrapper(judge)

# Create task configuration
alrage_qa_task = LightevalTaskConfig(
    name="alrage_qa",
    prompt_function=qa_prompt_arabic,
    suite=["community"],
    hf_repo="OALL/ALRAGE",
    hf_subset=None,
    hf_avail_splits=["train"],  # Only the train split is available
    evaluation_splits=["train"],  
    metric=[wrapped_judge],  
    trust_dataset=True,
    generation_size=-1,  ## updated
    stop_sequence=[],  ## updated
    version=0
)

TASKS_TABLE = [alrage_qa_task]
