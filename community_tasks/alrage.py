from typing import List, Dict, Optional
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc
from lighteval.metrics.llm_as_judge import JudgeLM
from lighteval.metrics.metrics import MetricCategory, Metric  # Import MetricCategory and Metric

class JudgeMetricWrapper(Metric):  # Extend from Metric
    def __init__(self, judge: JudgeLM):
        self.judge = judge
        self.metric_name = "llm_as_judge"  # Define a metric name
        self.category = MetricCategory.LLM_AS_JUDGE  # Add the category attribute
        self.corpus_level_fn = self.aggregate_scores  # Define the corpus level function

    def compute(self, responses: list[str], formatted_docs: list[Doc], **kwargs) -> dict[str, float]:
        """
        Compute the score using the judge's evaluate_answer method.
        
        Args:
            predictions (list[str]): The predicted answers.
            formatted_docs (list[Doc]): The formatted documents containing questions and gold answers.
        
        Returns:
            dict[str, float]: A dictionary containing the evaluation scores.
        """
        results = []
        for i, doc in enumerate(formatted_docs):
            question = doc.query
            gold = doc.choices[doc.gold_index] if doc.gold_index is not None else None
            answer = responses[i][0].result[0]


            # Directly use judge.evaluate_answer here
            score, _, _ = self.judge.evaluate_answer(question, answer, options=None, gold=gold)

            results.append({"score": score})
        # Return a dictionary with a key that can be accessed
        return results

    def aggregate_scores(self, scores: list[dict]) -> float:
        """Aggregate scores from the compute method."""
        return sum(score["score"] for score in scores) / len(scores) if scores else 0.0


def qa_prompt_arabic(line: Dict, task_name: str = None) -> Doc:
    """Format the prompt for question answering with candidates"""
    
    # Check the input line structure

    question = str(line["question"])
    
    # Convert candidates to string if it isn't already
    if isinstance(line["candidates"], list):
        candidates = [str(c) for c in line["candidates"]]
    else:
        candidates = str(line["candidates"]).split('\n')
    
    # Clean up candidates
    candidates = [c.strip() for c in candidates if c.strip()]

    instruction = "بناءً على السياقات المقترحة التالية، اجب عن السؤال التالي"
    query = f"{instruction}\n\nالسؤال:\n{question}\n\nالسياقات المقترحة:\n{', '.join(candidates)}\n"
    
    # Ensure gold_answer is a string
    gold_answer = str(line.get("gold_answer", ""))  # Ensure this is set correctly

    # Create Doc with proper string types
    doc = Doc(
        task_name=task_name or "alrage",
        query=query,
        instruction=instruction,
        choices=[gold_answer],  # Ensure this is populated correctly
        gold_index= 0
    )

    return doc
    
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
    generation_size=200,  ## updated
    stop_sequence=[],  ## updated
    version=0
)

TASKS_TABLE = [alrage_qa_task]