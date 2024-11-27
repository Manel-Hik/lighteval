from typing import List, Dict, Optional
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc
from lighteval.metrics.llm_as_judge import JudgeLM
from lighteval.metrics.metrics import MetricCategory  # Import MetricCategory

# Wrapper class for JudgeLM to include metric_name
class JudgeMetricWrapper:
    def __init__(self, judge: JudgeLM):
        self.judge = judge
        self.metric_name = "llm_as_judge"  # Define a metric name
        self.category = MetricCategory.LLM_AS_JUDGE  # Add the category attribute

    def evaluate(self, question: str, answer: str, options: list[str] = None, gold: str = None):
        return self.judge.evaluate_answer(question, answer, options, gold)

def qa_prompt_arabic(line: Dict, task_name: str = None) -> Doc:
    """Format the prompt for question answering with candidates"""
    question = line["question"]
    
    # Split the candidates string into a list
    candidates = line["candidates"].split('\n')  # Split by newline character
    
    # Remove any empty strings from the list
    candidates = [candidate for candidate in candidates if candidate.strip()]

    instruction = "بناءً على السياقات المقترحة التالية، اجب عن السؤال التالي"
    
    query = f"""{instruction}

السؤال:
{question}

السياقات المقترحة:
{', '.join(candidates)}  # Join candidates for better readability

الإجابة:"""

    # The gold answer is the correct answer
    gold_answer = line["gold_answer"]
    
    # Set choices to a list containing the gold answer
    choices = [gold_answer]  # Choices now represent the golden answer
    gold_index = 0  # The index of the gold answer in the choices list

    return Doc(
        task_name=task_name,
        query=query,
        instruction=instruction,
        choices=choices,  # Set choices to the golden answer
        gold_index=gold_index  # Index of the golden answer
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

def process_judge_response(response: str) -> float:
    """Process the judge's response to extract the score"""
    try:
        score = float(next(num for num in response.split() if num.replace('.','',1).isdigit()))
        return min(max(score / 10.0, 0.0), 1.0)
    except (StopIteration, ValueError):
        return 0.0

# Initialize the judge metric
judge = JudgeLM(
    model="Qwen/Qwen2.5-7B-Instruct",  
    templates=judge_template,
    process_judge_response=process_judge_response,
    judge_backend="transformers" 
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
    version=0
)

TASKS_TABLE = [alrage_qa_task]