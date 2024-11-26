from typing import List, Dict, Optional
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc
from lighteval.metrics.llm_as_judge import JudgeLM
from lighteval.metrics.metrics import MetricCategory  # Import MetricCategory

# Subclassing JudgeLM to add category attribute
class JudgeWithCategory(JudgeLM):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.category = MetricCategory.LLM_AS_JUDGE  # Add the category attribute

def qa_prompt_arabic(line: Dict, task_name: str = None) -> Doc:
    """Format the prompt for question answering with candidates"""
    question = line["question"]
    candidates = line["candidates"]
    
    instruction = "بناءً على السياقات المقترحة التالية، اجب عن السؤال التالي"
    
    query = f"""{instruction}

السؤال:
{question}

السياقات المقترحة:
{candidates}

الإجابة:"""

    return Doc(
        task_name=task_name,
        query=query,
        context=candidates,
        gold_answer=line["gold_answer"],
        instruction=instruction,
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

# Initialize the judge metric using the subclass
judge = JudgeWithCategory(
    model="Qwen/Qwen2.5-7B-Instruct",  
    templates=judge_template,
    process_judge_response=process_judge_response,
    judge_backend="transformers" 
)

# Create task configuration
alrage_qa_task = LightevalTaskConfig(
    name="alrage_qa",
    prompt_function=qa_prompt_arabic,
    suite=["community"],
    hf_repo="OALL/ALRAGE",
    hf_subset=None,
    hf_avail_splits=["train"],
    metric=[judge],
    trust_dataset=True,
    version=0
)

TASKS_TABLE = [alrage_qa_task]