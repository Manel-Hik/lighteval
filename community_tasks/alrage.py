from typing import List, Dict, Optional
from lighteval.metrics.llm_as_judge import JudgeLM
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc
from lighteval.metrics.metrics import Metrics

def qa_prompt_arabic(line, task_name: str = None):
    """Format the prompt for question answering with context"""
    question = line["question"] #question is the question from our alrage dataset
    context = line["chunks"]  # chunks is our chunked context from the alrage dataset
    
    instruction = "بناءً على السياق التالي، أجب عن السؤال بشكل دقيق ومختصر"
    
    query = f"""{instruction}

السياق:
{context}

السؤال:
{question}

الإجابة:"""

    return Doc(
        task_name=task_name,
        query=query,
        context=context,
        gold_answer=line["answer"], #answer is the answer from our alrage dataset
        instruction=instruction,
    )

def judge_template(question: str, answer: str, context: str, gold: str) -> list[dict]:
    """Enhanced template for the judge prompt in Arabic"""
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
            "content": f"""السياق: {context}

السؤال: {question}

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

# Initialize the judge
arabic_judge = JudgeLM(
    model="Qwen/Qwen2.5-72B-Instruct",
    templates=judge_template,
    process_judge_response=process_judge_response,
    judge_backend="transformers"
)

# Create task configuration
alrage_qa_task = LightevalTaskConfig(
    name="alrage_qa",
    prompt_function=qa_prompt_arabic,
    suite=["community"],
    hf_repo="the_alrage_hf_repo",#TO DO: I need to fix final format of the repo name  
    hf_subset="default",
    hf_avail_splits=["train"],
   
    metric=[arabic_judge],  # Using judge scoring
    trust_dataset=True,
    version=0,
)

TASKS_TABLE = [alrage_qa_task]

if __name__ == "__main__":
   
    question = "ما هو تأثير التلوث على البيئة البحرية؟"
    context = """التلوث البحري له آثار مدمرة على النظام البيئي البحري. يؤدي إلى موت الكائنات البحرية وتدمير الشعاب المرجانية. كما يؤثر على سلسلة الغذاء البحرية بأكملها."""
    answer = "التلوث يؤدي إلى موت الكائنات البحرية وتدمير الشعاب المرجانية"
    gold = "التلوث البحري يدمر النظام البيئي البحري من خلال قتل الكائنات البحرية وتدمير الشعاب المرجانية والتأثير على سلسلة الغذاء البحرية"
    
    score, prompt, response = arabic_judge.evaluate_answer(
        question=question,
        answer=answer,
        context=context,
        gold=gold
    )
    print(f"Score: {score}")
    print(f"Prompt: {prompt}")
    print(f"Response: {response}")