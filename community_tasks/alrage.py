import argparse
from typing import List, Dict, Optional
from lighteval.metrics.llm_as_judge import JudgeLM
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

def parse_args():
    parser = argparse.ArgumentParser(description='Arabic QA Evaluation with LLM Judge')
    
    # Model arguments
    parser.add_argument(
        '--model_name',
        type=str,
        default="mistralai/Mistral-7B-Instruct-v0.2",
        help='HuggingFace model name/path for answering questions'
    )
    parser.add_argument(
        '--tokenizer_name',
        type=str,
        default=None,  # Will use model_name if not specified
        help='HuggingFace tokenizer name/path (optional)'
    )
    
    # Generation parameters
    parser.add_argument('--max_new_tokens', type=int, default=512)
    parser.add_argument('--temperature', type=float, default=0.7)
    parser.add_argument('--top_p', type=float, default=0.9)
    parser.add_argument('--device_map', type=str, default="auto")
    
    return parser.parse_args()

def qa_prompt_arabic(line, task_name: str = None):
    """Format the prompt for question answering with context"""
    question = line["question"]
    context = line["chunks"]  # Assuming chunks are provided in the dataset
    
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
        gold_answer=line["answer"],
        instruction=instruction,
    )

def judge_template(question: str, answer: str, gold: str) -> list[dict]:
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

def initialize_models(args):
    """Initialize the answering model and judge"""
    
    tokenizer_name = args.tokenizer_name or args.model_name
    
    # Initialize the answering model
    answer_model = pipeline(
        "text-generation",
        model=args.model_name,
        tokenizer=tokenizer_name,
        device_map=args.device_map
    )
    
    # Initialize the judge
    arabic_judge = JudgeLM(
        model="Qwen/Qwen2.5-72B-Instruct",
        templates=judge_template,
        process_judge_response=process_judge_response,
        judge_backend="transformers"
    )
    
    return answer_model, arabic_judge

def get_model_answer(model, question: str, context: str, args) -> str:
    """Get answer from the model"""
    prompt = f"""بناءً على السياق التالي، أجب عن السؤال بشكل دقيق ومختصر

السياق:
{context}

السؤال:
{question}

الإجابة:"""
    
    response = model(
        prompt,
        max_new_tokens=args.max_new_tokens,
        do_sample=True,
        temperature=args.temperature,
        top_p=args.top_p,
    )[0]['generated_text']
    
    answer = response.split("الإجابة:")[-1].strip()
    return answer

if __name__ == "__main__":
    # Parse command-line arguments
    args = parse_args()
    
    # Initialize models
    answer_model, arabic_judge = initialize_models(args)
    
    # Create task configuration
    alrage_qa_task = LightevalTaskConfig(
        name="alrage_qa",
        prompt_function=qa_prompt_arabic,
        suite=["community"],
        hf_repo="your_repo/alrage",  # Replace with our actual repo
        hf_subset="default",
        hf_avail_splits=["train"],
        metric=[arabic_judge],
        trust_dataset=True,
        version=0,
        model=answer_model
    )
    
    TASKS_TABLE = [alrage_qa_task]
    
  
    question = "ما هو تأثير التلوث على البيئة البحرية؟"
    context = """التلوث البحري له آثار مدمرة على النظام البيئي البحري. يؤدي إلى موت الكائنات البحرية وتدمير الشعاب المرجانية. كما يؤثر على سلسلة الغذاء البحرية بأكملها."""
    gold = "التلوث البحري يدمر النظام البيئي البحري من خلال قتل الكائنات البحرية وتدمير الشعاب المرجانية والتأثير على سلسلة الغذاء البحرية"
    
    # Get answer from the model
    model_answer = get_model_answer(answer_model, question, context, args)
    print(f"Model's Answer: {model_answer}")
    
    # Evaluate the answer using the judge
    score, prompt, response = arabic_judge.evaluate_answer(
        question=f"السياق: {context}\n\nالسؤال: {question}",
        answer=model_answer,
        gold=gold
    )
    
    print(f"Score: {score}")
    print(f"Judge's Response: {response}")

#python alrage.py --model_name "mistralai/Mistral-7B-Instruct-v0.2"