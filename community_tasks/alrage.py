import argparse
from typing import List, Dict, Optional
from lighteval.metrics.llm_as_judge import JudgeLM
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from datasets import load_dataset

def parse_args():
    parser = argparse.ArgumentParser(description='Arabic RAG Evaluation with LLM as a Judge')
    
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
    parser.add_argument('--test_mode', type=str, choices=['example', 'dataset'], default='example',
                       help='Whether to test on example or dataset')
    
    return parser.parse_args()

def qa_prompt_arabic(line, task_name: str = None):
    """Format the prompt for question answering with candidates"""
    question = line["question"]
    candidates = line["candidates"]
    
    instruction = "بناءً على الإجابات المقترحة التالية، اختر الإجابة الأفضل والأنسب للسؤال"
    
    query = f"""{instruction}

السؤال:
{question}

الإجابات المقترحة:
{candidates}

الإجابة:"""

    return Doc(
        task_name=task_name,
        query=query,
        context=candidates,
        gold_answer=line["gold_answer"],
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
    """Initialize the generator model and judge"""
    
    tokenizer_name = args.tokenizer_name or args.model_name
    
    # Initialize the text generation model
    generator = pipeline(
        "text-generation",
        model=args.model_name,
        tokenizer=tokenizer_name,
        device_map=args.device_map
    )
    
    # Initialize the judge
    judge = JudgeLM(
        model="Qwen/Qwen2.5-72B-Instruct",
        templates=judge_template,
        process_judge_response=process_judge_response,
        judge_backend="transformers"
    )
    
    return generator, judge

def generate_answer(generator, question: str, context: str, args) -> str:
    """Generate answer using the text generation model
    Args:
        generator: The pipeline model for text generation
        question: The question text
        context: The context/candidates text
        args: Command line arguments
    """
    prompt = f"""بناءً على السياق التالي، أجب عن السؤال بشكل دقيق ومختصر

السياق:
{context}

السؤال:
{question}

الإجابة:"""
    
    response = generator(
        prompt,
        max_new_tokens=args.max_new_tokens,
        do_sample=True,
        temperature=args.temperature,
        top_p=args.top_p,
    )[0]['generated_text']
    
    generated_answer = response.split("الإجابة:")[-1].strip()
    return generated_answer

if __name__ == "__main__":
    # Parse command-line arguments
    args = parse_args()
    
    # Initialize models
    generator, judge = initialize_models(args)
    
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
        version=0,
        model=generator
    )
    
    TASKS_TABLE = [alrage_qa_task]
    
    if args.test_mode == 'example':
        # Test on separate example
        question = "ما هو تأثير التلوث على البيئة البحرية؟"
        context = """التلوث البحري له آثار مدمرة على النظام البيئي البحري. يؤدي إلى موت الكائنات البحرية وتدمير الشعاب المرجانية. كما يؤثر على سلسلة الغذاء البحرية بأكملها."""
        gold_answer = "التلوث البحري يدمر النظام البيئي البحري من خلال قتل الكائنات البحرية وتدمير الشعاب المرجانية والتأثير على سلسلة الغذاء البحرية"
        
        print("\n=== Testing on Example ===")
        # Generate answer using the model
        generated_answer = generate_answer(
            generator=generator,
            question=question,
            context=context,
            args=args
        )
        print(f"Question: {question}")
        print(f"Context: {context}")
        print(f"Generated Answer: {generated_answer}")
        print(f"Gold Answer: {gold_answer}")
        
        # Evaluate the answer using the judge
        score, prompt, response = judge.evaluate_answer(
            question=f"السياق: {context}\n\nالسؤال: {question}",
            answer=generated_answer,
            gold=gold_answer
        )
        
        print(f"Score: {score}")
        print(f"Judge's Response: {response}")
    
    else:
        # Test on HF dataset
        print("\n=== Testing on OALL/ALRAGE Dataset ===")
        dataset = load_dataset("OALL/ALRAGE")
        
        # Test on first example from dataset
        sample = dataset["train"][0]
        
        print(f"Question: {sample['question']}")
        print(f"Context: {sample['candidates']}")
        
        # Generate answer using the model
        generated_answer = generate_answer(
            generator=generator,
            question=sample["question"],
            context=sample["candidates"],
            args=args
        )
        print(f"Generated Answer: {generated_answer}")
        print(f"Gold Answer: {sample['gold_answer']}")
        
        # Evaluate the answer using the judge
        score, prompt, response = judge.evaluate_answer(
            question=f"السؤال: {sample['question']}\n\nالإجابات المقترحة: {sample['candidates']}",
            answer=generated_answer,
            gold=sample["gold_answer"]
        )
        
        print(f"Score: {score}")
        print(f"Judge's Response: {response}")

        
# Test on the example case
#python community_tasks/alrage.py --test_mode example --model_name "mistralai/Mistral-7B-Instruct-v0.2"

# Test on the OALL/ALRAGE dataset
#python community_tasks/alrage.py --test_mode dataset --model_name "mistralai/Mistral-7B-Instruct-v0.2"