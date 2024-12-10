from kfp import dsl
from kfp import compiler
from datasets import load_dataset
from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
    SentenceTransformerModelCardData,
)
from sentence_transformers.losses import MultipleNegativesRankingLoss
from sentence_transformers.training_args import BatchSamplers
from sentence_transformers.evaluation import TripletEvaluator

#
# Install dependencies at runtime
@dsl.component(packages_to_install=[
    'torch>=1.11.0',
    'accelerate==1.1.1',
    'transformers>=4.34.0',
    'sentence-transformers==3.3.1',
    'datasets==3.1.0',
    'kfp==2.10.1'])
def fine_tune_model(
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    dataset_name: str = "sentence-transformers/all-nli",
    subset: str = "triplet",
    output_dir: str = "models/mpnet-base-all-nli-triplet",
    data_range: int = 10_000,
    num_epochs: int = 1,
    train_batch_size: int = 16,
    eval_batch_size: int = 16,
):
    import subprocess
    import sys

    # Check installed packages
    subprocess.run([sys.executable, "-m", "pip", "freeze"], check=True)

    # Import the library
    from sentence_transformers import SentenceTransformer
    from datasets import load_dataset
    from sentence_transformers import (
        SentenceTransformer,
        SentenceTransformerTrainer,
        SentenceTransformerTrainingArguments,
        SentenceTransformerModelCardData,
    )
    from sentence_transformers.losses import MultipleNegativesRankingLoss
    from sentence_transformers.training_args import BatchSamplers
    from sentence_transformers.evaluation import TripletEvaluator
    print("SentenceTransformer module imported successfully!")
    from datasets import load_dataset
    print("datasets module imported successfully!")

    # 1. Load a model to finetune with 2. (Optional) model card data
    # model = SentenceTransformer(
    #     "microsoft/mpnet-base",
    #     model_card_data=SentenceTransformerModelCardData(
    #         language="en",
    #         license="apache-2.0",
    #         model_name="MPNet base trained on AllNLI triplets",
    #     )
    # )

    model = SentenceTransformer(model_name)


    # 3. Load a dataset to finetune on
    dataset = load_dataset(dataset_name, subset)
    train_dataset = dataset["train"].select(range(data_range))
    eval_dataset = dataset["dev"]
    test_dataset = dataset["test"]

    # 4. Define a loss function
    loss = MultipleNegativesRankingLoss(model)

    # 5. (Optional) Specify training arguments
    args = SentenceTransformerTrainingArguments(
        # Required parameter:
        output_dir=output_dir,
        # Optional training parameters:
        num_train_epochs=num_epochs,
        per_device_train_batch_size=train_batch_size,
        per_device_eval_batch_size=eval_batch_size,
        warmup_ratio=0.1,
        fp16=True,  # Set to False if GPU can't handle FP16
        bf16=False,  # Set to True if GPU supports BF16
        batch_sampler=BatchSamplers.NO_DUPLICATES,  # MultipleNegativesRankingLoss benefits from no duplicates
        # Optional tracking/debugging parameters:
        eval_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=2,
        logging_steps=100,
        run_name="mpnet-base-all-nli-triplet",  # Used in W&B if `wandb` is installed
    )

    # 6. (Optional) Create an evaluator & evaluate the base model
    dev_evaluator = TripletEvaluator(
        anchors=eval_dataset["anchor"],
        positives=eval_dataset["positive"],
        negatives=eval_dataset["negative"],
        name="all-nli-dev",
    )
    dev_evaluator(model)

    # 7. Create a trainer & train
    trainer = SentenceTransformerTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        loss=loss,
        evaluator=dev_evaluator,
    )
    trainer.train()

    # (Optional) Evaluate the trained model on the test set, after training completes
    test_evaluator = TripletEvaluator(
        anchors=test_dataset["anchor"],
        positives=test_dataset["positive"],
        negatives=test_dataset["negative"],
        name="all-nli-test",
    )
    test_evaluator(model)

    # 8. Save the trained model
    model.save_pretrained("models/mpnet-base-all-nli-triplet/final")

    # 9. (Optional) Push it to the Hugging Face Hub
    # model.push_to_hub("mpnet-base-all-nli-triplet")

@dsl.component
def say_hello(name: str) -> str:
    hello_text = f'Hello, {name}!'
    print(hello_text)
    return hello_text

@dsl.pipeline
def launch_pipeline(recipient: str) -> str:
    hello_task = say_hello(name=recipient)
    fine_tune_model()
    return hello_task.output

# Compile the pipeline to a JSON file
compiler.Compiler().compile(
    pipeline_func=launch_pipeline, 
    package_path="hello_pipeline.json"
)
print("Pipeline compiled successfully.")
