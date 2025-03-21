import pandas as pd
import random as rd


def merge_task_to_solution(path_to_solutions: str, path_to_tasks: str, save_to: str) -> None:
    solution_df = pd.read_excel(path_to_solutions)
    task_df = pd.read_excel(path_to_tasks)

    merged_df = pd.merge(solution_df, task_df, left_on='task_id', right_on='id', how='left')
    merged_df.rename(columns={'id_x': 'id'}, inplace=True)

    merged_df.to_excel(save_to, index=False)


def format_few_shot_examples(examples_path: str, system_prompt: str, example_ratio=0.05, seed=42) -> list:
    data = pd.read_excel(examples_path)

    rd.seed(seed)

    examples = []
    test_solutions = data.shape[0]
    total = int(example_ratio * test_solutions)
    for i in rd.choices(list(range(test_solutions)), [1/test_solutions] * test_solutions, k=total):
        row = data.iloc[i]
        examples.append({'role': 'system', 'text': system_prompt.format(row["description"], row["author_solution"])})
        examples.append({'role': "user", 'text': "Ошибочное решение студента:" + row["student_solution"]})
        examples.append({'role': "assistant", 'text': row["author_comment"]})
    return examples


if __name__ == "__main__":
    merge_task_to_solution(path_to_solutions="./data/raw/train/solutions.xlsx",
                           path_to_tasks="./data/raw/train/tasks.xlsx",
                           save_to="./data/processed/train/solutions.xlsx")
                           
    merge_task_to_solution(path_to_solutions="./data/raw/test/solutions.xlsx",
                           path_to_tasks="./data/raw/test/tasks.xlsx",
                           save_to="./data/processed/test/solutions.xlsx")
