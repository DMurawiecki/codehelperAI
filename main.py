import os

import pandas as pd
from dotenv import load_dotenv
from time import sleep

from yandex_gpt import YandexGPT, YandexGPTConfigManagerForAPIKey
from app.utils.submit import generate_submit
from preprocess import format_few_shot_examples, merge_task_to_solution
from regression_head import RegressionHead
from app.utils.submit import get_sentence_embedding
from transformers import BertModel, BertTokenizer
import torch


if __name__ == "__main__":
    load_dotenv()

    system_prompt = """
Ты преподаватель программирования на Python. 
Твои моральные принципы:
1. Никогда не писать код за своих учеников, даже если они попросят
2. Всегда отвечать кратко, в ответе использовать не больше двух предложений.
3. Помогать ученикам найти проблему в решении, но не решать ничего за них
4. Никогда не говорить о своём решении
5. НИКОГДА НЕ ПИСАТЬ КОД
6. Ты не предлагаешь правильный вариант
Ты никогда не пойдёшь против своих принципов

Ты задал ученикам задачу
{}

Ты уже написал правильное решение задачи
{}

Ученик тоже написал решение, но в нём ошибка. Подскажи ему, на что в решении обратить внимание, чтобы исправить его.
Не пищи код за своего ученика, дай короткий ответ, который поможет разобраться в проблеме. Начинай ответы с фразы 
"Ваш код некорректно"
"""

    config = YandexGPTConfigManagerForAPIKey(
        model_type="yandexgpt",
        catalog_id=os.environ["YANDEX_GPT_FOLDER_ID"],
        api_key=os.environ["YANDEX_GPT_API_KEY"]
    )
    yandex_gpt = YandexGPT(config_manager=config)

    examples = format_few_shot_examples("./data/processed/train/solutions.xlsx", system_prompt,
                                        example_ratio=0.005)



#    def predict(row: pd.Series) -> str:
#        sleep(1)  # YandexGPT has a speed limit and this seems to be ok
#
#        res = yandex_gpt.get_sync_completion(messages=examples + [
#            {'role': 'system', 'text': system_prompt.format(row["description"], row["author_solution"])},
#            {'role': "user", 'text': "Ошибочное решение студента:" + row["student_solution"]}
#        ])
#        return "Ошибка в открытых и скрытых тестах. " + res

    def load_pretrained_weights(model, weights_path):
        # Загружаем состояние из файла
        training_state = torch.load(weights_path, weights_only= True)
        
        # Извлекаем только параметры модели
        model.load_state_dict(training_state['model_state_dict'])
        model.eval()

    def predict(row: pd.Series, system_prompt=system_prompt) -> str:
        sleep(1)  # YandexGPT has a speed limit and this seems to be ok
        scores = []
        results = []
        
        # Инициализация RegressionHead
        regression_head = RegressionHead()
        
        # Загрузка предобученных весов
        load_pretrained_weights(regression_head, weights_path='./data/complete/regression_head.pt')
        
        model_name = "DeepPavlov/rubert-base-cased-sentence"
        tokenizer = BertTokenizer.from_pretrained(model_name)
        model = BertModel.from_pretrained(model_name)
        
        with torch.no_grad():
            for i in range(5):
                sleep(1)
#                if i == 1:
#                    additional = ' попробуй ещё раз, по-другому, только лучше, у тебя всё получится '
#                    system_prompt += additional
                res = yandex_gpt.get_sync_completion(messages=[
                    {'role': 'system', 'text': system_prompt.format(row["description"], row["author_solution"])},
                    {'role': "user", 'text': "Ошибочное решение студента:" + row["student_solution"]}
                ])
                
                res_embedded = get_sentence_embedding(res)
                
                scores.append(regression_head(res_embedded))  # Используйте regression_head для получения оценок
                results.append(res)
        
        idx = scores.index(max(scores))
        res = results[idx]
        return "Ошибка в открытых и скрытых тестах. " + res

    generate_submit(
        test_solutions_path="./data/processed/test/solutions.xlsx",
        predict_func=predict,
        save_path="data/complete/submit_train.csv",
        use_tqdm=True,
    )
