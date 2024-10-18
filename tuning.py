import os

import pandas as pd
from dotenv import load_dotenv
from time import sleep
from tqdm import tqdm
from yandex_gpt import YandexGPT, YandexGPTConfigManagerForAPIKey
from app.utils.submit import generate_submit, string2embedding, get_sentence_embedding
from preprocess import format_few_shot_examples, merge_task_to_solution
from regression_head import RegressionHead
from app.utils.submit import get_sentence_embedding
from transformers import BertModel, BertTokenizer
from torch.utils.data import Dataset, DataLoader
#from sklearn import cosine_similarity
import torch.nn.functional as F
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

#    examples = format_few_shot_examples("./data/processed/train/solutions.xlsx", system_prompt,
#                                        example_ratio=0.005)
#                                        
    model_name = "DeepPavlov/rubert-base-cased-sentence"
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertModel.from_pretrained(model_name)
    
    def save_model(model, path):
        training_state = {
                        "model_state_dict": model.state_dict(),
                        # "optimizer_state_dict": optimizer.state_dict(),
                        # "train_step": i,
                    }
        torch.save(training_state, path)
        print('Model was saved')


    def predict(row: pd.Series) -> str:
        sleep(1)  # YandexGPT has a speed limit and this seems to be ok
        #TODO examples
        res = yandex_gpt.get_sync_completion(messages= [
            {'role': 'system', 'text': system_prompt.format(row["description"], row["author_solution"])},
            {'role': "user", 'text': "Ошибочное решение студента:" + row["student_solution"]}
        ])
        return "Ошибка в открытых и скрытых тестах. " + res

    def train_model(model, dataloader, optimizer, num_epochs=100):
        model.train()
        for epoch in tqdm(range(num_epochs), desc='Training'):
            for batch in dataloader:
                optimizer.zero_grad()
                inputs, target = batch
                inputs1 = [str(i) for i in inputs]
               
                inputs_embedded = [get_sentence_embedding(str(i)) for i in inputs1]
                
                # Преобразуем список векторов в тензор
                inputs_embedded_tensor = torch.stack([embedding.clone().detach().requires_grad_(True) for embedding in inputs_embedded]).float()
                target_embedded = [t for t in target]
                target_embedded_tensor = torch.stack([embedding.clone().detach() for embedding in target_embedded]).float()

                
                # Вычисляем косинусную близость
#                cs_true = cosine_similarity(inputs_embedded_tensor.detach().numpy(), target_embedded_tensor.detach().numpy())
                cs_true = F.cosine_similarity(inputs_embedded_tensor, target_embedded_tensor)
                cs_true_tensor = cs_true.clone().detach().requires_grad_(True).float()
                
                # Получаем выходы модели
                outputs = model(inputs_embedded_tensor)
                
                # Вычисляем лосс
#                loss = F.mse_loss(outputs.squeeze(), cs_true_tensor.mean(dim=1))
                loss = F.mse_loss(outputs.squeeze(), cs_true_tensor.squeeze())
                
                # Обратное распространение и шаг оптимизации
                loss.backward()
                optimizer.step()

                print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

                
        
    class CustomDataset(Dataset):
        def __init__(self, dataframe):
            self.dataframe = dataframe

        def __len__(self):
            return len(self.dataframe)

        def __getitem__(self, idx):
            # Получаем входные данные и целевое значение
            features = self.dataframe['inputs'][idx]  # Предполагается, что 'inputs' в первой колонке
            target = string2embedding(self.dataframe['target'][idx])
            return features, target
        

    def tune(load_df=False):
        model = RegressionHead()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        print('-' * 30)
        print('Preparing data')
        if not load_df:
            df_train = pd.read_excel("./data/processed/train/solutions.xlsx")
            ###################################
#            df_train = df_train.head(6)
            ###################################
            df_train['predictions'] = df_train.apply(predict, axis=1)
            target = df_train['author_comment_embedding']

            df = pd.DataFrame({
                'inputs': df_train['predictions'],
                'target': target
            })
            df.to_csv("./data/processed/train/solutions_train.csv")
            print('Data saved')
        df = pd.read_csv("./data/processed/train/solutions_train.csv")
        dataset = CustomDataset(df)
        dataloader = DataLoader(dataset, batch_size=5, shuffle=True)
        print('-' * 30)
        print('Starting to train the model')
        train_model(model, dataloader, optimizer)
        print('-' * 30)
        print('Model was trained')
        
        save_model(model, './data/complete/regression_head.pt')

    tune(load_df=True)
