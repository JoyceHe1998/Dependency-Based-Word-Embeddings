import model_result
import pandas as pd
import numpy as np

df = pd.read_csv('aminer_mag_combined_cs_keywords.csv')
# df['SIMILAR_WORDS'][0] = model_result.get_most_similar_words('computerscience')
result_list = [None] * len(df['NORMALIZED_NAME'])
for i in range(120):
    if i % 10 == 0:
        print(i)
    keyword = df['NORMALIZED_NAME'][i].replace(' ', '')
    result = model_result.get_most_similar_words(keyword)
    result = result if result != -1 else 'Not Found'
    result_list[i] = result
df['SIMILAR_WORDS'] = pd.Series(np.array(result_list))
df.to_csv('aminer_mag_combined_cs_keywords.csv', index=False)

