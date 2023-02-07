import pandas as pd
import glob
import os

# Auxiliary functions

def minmax(x):
  if x < -7:
    return x

def time_to_take_out_the_trash(x):
    length_x = len(x)
    if x.count(' ') / length_x < 0.25:
        if x.count(' ') / len(x) < 0.25:
            if len(x.splitlines()) / length_x < 0.25:
                if '{' not in x and '}' not in x and "Welcome to r/" not in x:
                    if '`' not in x and '´' not in x:
                        return x



if __name__ == '__main__':


    ## Subreddits thought to contain controversial or irrelevant content

    excluded_dfs = [12, 13, 27,28,31,32,34,42,49,55,58,62,63,64,71,80,81,93,100,103,104,105,106,107,116]



    for i in range(1, 200, 1):
        print('comments_' + str(i) + '.csv')
        if i not in excluded_dfs:
                df1 = pd.read_csv('comments_level_1_' + str(i) + '.csv', lineterminator='\n')
                df1.dropna(inplace=True)
                df1['score'] = df1['score'].apply(lambda x: minmax(x))
                df1.dropna(inplace=True)

                df2 = pd.read_csv('comments_level_2_' + str(i) + '.csv', lineterminator='\n')
                df2['score'] = df2['score'].apply(lambda x: minmax(x))
                df2.dropna(inplace=True)
                try:
                    df3 = pd.read_csv('comments_level_3_' + str(i) + '.csv', lineterminator='\n')
                    df3['score'] = df3['score'].apply(lambda x: minmax(x))
                    df3.dropna(inplace=True)
                except:
                    pass
                if df3:
                    df = pd.concat([df1, df2, df3], axis=0, ignore_index=True)
                    del df1, df2, df3
                    df.dropna(inplace=True)
                    print(len(df))
                else:
                    df = pd.concat([df1, df2], axis=0, ignore_index=True)
                    del df1, df2
                    df.dropna(inplace=True)
                    print(len(df))
                
                n_comments_dict = {}
                for index, row in df.iterrows():
                    if row['root'] in n_comments_dict:
                        n_comments_dict[row['root']] += 1
                    else:
                        n_comments_dict[row['root']] = 1
                
                df['text'] = df['text'].apply(lambda x: time_to_take_out_the_trash(x))
                df.dropna(inplace=True)
                
                df.drop(columns=['root'], inplace=True)
                df.to_csv('./negative/negative_' + str(i) + '.csv', index=False)
                print(len(df))
                del df

    path = './negative/'
    all_files = glob.glob(os.path.join(path , "*.csv"))

    li = []

    for filename in all_files:
        df = pd.read_csv(filename, index_col=None, header=0)
        li.append(df)

    trainr = pd.concat(li, axis=0, ignore_index=True)
    trainr.to_csv('final.csv', index=False)
    