def fakeDataFrame(dataframe):
    new_dataframe = dataframe.copy(deep=True)

    for col in new_dataframe:
        for i in range(new_dataframe[col].size):
            chance = random.random()
            if(chance < 0.2):
                if(new_dataframe[col][i] == 5):
                    new_dataframe[col][i] = new_dataframe[col][i]-1
                elif(new_dataframe[col][i] == 0):
                    new_dataframe[col][i] = new_dataframe[col][i]+1
                else:
                    if(random.random() > 0.5):
                        new_dataframe[col][i] = new_dataframe[col][i]-1
                    else:
                        new_dataframe[col][i] = new_dataframe[col][i]+1

    return new_dataframe

def makeManyRevievs(dataToChange):
    
    new_data1 = fakeDataFrame(dataToChange)
    new_data2 = fakeDataFrame(dataToChange)

    new_data1.columns = new_data1.columns.str.replace('0', '1')
    new_data2.columns = new_data2.columns.str.replace('0', '2')

    new_data = pd.concat([new_data1, new_data2], axis=1)
    new_data = pd.concat([dataToChange, new_data], axis=1)

    new_data.to_csv("C:\\Users\\Luberski\\Repos\\Systemy_Multimedialne\\Lab10\\many_revievs.csv", index=False)
    return new_data