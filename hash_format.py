import pandas as pd
#sheets = ['SET 51 15 days', 'SET52 15 days', 'set53 15 days']
filename = 'all_data_setD.xlsx'
sheets = ['setD 90 days', ]
def list_to_string(mylist): 
    # return list as string with parentheses at the beginning and end
    return str(mylist).replace('[', '(').replace(']', ')')

for i in range(len(sheets)): 
    data = pd.read_excel(filename, sheet_name=sheets[i])
    #print(data.head())
    fraud = data['bchash'].unique().tolist()
    print(len(fraud))

    
    with open(f'setD_90.txt', 'w') as f:
        f.write(list_to_string(fraud))
    