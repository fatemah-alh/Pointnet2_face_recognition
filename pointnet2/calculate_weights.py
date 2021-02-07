
import pandas as pd
import numpy as np
def Calculat_weights():
    #read a data frame which contains all point clouds
    df=pd.read_csv('/data/falhamdoosh/data_unicod/dataframe.csv')
    grp=df.groupby('label').count()
    label=list(grp['filename'].index)
    count=list(grp['filename'])
    d={'label':label,'count':count}
    dff=pd.DataFrame(data=d)
    total=dff['count'].sum()
    #calculate weights
    dff['weight']=(total)/(len(dff)*dff['count'])    
    weights=list(dff['weight'])
    print(dff,len(dff))
    # check the direction
    np.save('/data/falhamdoosh/data_unicod/weights.npy',weights)
    print("Saved successfully!")
Calculat_weights()



