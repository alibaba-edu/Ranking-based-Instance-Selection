import pandas as pd
import glob

def get_img():
    find_img=pd.read_csv('CARS_98N_list.csv').to_numpy()
    find_img=[['https://i.pinimg.com/736x/'+j[0][len('https://i.pinimg.com/736x/'):],j[1]] for j in find_img]
    return find_img

import os,wget,tqdm
def DownloadImage(key,clas,url,out_dir):
    if not os.path.exists(os.path.join(out_dir,clas)):
        os.mkdir(os.path.join(out_dir,clas))
    filename = os.path.join(out_dir,clas, key)
    if os.path.exists(filename):
        return
    try:
        wget.download(url, out=filename)
    except:
        print(url,'cannot be download')
        return
if __name__ == "__main__":
    df=get_img()
    for i in tqdm.tqdm(df):
        key=i[0].split('/')[-1]
        clas=str(i[1])
        url=str(i[0])
        out_dir='/data/cars_noise/img_v3'
        if key.endswith('.gif'):
            continue
        DownloadImage(key,clas,url,out_dir)