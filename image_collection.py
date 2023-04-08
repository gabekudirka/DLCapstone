import requests
import shutil
import pandas as pd
import time
import threading

def request_imgs(df, dir_path, resolution=440):
    for i, row in df.iterrows():
        # xmin, ymin, xmax, ymax, filename, _, _ = row
        URL = f'https://imagery.dcgis.dc.gov/dcgis/rest/services/Ortho/Ortho_2021/ImageServer/exportImage?bbox={row["xmin"]}%2C{row["ymin"]}%2C{row["xmax"]}%2C{row["ymax"]}.0&bboxSR=&size={resolution}%2C{resolution}&imageSR=&time=&format=png&pixelType=U8&noData=&noDataInterpretation=esriNoDataMatchAny&interpolation=+RSP_BilinearInterpolation&compression=&compressionQuality=&bandIds=&sliceId=&mosaicRule=&renderingRule=&adjustAspectRatio=true&validateExtent=false&lercVersion=1&compressionTolerance=&f=image'
        r = requests.get(url = URL, stream=True, timeout=None)
        if r.status_code == 200:
            path = dir_path + row['filename']
            with open(path, 'wb') as f:
                r.raw.decode_content = True
                shutil.copyfileobj(r.raw, f) 
        time.sleep(0.1)
        
def multi_threaded_requests(df, dir_path, num_threads):
    N, _ = df.shape
    rows_per_thread = N // num_threads
    splits = [df.iloc[i:i+rows_per_thread] for i in range(0, len(df), rows_per_thread)]
    threads = [threading.Thread(target=request_imgs, args=(split, dir_path)) for split in splits]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

if __name__ == '__main__':
    df_coords = pd.read_csv('./data/img_metadata/399697_135518_401430_136935.csv')
    dir_path = './data/aerial_images/399697_135518_401430_136935/'
    multi_threaded_requests(df_coords, dir_path, 6)

    