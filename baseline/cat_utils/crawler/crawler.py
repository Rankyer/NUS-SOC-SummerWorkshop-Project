import time
from PIL import Image
import requests
import base64
import io
from selenium import webdriver
from selenium.webdriver import ChromeOptions
from selenium.webdriver.common.by import By
import os
import sys

# Chromedriver路径
PATH = "/Users/shirleneliew/Documents/chromedriver-mac-x64"

# 初始化Chromedriver
options = ChromeOptions()
options.add_argument("--start-maximized")
# options.add_argument("--headless")
options.add_argument("--no-sandbox")
options.add_experimental_option("useAutomationExtension", False)
options.add_experimental_option("excludeSwitches", ["enable-automation"])
options.executable_path = PATH

driver = webdriver.Chrome(options=options)

# 定义scrape_images函数
def scrape_images(query, num_images, save_path, base_url, max_page):
    page = 1
    loaded_images = 0
    
    while loaded_images < num_images and page <= max_page:
        search_url = f"{base_url}{page}"
        driver.get(search_url)
        
        # 获取所有图片元素
        img_elements = driver.find_elements(By.TAG_NAME, 'img')
        src = [img.get_attribute('src') for img in img_elements]

        # 在桌面上创建保存目录
        desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")
        save_path = os.path.join(desktop_path, save_path)
        os.makedirs(save_path, exist_ok=True)

        # 定义最小图片大小以跳过缩略图
        min_img_size = 100000

        # 遍历获取的图片URL
        for i in range(len(src)):
            try:
                if src[i] is None or loaded_images >= num_images:
                    continue
                else:
                    img_name = f"{query}_{loaded_images+1}.jpg"
                    img_path = os.path.join(save_path, img_name)
                    
                    # base64编码的图片
                    if src[i].startswith('data'):
                        imgdata = base64.b64decode(str(src[i]).split(',')[1])
                        img = Image.open(io.BytesIO(imgdata))
                        
                        img_size = sys.getsizeof(img.tobytes())
                        if img_size > min_img_size:
                            img.save(img_path)
                            loaded_images += 1
                            print(f"Image {loaded_images} downloaded successfully")
                        else:
                            print(f"Image {loaded_images+1} too small to download")
                    # URL图片
                    else:
                        img = Image.open(requests.get(src[i], stream=True).raw).convert('RGB')               
                        img_size = sys.getsizeof(img.tobytes())
                        if img_size > min_img_size:
                            img.save(img_path)
                            loaded_images += 1
                            print(f"Image {loaded_images} downloaded successfully")
                        else:
                            print(f"Image {loaded_images+1} too small to download")
            except Exception as e:
                print(f"Failed to download image {loaded_images+1}: {e}")

        page += 1
        print(f"Loaded images: {loaded_images}, current page: {page-1}")

# 执行图片抓取
queries = ["Ragdolls", "Singapura cats", "Persian cats", "Sphynx cats", "Pallas cats"]
base_urls = [
    "https://www.shutterstock.com/zh/search/ragdoll?image_type=photo&page=",
    "https://www.shutterstock.com/zh/search/singapura-cats?image_type=photo&page=",
    "https://www.shutterstock.com/zh/search/persian-cats?image_type=photo&page=",
    "https://www.shutterstock.com/zh/search/sphynx-cats?image_type=photo&page=",
    "https://www.shutterstock.com/zh/search/pallas-cats?image_type=photo&page="
]
save_paths = [
    "cat_plus/Ragdolls",
    "cat_plus/Singapura cats",
    "cat_plus/Persian cats",
    "cat_plus/Sphynx cats",
    "cat_plus/Pallas cats"
]
num_images = 2000
max_pages = [200, 5, 200, 200, 200]  # 对应不同种类的最大页数限制

for query, base_url, save_path, max_page in zip(queries, base_urls, save_paths, max_pages):
    scrape_images(query, num_images, save_path, base_url, max_page)

driver.quit()
