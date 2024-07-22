import os
import time
from PIL import Image
import requests
import base64
import io
from selenium import webdriver
from selenium.webdriver import ChromeOptions
from selenium.webdriver.common.by import By

def scrape_images(query, num_images, save_path, base_url, max_page):
    options = ChromeOptions()
    options.add_argument("--start-maximized")
    options.add_argument("--no-sandbox")
    options.add_experimental_option("useAutomationExtension", False)
    options.add_experimental_option("excludeSwitches", ["enable-automation"])
    driver = webdriver.Chrome(options=options)
    
    page = 1
    loaded_images = 0
    while loaded_images < num_images and page <= max_page:
        search_url = f"{base_url}{page}"
        driver.get(search_url)
        
        img_elements = driver.find_elements(By.TAG_NAME, 'img')
        src = [img.get_attribute('src') for img in img_elements]

        os.makedirs(save_path, exist_ok=True)

        min_img_size = 100000

        for i in range(len(src)):
            try:
                if src[i] is None or loaded_images >= num_images:
                    continue
                img_name = f"{query}_{loaded_images+1}.jpg"
                img_path = os.path.join(save_path, img_name)
                
                if src[i].startswith('data'):
                    imgdata = base64.b64decode(str(src[i]).split(',')[1])
                    img = Image.open(io.BytesIO(imgdata))
                    img_size = len(imgdata)
                    if img_size > min_img_size:
                        img.save(img_path)
                        loaded_images += 1
                        print(f"Image {loaded_images} downloaded successfully")
                    else:
                        print(f"Image {loaded_images+1} too small to download")
                else:
                    img = Image.open(requests.get(src[i], stream=True).raw).convert('RGB')               
                    img_size = len(img.tobytes())
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
    driver.quit()

if __name__ == "__main__":
    queries = ["apple", "banana", "orange", "grape", "watermelon"]
    base_urls = [
        "https://www.shutterstock.com/zh/search/apple?image_type=photo&page=",
        "https://www.shutterstock.com/zh/search/banana?image_type=photo&page=",
        "https://www.shutterstock.com/zh/search/orange?image_type=photo&page=",
        "https://www.shutterstock.com/zh/search/grape?image_type=photo&page=",
        "https://www.shutterstock.com/zh/search/watermelon?image_type=photo&page="
    ]
    save_paths = [
        "dataset/apple",
        "dataset/banana",
        "dataset/orange",
        "dataset/grape",
        "dataset/watermelon"
    ]
    num_images = 100
    max_pages = [200, 200, 200, 200, 200]

    for query, base_url, save_path, max_page in zip(queries, base_urls, save_paths, max_pages):
        scrape_images(query, num_images, save_path, base_url, max_page)
