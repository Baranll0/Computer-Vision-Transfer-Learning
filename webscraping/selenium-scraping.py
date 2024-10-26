import time
from selenium import webdriver
from bs4 import BeautifulSoup
import pandas as pd
import csv

driver = webdriver.Chrome()
# Open the webpage
url = "https://www.haberler.com/spor/lionel-messi-dunya-futbol-tarihinde-bir-ilke-17960665-haberi/"
driver.get(url)

# Find the image element using XPath
xpath = '//*[@id="HaberImage"]'
try:
    image_element = driver.find_element("xpath", xpath)
    image_src = image_element.get_attribute("src")

    # Save the image src to a CSV file
    with open("image_src.csv", mode="a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([image_src])
    print("Image src saved to image_src.csv")
except Exception as e:
    print(f"Error: {e}")

# Close the driver
driver.quit()