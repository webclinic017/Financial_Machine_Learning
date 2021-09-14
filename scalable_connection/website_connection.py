"""
Name : website_connection.py in Project: Financial_ML
Author : Simon Leiner
Date    : 22.06.2021
Description: 
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.action_chains import ActionChains

# See: https://selenium-python.readthedocs.io/

# create webdriver object
driver = webdriver.Chrome()

# open login website of scalalbe capital
driver.get("https://de.scalable.capital/login")

# handling cookies
WebDriverWait(driver, 10).until(EC.element_to_be_clickable((By.XPATH,'//*[@id="CybotCookiebotDialogBodyLevelButtonLevelOptinAllowallSelection"]'))).click()

"""
Find HTML STUFF:
Chrome, rightclick, inspect Element

Access by different attributes:
1. ID: best : find_element_by_id
2. Class: third: find_element_by_class_name
3. Name: second: find_element_by_name
4. Tag Name
and many more
"""

# get element login with email adress
email_login = driver.find_element_by_name("username")

email_login.send_keys("")

email_pw = driver.find_element_by_name("password")

email_pw.send_keys("")

login_button = driver.find_element_by_xpath('//*[@id="page"]/div/div[3]/div[2]/div/div/div/form/div/div[2]/button')

login_button.click()

# actions = ActionChains(driver)
# actions.click(element)
# actions.perform()
# actions.move_to_element(element)

#
# try:
#     wait max 10 secs to find the element or quit
    # element = WebDriverWait(driver,10).until(EC.presence_of_element_located((By.ID, "awdasd")))
# except:
#     driver.quit()




# element.clear()
# element.click()
# element.submit()
# driver.back()
# driver.forward()

# send keys
# element.send_keys("Text to Type into searchbox")

# hit enter
# element.send_keys(Keys.RETURN)

# selenuim


# close tap
# driver.close()

# close browser
# driver.quit()

# can also tell the driver to wait
