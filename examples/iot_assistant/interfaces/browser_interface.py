import re
from selenium.webdriver.common.keys import Keys
from selenium import webdriver
from selenium.webdriver.common.action_chains import ActionChains
import time
import openai
import ast
import yaml
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load configuration from YAML file
with open("config.yml", 'r') as stream:
    config = yaml.safe_load(stream)

# Assign variables from the config
openai.api_key = config['openai']['api_key']

# Global driver variable to maintain browser session
driver = None

def initialize_driver(url):
    global driver
    if driver is None:
        driver = webdriver.Chrome()
        logging.info("Initialized new WebDriver instance")
    driver.get(url)
    time.sleep(1)
    return driver

def type_text(target_xpath, target_input, driver):
    try:
        input_box = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.XPATH, target_xpath))
        )
        input_box.send_keys(target_input)
        logging.info(f"Typed '{target_input}' in element with XPath '{target_xpath}'")
    except Exception as e:
        logging.error(f"Error typing text: {e}")

def press_key(target_key, driver):
    key_mapping = {
        "enter": Keys.ENTER,
        "arrowleft": Keys.ARROW_LEFT,
        "arrowright": Keys.ARROW_RIGHT,
        "arrowup": Keys.ARROW_UP,
        "arrowdown": Keys.ARROW_DOWN,
        "backspace": Keys.BACK_SPACE
    }
    if target_key.lower() in key_mapping:
        active_input_box = driver.switch_to.active_element
        active_input_box.send_keys(key_mapping[target_key.lower()])
        logging.info(f"Pressed key '{target_key}'")
    else:
        logging.error(f"Invalid key: {target_key}")

def click_element(target_xpath, driver):
    try:
        element = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.XPATH, target_xpath))
        )
        element.click()
        logging.info(f"Clicked element with XPath '{target_xpath}'")
    except Exception as e:
        logging.error(f"Error clicking element: {e}")

def click_option(target_xpath, driver):
    try:
        option_element = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.XPATH, target_xpath))
        )
        option_element.click()
        logging.info(f"Clicked option with XPath '{target_xpath}'")
    except Exception as e:
        logging.error(f"Error clicking option: {e}")

def move_mouse(target_xpath, driver):
    try:
        element = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.XPATH, target_xpath))
        )
        actions = ActionChains(driver)
        actions.move_to_element(element).perform()
        logging.info(f"Moved mouse to element with XPath '{target_xpath}'")
    except Exception as e:
        logging.error(f"Error moving mouse: {e}")

def navigate_to_url(url, driver):
    driver.get(url)
    logging.info(f"Navigated to URL '{url}'")

def scroll_to_element(target_xpath, driver):
    try:
        element = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.XPATH, target_xpath))
        )
        driver.execute_script("arguments[0].scrollIntoView();", element)
        logging.info(f"Scrolled to element with XPath '{target_xpath}'")
    except Exception as e:
        logging.error(f"Error scrolling to element: {e}")

def scroll_page(direction, driver):
    scroll_mapping = {
        "up": "-250",
        "down": "250"
    }
    if direction.lower() in scroll_mapping:
        driver.execute_script(f"window.scrollBy(0, {scroll_mapping[direction.lower()]})")
        logging.info(f"Scrolled page '{direction}'")
    else:
        logging.error(f"Invalid scroll direction: {direction}")

def switch_to_frame(frame_xpath, driver):
    try:
        frame_element = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.XPATH, frame_xpath))
        )
        driver.switch_to.frame(frame_element)
        logging.info(f"Switched to frame with XPath '{frame_xpath}'")
    except Exception as e:
        logging.error(f"Error switching to frame: {e}")

def switch_to_default_content(driver):
    driver.switch_to.default_content()
    logging.info("Switched to default content")

def execute_instruction(instruction, driver):
    type_regex = "^type\s.{1,}$"
    press_regex = "^press\s(enter|arrowleft|arrowright|arrowup|arrowdown|backspace)$"
    clickxpath_regex = "^clickxpath\s.{1,}$"
    clickoption_regex = "^clickoption\s.{1,}$"
    movemouse_regex = "^movemouse\s.{1,}$"
    selectinput_regex = "^selectinput\s.{1,}$"
    navigate_regex = "^navigate\s.{1,}$"
    scroll_regex = "^scroll\s(up|down)$"
    scroll_to_element_regex = "^scrolltoelement\s.{1,}$"
    switch_to_frame_regex = "^switchtoframe\s.{1,}$"
    switch_to_default_content_regex = "^switchtodefaultcontent$"

    if re.match(type_regex, instruction):
        parts = instruction.split(" ", 2)
        target_xpath = parts[1]
        target_input = parts[2]
        type_text(target_xpath, target_input, driver)
        return f"Typed '{target_input}' in element with XPath '{target_xpath}'"
    elif re.match(press_regex, instruction):
        target_key = instruction.split(" ")[1]
        press_key(target_key, driver)
        return f"Pressed key '{target_key}'"
    elif re.match(clickxpath_regex, instruction):
        target_xpath = instruction.split(" ", 1)[1]
        click_element(target_xpath, driver)
        return f"Clicked element with XPath '{target_xpath}'"
    elif re.match(clickoption_regex, instruction):
        target_xpath = instruction.split(" ",1)[1]

        click_option(target_xpath, driver)
        return f"Clicked option with XPath '{target_xpath}'"
    elif re.match(movemouse_regex, instruction):
        target_xpath = instruction.split(" ", 1)[1]
        move_mouse(target_xpath, driver)
        return f"Moved mouse to element with XPath '{target_xpath}'"
    elif re.match(selectinput_regex, instruction):
        target_xpath = instruction.split(" ", 1)[1]
        # Assuming a function to select input box
        select_input_box(target_xpath, driver)
        return f"Selected input box with XPath '{target_xpath}'"
    elif re.match(navigate_regex, instruction):
        url = instruction.split(" ", 1)[1]
        navigate_to_url(url, driver)
        return f"Navigated to URL '{url}'"
    elif re.match(scroll_regex, instruction):
        direction = instruction.split(" ")[1]
        scroll_page(direction, driver)
        return f"Scrolled page '{direction}'"
    elif re.match(scroll_to_element_regex, instruction):
        target_xpath = instruction.split(" ", 1)[1]
        scroll_to_element(target_xpath, driver)
        return f"Scrolled to element with XPath '{target_xpath}'"
    elif re.match(switch_to_frame_regex, instruction):
        frame_xpath = instruction.split(" ", 1)[1]
        switch_to_frame(frame_xpath, driver)
        return f"Switched to frame with XPath '{frame_xpath}'"
    elif re.match(switch_to_default_content_regex, instruction):
        switch_to_default_content(driver)
        return "Switched to default content"
    else:
        return f"Invalid instruction: {instruction}"

def execute_instructions(instructions_list, wait):
    global driver
    summary = []
    instructions_list = ast.literal_eval(instructions_list)
    for instruction in instructions_list:
        function_name = instruction["function"]
        args = instruction["args"]
        if function_name == "initialize_driver":
            driver = initialize_driver(*args)
            summary.append(f"Initialized driver and navigated to URL '{args[0]}'")
        else:
            formatted_instruction = f"{function_name} {' '.join(args)}"
            result = execute_instruction(formatted_instruction, driver)
            summary.append(result)
    time.sleep(wait)
    return summary

def browser_agent(prompt):
    completion = openai.ChatCompletion.create(
        model="gpt-4",
        temperature=0,
        messages=[
            {"role": "system", "content": "You are a browser automation program that understands natural language requests. Your goal is to convert user requests into a list of instructions for the 'execute_instruction()' function. To initialize the driver, use the 'initialize_driver(url)' function, which takes a URL as its argument. The 'execute_instruction()' function supports methods like type, press, clickxpath, clickoption, movemouse, selectinput, navigate, scroll, scrolltoelement, switchtoframe, and switchtodefaultcontent. Arguments should be single strings and use XPath expressions when applicable. Format your response as a list of dictionaries, where each dictionary contains a 'function' key with the function name and an 'args' key with a list of the function's arguments. Ensure that there are no nested lists within the 'args' key, and args that are not 'type' function are lowercase."},
            {"role": "user", "content": prompt},
        ]
    )
    reply_content = completion.choices[0].message.content
    summary = execute_instructions(reply_content, wait=20)
    
    return "\n".join(summary)

# # Example diverse instructions
# instructions = browser_agent("Go to youtube, search for good food in Milpitas, click the first result, scroll down the page, and then switch back to the main content.")
# print(instructions)
# summary = execute_instructions(instructions, wait=20)
# print("Summary of actions performed:")
# print("\n".join(summary))
