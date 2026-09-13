from openai import OpenAI
from pydantic import BaseModel, Field, ValidationError
from typing import List, Optional
from dotenv import load_dotenv
import json
import os
import shutil
import io
import sys
import unittest
from unittest.mock import patch

# Ensure emoji/unicode output (used throughout this CLI) doesn't crash on
# consoles whose default encoding can't represent it (e.g. Windows cp1252).
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

load_dotenv()

FILEPATH = 'structured_restaurant_data.json'
BACKUP_PATH = 'structured_restaurant_data.json.bak'
EXAMPLE_RESTAURANT_PARAGRAPH = 'Down in **Santa Monica**, **Mar de Cortez** serves as a **sun-drenched**, **casual taqueria** specializing in **Baja-style seafood**. With a **4.2/5** rating, it captures the salt-air energy of the coast through its signature beer-battered snapper tacos and zesty octopus ceviche, making it a premier spot for open-air dining near the pier. Price range: '

MODEL_ID = "openai/gpt-4o-mini"  # small, inexpensive model (via OpenRouter) -- plenty for structured extraction
MAX_REPAIR_ATTEMPTS = 3


class RestaurantData(BaseModel):
    id: Optional[int] = None
    name: str
    neighborhood: Optional[str] = None
    cuisine: Optional[str] = None
    description: Optional[str] = None
    rating: Optional[float] = None
    price_range: Optional[str] = None

## Exercise 1: Integrate the LLM model from Lesson 1

# You will need the LLMs you defined in lesson 1 to structure new restaurant paragraph inputs. In addition to these functions, you will need to implement a new function `new_data_entry_process(paragraph, itemId)`, which takes inputs:

# -   `paragraph`: the new restaurant paragraph;
    
# -   `itemId`: the ID of this new item.
    

# This new function combines and uses the generative models you defined in lesson 1 to structure a given new restaurant paragraph.

# In your `restaurant_data_management.py`, copy and paste the following code block and complete the functions.

#   **Important**: Take a screenshot of your implementation of the `new_data_entry_process()` and name it `M1L3_new_data_entry_process.jpg`.

#Update your restaurant_data_structure_prompt_generation
def restaurant_data_structure_prompt_generation(restaurant_paragraph):
    system_msg = (
        "You are a data extraction engine. Read the restaurant description and "
        "return ONLY a single JSON object (no markdown, no commentary) with these keys: "
        "name (string), neighborhood (string or null), cuisine (string or null), "
        "description (string or null), rating (number or null), price_range (string or null)."
    )
    prompt_txt = f'Restaurant description:\n"""\n{restaurant_paragraph}\n"""\n\nJSON:'
    return system_msg, prompt_txt

def llm_model(system_msg, prompt_txt, params=None):
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.environ.get("OPENROUTER_API_KEY"),
    )
    response = client.chat.completions.create(
        model=MODEL_ID,
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": prompt_txt},
        ],
    )
    return response.choices[0].message.content

def JSON_auto_repair_prompts(response, error_message):
    system_msg = (
        "You are a JSON repair engine. You will be given a broken/invalid JSON string "
        "and the validation error it produced. Fix it and return ONLY the corrected, "
        "valid JSON object, with no markdown formatting and no commentary."
    )
    prompt_txt = (
        f"Invalid JSON:\n\"\"\"\n{response}\n\"\"\"\n\n"
        f"Validation error:\n{error_message}\n\n"
        "Corrected JSON:"
    )
    return system_msg, prompt_txt

def _strip_code_fences(text):
    text = text.strip()
    if text.startswith("```"):
        text = text.split("```")[1]
        text = text.removeprefix("json").strip()
    return text.strip()

def new_data_entry_process(paragraph, itemId):
    system_msg, prompt_txt = restaurant_data_structure_prompt_generation(paragraph)
    raw_response = llm_model(system_msg, prompt_txt)

    for attempt in range(MAX_REPAIR_ATTEMPTS):
        try:
            candidate = _strip_code_fences(raw_response)
            parsed = json.loads(candidate)
            restaurant = RestaurantData(id=itemId, **parsed)
            return restaurant.model_dump()
        except (json.JSONDecodeError, ValidationError) as e:
            print(f"⚠️ Invalid JSON on attempt {attempt + 1}/{MAX_REPAIR_ATTEMPTS}: {e}")
            repair_system_msg, repair_prompt_txt = JSON_auto_repair_prompts(raw_response, str(e))
            raw_response = llm_model(repair_system_msg, repair_prompt_txt)

    raise ValueError(f"Failed to obtain valid JSON after {MAX_REPAIR_ATTEMPTS} repair attempts.")

def load_data(file_path):
    if not os.path.exists(file_path):
        return []
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_data(file_path, backup_path, data):
    if os.path.exists(file_path):
        shutil.copy(file_path, backup_path)
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def show_restaurant_card(res, index):
    print(f"\n--- Record {index} ---")
    for key, value in res.items():
        print(f"{key}: {value}")

def _valid_index(index_str, data):
    return index_str.isdigit() and 0 <= int(index_str) < len(data)

def manage_restaurants(file_path, backup_path):
    while True:
        data = load_data(file_path)
        print(f"\n🏨 RESTAURANT DATABASE | Records: {len(data)}")
        print("1. Browse All (Names)")
        print("2. View Detailed Record")
        print("3. Add New Restaurant")
        print("4. Edit Restaurant Info")
        print("5. Delete Restaurant")
        print("6. Exit")

        choice = input("\nAction: ")

        if choice == '1':
            print("\n--- Current Listings ---")
            for index, res in enumerate(data):
                print(f"{index}. {res.get('name', 'N/A')}")

        elif choice == '2':
            index = input("Enter record index: ")
            if _valid_index(index, data):
                show_restaurant_card(data[int(index)], int(index))
            else:
                print("Invalid index.")

        elif choice in ['3', '4', '5']:
            # Strict Security Warning
            print("\n❗ SECURITY WARNING: You are entering write-mode.")
            print("Changes will be saved to the database immediately.")
            confirm = input("Are you sure? (type 'yes' to proceed): ").lower()
            if confirm != 'yes':
                print("Operation cancelled.")
                continue

            if choice == '3': # ADD NEW DATA
                itemId = 1000000 + len(data) + 1 #the item id for the new data
                paragraph = input("Describe the new restaurant: ")
                new_restaurant = new_data_entry_process(paragraph, itemId)
                data.append(new_restaurant)
                save_data(file_path, backup_path, data)
                print("✅ Restaurant added.")

            elif choice == '4': # EDIT DATA
                index = input("Enter record index to edit: ")
                if _valid_index(index, data):
                    record = data[int(index)]
                    for key in record:
                        new_value = input(f"{key} [{record[key]}]: ")
                        if new_value:
                            record[key] = new_value
                    save_data(file_path, backup_path, data)
                    print("✅ Record updated.")
                else:
                    print("Invalid index.")

            elif choice == '5': # DELETE DATA
                index = input("Enter record index to delete: ")
                if _valid_index(index, data):
                    data.pop(int(index))
                    save_data(file_path, backup_path, data)
                    print("✅ Record deleted.")
                else:
                    print("Invalid index.")

        elif choice == '6': # EXIT
            break
        else:
            print("Invalid input.")

# RUN THE UI
if __name__ == "__main__":
    manage_restaurants(FILEPATH, BACKUP_PATH)
