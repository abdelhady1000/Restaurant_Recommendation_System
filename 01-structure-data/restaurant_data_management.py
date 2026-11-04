from ibm_watsonx_ai import Credentials
from ibm_watsonx_ai.foundation_models import ModelInference
from pydantic import BaseModel, Field, ValidationError
from typing import List, Optional
import json
import os
import shutil
import io
import unittest
from unittest.mock import patch

FILEPATH = 'structured_restaurant_data.json'
BACKUP_PATH = 'structured_restaurant_data.json.bak'
EXAMPLE_RESTAURANT_PARAGRAPH = 'Down in **Santa Monica**, **Mar de Cortez** serves as a **sun-drenched**, **casual taqueria** specializing in **Baja-style seafood**. With a **4.2/5** rating, it captures the salt-air energy of the coast through its signature beer-battered snapper tacos and zesty octopus ceviche, making it a premier spot for open-air dining near the pier. Price range:' 



def restaurant_data_structure_prompt_generation(restaurant_paragraph):
    base_system_msg = f"""
    You are a precise data extraction assistant. You convert unstructured restaurant descriptions into a single valid JSON object.
    Rules:
    - Output ONLY the JSON object: no explanations, no markdown code fences, no text before or after it.
    - Use exactly these keys: name, location, type, food_style, rating, price_range, signatures, vibe, environment, shortcomings.
    - Use only information stated in the description. Never invent facts.
    - Use double quotes for all keys and strings, and separate every field with a comma.
    - Use single curly braces for the JSON object.
    """
    
    base_user_prompt = f"""
    Task:
    Extract the restaurant information from the description below and return it as one JSON object following the format of the example.
    Field requirements:
    - name (string): the restaurant name.
    - location (string): the city or area.
    - type (string): the kind of restaurant.
    - food_style (string): the cuisine or food style.
    - rating (number): the rating as a float.
    - price_range (integer): convert the dollar signs into the number of dollar symbols (for example, $ = 1, $$ = 2, $$$ = 3).
    - signatures (list of strings): the signature dishes.
    - vibe (string): the short vibe or atmosphere phrase.
    - environment (string): a description of the setting and surroundings.
    - shortcomings (list of strings): the stated drawbacks, or an empty list [] if none are mentioned.
    Return only the JSON object.

    Restaurant description:
    {restaurant_paragraph}

    Example:
    Input Restaurant Description: "**The Gilded Artichoke** brings a **bohemian chic** energy to the hills of **Silver Lake**, operating as an **upscale bistro** that prioritizes **Farm-to-Table Californian** ingredients. The space feels like a high-end greenhouse with its reclaimed wood and floor-to-ceiling windows, perfectly complementing the **4.5/5** rating earned by its lavender-rubbed roasted chicken and delicate heirloom tomato tarts.  Price range: $$$$"
    Output:
   ""
    "name": "Mar de Cortez",
    "location": "Santa Monica",
    "type": "casual taqueria",
    "food_style": "Baja-style seafood",
    "rating": 4.2,
    "price_range": 1,
    "signatures": [
        "beer-battered snapper tacos",
        "zesty octopus ceviche"
    ],
    "vibe": "salt-air energy",
    "environment": "a premier sun-drenched spot for open-air dining near the pier."
    "shortcomings": []
""
    
    """
    return base_system_msg, base_user_prompt

# Might need to explain why we are using granite here (cheap)
def llm_model(system_msg, prompt_txt):
    #system_msg: the system message given to the LLM
    #prompt_txt: the user prompt
    
    model_id = "ibm/granite-4-h-small"

    project_id="skills-network"

    credentials = Credentials(
                    url = "https://us-south.ml.cloud.ibm.com"
                    )

    ### 1.1: Define the model by ModelInference
    model = ModelInference(
        model_id=model_id,
        credentials=credentials,
        project_id=project_id
    )

    ### 1.2: Define the messages
    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": prompt_txt}
    ]

    ### 1.3: Get the final response output and return it
    response = model.chat(messages=messages)
    output_text = response["choices"][0]["message"]["content"]
    
    return output_text


class Restaurant(BaseModel):
    name: str
    location: str
    type: str
    food_style: str
    rating: Optional[float] = None
    price_range: Optional[int] = None
    signatures: List[str] = Field(default_factory=list)
    vibe: Optional[str] = None
    environment: str
    shortcomings: List[str] = Field(default_factory=list)



def JSON_auto_repair_prompts(candidate_json_output, error_message):
    auto_repair_system_msg = """
    You are an expert JSON repair assistant. You fix invalid or non-conforming JSON so that it passes schema validation.
    Rules:
    - Output ONLY the corrected JSON object: no explanations, no markdown code fences.
    - Keep all correct information unchanged; fix only what the error message points to.
    - Required keys: name (string), location (string), type (string), food_style (string), rating (float or null), price_range (integer or null), signatures (list of strings), vibe (string or null), environment (string), shortcomings (list of strings).
    - Never invent facts that are not already in the original output.
    """
    auto_repair_prompt = f"""
    The following JSON output failed schema validation.

    Original output:
    {candidate_json_output}

    Validation error message:
    {error_message}

    Correct the output according to the error message and return only the fixed, valid JSON object.
    """
    return auto_repair_system_msg, auto_repair_prompt


def new_data_entry_process(paragraph, itemId):
    ### 2.1: Generate the prompts and get the first structured output
    base_system_msg, base_user_prompt = restaurant_data_structure_prompt_generation(paragraph)
    candidate_json_output = llm_model(base_system_msg, base_user_prompt)

    ### 2.2: Validation and Auto Correction loop on the output
    repair_attempts = 0
    while True:
        try:
            Restaurant.model_validate_json(candidate_json_output)
            break
        except ValidationError as e:
            repair_attempts += 1
            if repair_attempts > 3:
                print(f'Restaurant {itemId} could not be repaired after 3 attempts')
                return None
            repair_system_msg, repair_prompt = JSON_auto_repair_prompts(candidate_json_output, e.json())
            candidate_json_output = llm_model(repair_system_msg, repair_prompt)

    ### 2.3: Return the validated entry together with its ID
    restaurant_data = json.loads(candidate_json_output)
    restaurant_data["itemId"] = itemId
    return restaurant_data

def load_data(file_path):
    if not os.path.exists(file_path):
        return []
    with open(file_path, 'r', encoding='utf-8') as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            return []


def save_data(file_path, data, backup_path=BACKUP_PATH):
    # Back up the existing file before overwriting it
    if os.path.exists(file_path):
        shutil.copy(file_path, backup_path)
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


def show_restaurant_card(res, index):
    print(f"\n--- Record {index} ---")
    for key, value in res.items():
        print(f"{key}: {value}")


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
            for i, res in enumerate(data):
                print(f"{i}. {res.get('name', 'N/A')}")

        elif choice == '2':
            index = input("Enter record index: ")
            if index.isdigit() and 0 <= int(index) < len(data):
                index = int(index)
                show_restaurant_card(data[index], index)
            else:
                print("invalid index.")

        elif choice in ['3', '4', '5']:
            # Strict Security Warning
            print("\n❗ SECURITY WARNING: You are entering write-mode.")
            print("Changes will be saved to the database immediately.")
            confirm = input("Are you sure? (type 'yes' to proceed): ").lower()
            if confirm != 'yes':
                print("Operation cancelled.")
                continue

            if choice == '3':  # ADD NEW DATA
                itemId = 1000000 + len(data) + 1  # the item id for the new data

                paragraph = input("Enter the new restaurant description: ")
                new_entry = new_data_entry_process(paragraph, itemId)
                if new_entry is None:
                    print("❌ Could not structure the description. Nothing was added.")
                    continue
                data.append(new_entry)
                save_data(file_path, data)

                print("✅ Restaurant added.")

            elif choice == '4':  # EDIT DATA
                index = input("Enter record index: ")
                if index.isdigit() and 0 <= int(index) < len(data):
                    index = int(index)
                    record = data[index]
                    for key in list(record.keys()):
                        new_value = input(f"{key} [{record[key]}]: ")
                        if new_value != "":
                            record[key] = new_value
                    save_data(file_path, data)
                    print("✅ Record updated.")
                else:
                    print("invalid index.")

            elif choice == '5':  # DELETE DATA
                index = input("Enter record index: ")
                if index.isdigit() and 0 <= int(index) < len(data):
                    data.pop(int(index))
                    save_data(file_path, data)
                    print("✅ Record deleted.")
                else:
                    print("invalid index.")

        elif choice == '6':  # EXIT
            break
        else:
            print("Invalid input.")



class TestRestaurantDatabase(unittest.TestCase):
    
    def setUp(self):
        """Create a temporary clean database for testing."""
        self.test_file = 'structured_restaurant_data_unit_test.json'
        self.test_file_backup = 'structured_restaurant_data_unit_test.json.bak'
        self.initial_data = [{"name": "Test Cafe", "location": "Test City"}]
        with open(self.test_file, 'w') as f:
            json.dump(self.initial_data, f)

    def tearDown(self):
        """Clean up the test file after tests."""
        if os.path.exists(self.test_file):
            os.remove(self.test_file)
        if os.path.exists(self.test_file_backup):
            os.remove(self.test_file_backup)

    @patch('builtins.input')
    @patch('sys.stdout', new_callable=io.StringIO)
    def test_add_and_delete_restaurant_success(self, mock_stdout, mock_input):
        """
        Test Scenario: Add a new restaurant.
        Inputs: '3' (Add), 'yes' (Confirm), 'New Burger Joint', '6' (Exit)
        """
        # We mock the sequence of user inputs
        mock_restaurant = 'The Copper Sprout is a high-concept, Modern Appalachian farm-to-table destination that blends an industrial-chic aesthetic with rustic forest charm, featuring reclaimed wood and amber lighting to create a sophisticated yet cozy vibe. Priced in the $$ category, the menu celebrates seasonal foraging and local heritage, headlined by signature dishes like Cast-Iron Smoked Trout with pickled fiddlehead ferns and hand-foraged Wild Mushroom Risotto with aged goat cheese. The experience is designed to be intimate and earthy, making it a premier spot for those seeking high-quality, smokehouse-influenced cuisine in a refined, atmospheric setting.'
        mock_input.side_effect = ['3', 'yes', mock_restaurant, '6']
        
        # Run the app
        try:
            manage_restaurants(self.test_file, self.test_file_backup)
        except SystemExit:
            pass # Handle exit if your script uses sys.exit()

        # Check if the data was actually saved
        with open(self.test_file, 'r') as f:
            data = json.load(f)
        
        print(data)
        self.assertEqual(len(data), 2)
        self.assertIn("✅ Restaurant added.", mock_stdout.getvalue())

        mock_input.side_effect = ['5', 'yes', 1, '6']
        
        # Run the app
        try:
            manage_restaurants(self.test_file, self.test_file_backup)
        except SystemExit:
            pass # Handle exit if your script uses sys.exit()

        # Check if the data was actually saved
        with open(self.test_file, 'r') as f:
            data = json.load(f)
        
        print(data)
        self.assertEqual(len(data), 1)

    @patch('builtins.input')
    @patch('sys.stdout', new_callable=io.StringIO)
    def test_delete_security_cancel(self, mock_stdout, mock_input):
        """
        Test Scenario: Try to delete but say 'no' to security warning.
        Inputs: '5' (Delete), 'no' (Cancel), '6' (Exit)
        """
        mock_input.side_effect = ['5', 'no', '6']
        
        manage_restaurants(self.test_file, self.test_file_backup)
        
        with open(self.test_file, 'r') as f:
            data = json.load(f)
        
        self.assertEqual(len(data), 1) # Data should remain unchanged
        self.assertIn("Operation cancelled.", mock_stdout.getvalue())
		
if __name__ == "__main__":
    unittest.main() # Unit Test
	    # manage_restaurants(FILEPATH, BACKUP_PATH) # Actual UI Call
