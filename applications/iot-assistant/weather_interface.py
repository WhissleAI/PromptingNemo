import requests
import yaml
import openai

# Load configuration from YAML file
with open("config.yml", 'r') as stream:
    config = yaml.safe_load(stream)

# Set up API key from the config
openai.api_key = config['openai']['api_key']
weather_api_key = config['weatherapi']['api_key']

def get_weather(location):
    """ Get weather for a specific location using WeatherAPI. """
    url = f"http://api.weatherapi.com/v1/current.json?key={weather_api_key}&q={location}&aqi=no"
    response = requests.get(url)
        
    if response.status_code == 200:
        weather_data = response.json()
        location_name = weather_data['location']['name']
        country = weather_data['location']['country']
        temp_celsius = weather_data['current']['temp_c']
        condition = weather_data['current']['condition']['text']
        humidity = weather_data['current']['humidity']
        
        weather_report = (
            f"Weather in {location_name}, {country}:\n"
            f"Temperature: {temp_celsius}°C\n"
            f"Condition: {condition}\n"
            f"Humidity: {humidity}%"
        )
        return weather_report
    else:
        return "Failed to retrieve weather data."

def weather_agent(prompt):
    completion = openai.ChatCompletion.create(
        model="gpt-4",
        temperature=0,
        messages=[
            {"role":"system", "content": "In the query, you extract a location from the user's request. Output only location name."},
            {"role":"user", "content": prompt},
        ]
    )
    location_response = completion.choices[0].message.content.strip()
    print(f"Location response: {location_response}")
    # Assuming the model will reply with a phrase like "The weather in [location] is ..."
    # Extract the location from the response, e.g., "The weather in San Jose is ..."
    if "in " in location_response:
        location = location_response.split("in ")[1].split(" ")[0]
    else:
        # Default location extraction
        location = location_response
    
    print(f"Extracted location: {location}")
    weather_report = get_weather(location)
    return weather_report

