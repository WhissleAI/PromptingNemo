# Whissle Assistant

Whissle is an AI-powered virtual assistant capable of performing a wide range of tasks, including sending emails, interacting with Spotify, sending WhatsApp messages, managing calendar events, retrieving stock prices, managing notes, and more. Whissle is designed to provide interesting, intellectually stimulating conversations while helping users with their daily tasks.

## Features

- **Wake Word Detection:** Listens for the wake word "hello" to start a conversation.
- **Speech Recognition:** Transcribes spoken input using Google Speech Recognition.
- **Task Execution:** Identifies and executes various tasks such as sending emails, playing music on Spotify, sending SMS, and more.
- **Chatting:** Engages in conversational interactions using OpenAI's GPT-4.
- **Supports Multiple Interfaces:** Includes agents for email, Spotify, SMS, WhatsApp, weather, stock prices, notes, messages, and screenshots.

## Getting Started

### Prerequisites

- Python 3.8 or higher
- `pip` package manager
- OpenAI API key
- Ngrok account (for transcribing audio)
- Google Speech Recognition

### Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/yourusername/whissle-assistant.git
    cd whissle-assistant
    ```

2. Install the required dependencies:
    ```sh
    pip install -r requirements.txt
    ```

3. Configure the application:
    - Create a `config.yml` file with the following content:
      ```yaml
      openai:
        api_key: YOUR_OPENAI_API_KEY
      ```

4. Run the application:
    ```sh
    python assistant.py
    ```

## Configuration

The configuration file `config.yml` should include the necessary API keys and other configuration settings.

Example `config.yml`:
```yaml
openai:
  api_key: YOUR_OPENAI_API_KEY

