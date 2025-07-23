
from agent import Agent
from models import OllamaModel
from SimpleAgent_with_tools.Tools.tools import basic_calculator, reverse_string

from dotenv import load_dotenv
load_dotenv()


if __name__ == "__main__":
    """
    Instructions for using this agent:

    Example queries you can try:
    1. Calculator operations:
       - "Calculate 15 plus 7"
       - "What is 100 divided by 5?"
       - "Multiply 23 and 4"

    2. String reversal:
       - "Reverse the word 'hello world'"
       - "Can you reverse 'Python Programming'?"

    3. General questions (will get direct responses):
       - "Who are you?"
       - "What can you help me with?"

    Ollama Commands (run these in terminal):
    - Check available models:    'ollama list'
    - Check running models:      'ps aux | grep ollama'
    - List model tags:          'curl http://localhost:11434/api/tags'
    - Pull a new model:         'ollama pull mistral'
    - Run model server:         'ollama serve'
    """

    tools = [basic_calculator, reverse_string]

    # Uncomment below to run with OpenAI
    # model_service = OpenAIModel
    # model_name = 'gpt-3.5-turbo'
    # stop = None

    # Using Ollama with llama2 model
    model_service = OllamaModel
    model_name = "mistral"  # Can be changed to other models like 'mistral', 'codellama', etc.
    stop = "<|eot_id|>"

    agent = Agent(tools=tools, model_service=model_service, model_name=model_name, stop=stop)

    print("\nWelcome to the AI Agent! Type 'exit' to quit.")
    print("You can ask me to:")
    print("1. Perform calculations (e.g., 'Calculate 15 plus 7')")
    print("2. Reverse strings (e.g., 'Reverse hello world')")
    print("3. Answer general questions\n")

    try:
        while True:
            prompt = input("Ask me anything: ")
            if prompt.lower() == "exit":
                break

            agent.work(prompt)
    except KeyboardInterrupt or SystemExit:
        print("\nProgram interrupted by user. Performing cleanup...")

    finally:
        print(" *** Agent finished *** ")
