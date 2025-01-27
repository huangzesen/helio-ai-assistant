import subprocess
import re

def query_local_model(prompt, history):
    conversation = "\n".join(history + [f"User: {prompt}"])
    try:
        result = subprocess.run(
            ["ollama", "run", "deepseek-r1:14b", conversation],
            capture_output=True,
            text=True,
            check=True
        )
        response = result.stdout.strip()

        # Remove the <think> block
        primary_response = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL).strip()

        return primary_response
    except subprocess.CalledProcessError as e:
        print(f"Error running local model: {e}")
        print(f"Command output: {e.output}")
        print(f"Command stderr: {e.stderr}")
        return None