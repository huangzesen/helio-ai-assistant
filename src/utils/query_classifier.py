import subprocess
import re
import time

def is_nasa_ads_query(query, verbose=False):
    prompt = f"Is the following query related to NASA ADS? Please answer with yes or no. Query: {query}"
    try:
        start_time = time.time()  # Start the timer
        result = subprocess.run(
            ["ollama", "run", "deepseek-r1:1.5b", prompt],
            capture_output=True,
            text=True,
            check=True
        )
        end_time = time.time()  # End the timer
        elapsed_time = end_time - start_time  # Calculate elapsed time

        response = result.stdout.strip()
        
        if verbose:
            print(f"Time taken: {elapsed_time:.2f} seconds")  # Print the elapsed time
            print(f"Full response: {response}")  # Print the full response

        # Remove the <think> block
        response = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL).strip()

        # Extract the final answer
        match = re.search(r'\b(yes|no)\b', response, re.IGNORECASE)
        if match:
            answer = match.group(1).lower()
            if verbose:
                print(f"Final answer: {answer}")  # Print the final answer
            return answer == 'yes'
        else:
            if verbose:
                print("Could not find a definitive yes or no answer.")
            return False
    except subprocess.CalledProcessError as e:
        if verbose:
            print(f"Error running local model: {e}")
            print(f"Command output: {e.output}")
            print(f"Command stderr: {e.stderr}")
        return False