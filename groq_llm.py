from groq import Groq
from groq import Groq
from dotenv import load_dotenv
load_dotenv()
client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)
def ask_llm(question):
    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "user",
                    "content": question
                }
            ],
            temperature=0.7,
            max_tokens=1024
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error: {e}"
def main():
    print("=== Groq LLM Chat ===")
    while True:
        user_prompt = input("\nAsk something to the LLM (or type 'exit'): ")
        if user_prompt.lower() == "exit":
            print("Goodbye!")
            break
        answer = ask_llm(user_prompt)
        print("\nLLM Response:\n")
        print(answer)
if __name__ == "__main__":
    main()
client = Groq(
    api_key="your_api_key_here"
)
user_prompt = input("Ask something to the LLM:\n ")
response = client.chat.completions.create(
    model="llama-3.3-70b-versatile",
    messages=[
        {
            "role": "user",
            "content": user_prompt
        }
    ]
)
print("\nLLM Response:\n")
print(response.choices[0].message.content)
