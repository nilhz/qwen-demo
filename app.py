import llama_cpp
import sys

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

MODEL_PATH = "./LLM/Qwen3.5-4B-Q4_K_M.gguf"
llm = llama_cpp.Llama(
    model_path=MODEL_PATH,
    model_type='qwen35',
    n_ctx=8*1024,
    n_threads=8,
    n_gpu_layers=-1,
    verbose=False
)

print(f"\nAssistant GPU offload supported: {llama_cpp.llama_supports_gpu_offload()}")
print("Chat with Qwen 3.5-4B model. Type '/quit' to exit, '/clear' to reset the conversation.")

conversation = [
    {"role": "system", "content": "Você é um assistente de IA. Responda de forma sucinta, com menos de 400 tokens."}
]

while True:
    user_input = input("\nYou: ").encode('utf-8', errors='replace').decode('utf-8', errors='replace')
    if user_input.lower() == '/quit':
        print("Goodbye!")
        break

    if user_input.lower() == '/clear':
        conversation = [conversation[0]]
        print("Conversation cleared.")
        continue

    message = {"role": "user", "content": user_input}
    conversation.append(message)

    output = llm.create_chat_completion(
        messages=conversation,
        max_tokens=512,
        temperature=0.7,
        top_p=0.9,
        stream=True
    )

    print("\nAssistant: ", end="", flush=True)
    response = ""
    for chunk in output:
        try:
            delta = chunk["choices"][0]["delta"].get("content", "")
            if delta:
                delta = delta.encode('utf-8', errors='replace').decode('utf-8', errors='replace')
                print(delta, end="", flush=True)
                response += delta
        except (UnicodeEncodeError, UnicodeDecodeError):
            continue
    
    response = response.encode('utf-8', errors='replace').decode('utf-8', errors='replace')
    print()

    conversation.append({"role": "assistant", "content": response})