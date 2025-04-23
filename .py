import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

models = genai.list_models()
for model in models:
    print(model.name)
print("Model ID:", model.id)
print("Total models listed:", len(models))
print("Models available:", [model.name for model in models])
print("Done listing models.")
genai.shutdown()
print("Shutdown complete.")
print("All operations completed successfully.")
print("Exiting the program.")
exit(0)
print("Program terminated.")
