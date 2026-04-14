# AIzaSyBH5ogwI59x_Czui8JKa8d7dtSISmXb5L4

from google import genai

client = genai.Client()

response = client.models.generate_content(
    model="gemini-3-flash-preview",
    contents="Explain how AI works in a few words",
)

print(response.text)
