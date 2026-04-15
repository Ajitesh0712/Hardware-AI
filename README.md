HardwareAI
HardwareAI is an AI-powered electronic product recommendation system that allows users to search and refine product queries using natural language.
Features


Natural language search (e.g., "gaming laptop under 1 lakh")


Conversational refinement (e.g., "cheaper", "better battery")


AI-based requirement extraction using Gemini API


Product data fetched from Amazon via RapidAPI


Basic sentiment analysis from product reviews


Recommendation engine for ranking products


Tech Stack


Backend: Python (Flask)


AI: Google Gemini API


Database: MySQL


Frontend: HTML, CSS, JavaScript


External APIs: RapidAPI (Amazon data)


Project Structure
backend/
frontend/
schema.sql
requirements.txt
.env.example
.gitignore
README.md
Setup Instructions


Clone the repository:
git clone https://github.com/your-username/hardwareai.git
cd hardwareai


Create environment file:
cp .env.example .env


Add your API keys in .env:
GEMINI_API_KEY=your_api_key_here
RAPIDAPI_KEY=your_api_key_here
DB_HOST=localhost
DB_USER=root
DB_PASSWORD=your_password
DB_NAME=hardwareai


Install dependencies:
pip install -r requirements.txt


Run the backend server:
python3 app.py


Open in browser:
http://localhost:5001


Usage Examples


gaming laptop under 100000


cheaper options


better battery laptop


lightweight laptop for students


Future Improvements


Fully AI-driven ranking system


Support for multiple e-commerce platforms (Flipkart, etc.)


User preference memory


Advanced conversational interface


Notes


Do not upload .env file to GitHub


Use .env.example as a template for environment variables

