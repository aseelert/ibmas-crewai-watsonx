import os
import shutil
from datetime import datetime
from crewai import Crew, Task, Agent
from crewai_tools import SerperDevTool, ScrapeWebsiteTool
from langchain_ibm import WatsonxLLM

# List of companies with Yahoo Finance stock symbols
companies = {
    "IBM": "IBM",
    "Infosys": "INFY",
    "Accenture": "ACN",
    "Google_Alphabet": "GOOGL"
}

# Parameters for Watsonx LLM
parameters = {"decoding_method": "sample", "max_new_tokens": 500, "temperature": 0.2}

# Dedicated models for each agent
historical_data_model = "ibm/granite-3-8b-instruct"
current_data_model = "meta-llama/llama-3-405b-instruct"
report_generation_model = "mistralai/mistral-large"

# Calculate the date range (last two years)
end_date = datetime.now().date()
start_date = datetime(datetime.now().year - 2, 1, 1).date()

# Initialize Watsonx LLM instances
historical_data_llm = WatsonxLLM(
    model_id=historical_data_model,
    url="https://us-south.ml.cloud.ibm.com",
    params=parameters,
    project_id=os.getenv("WATSONX_PROJECT_ID"),
    apikey=os.getenv("WATSONX_APIKEY")
)

current_data_llm = WatsonxLLM(
    model_id=current_data_model,
    url="https://us-south.ml.cloud.ibm.com",
    params=parameters,
    project_id=os.getenv("WATSONX_PROJECT_ID"),
    apikey=os.getenv("WATSONX_APIKEY")
)

report_writer_llm = WatsonxLLM(
    model_id=report_generation_model,
    url="https://us-south.ml.cloud.ibm.com",
    params=parameters,
    project_id=os.getenv("WATSONX_PROJECT_ID"),
    apikey=os.getenv("WATSONX_APIKEY")
)

# Tool for online search using SerperDevTool (if needed)
search_tool = SerperDevTool(api_key=os.getenv("SERPER_API_KEY"))

# Helper function to prepare directory and clear existing files
def prepare_directory(directory_path):
    """Ensure directory exists, clearing it if it already contains files."""
    if os.path.exists(directory_path):
        shutil.rmtree(directory_path)  # Clear existing content
    os.makedirs(directory_path, exist_ok=True)

# Prepare the main data directory and individual company directories
prepare_directory("data")
for company_name in companies.keys():
    prepare_directory(f"data/{company_name.lower()}")

# Define agents with refined roles and backstories
historical_data_collector = Agent(
    llm=historical_data_llm,
    role="Historical Data Specialist",
    goal="Provide a two-year financial trend analysis on revenue growth, market share, and profitability for each company.",
    backstory=(
        "A skilled financial analyst specializing in uncovering trends in historical performance data. Known for expertise in long-term "
        "trend identification, financial ratio analysis, and strategic insights based on historical data."
    ),
    allow_delegation=False,
    tools=[search_tool],
    verbose=1,
)

# Define current data collector agents, focusing on scraping
current_data_collectors = {}
for company_name, stock_symbol in companies.items():
    scraper_website_tool = ScrapeWebsiteTool(website_url=f"https://finance.yahoo.com/quote/{stock_symbol}")
    current_data_collectors[company_name] = Agent(
        llm=current_data_llm,
        role="Real-Time Market Data Analyst",
        goal="Capture the latest stock price, quarterly earnings, and relevant news articles for real-time insights.",
        backstory=(
            "A market data analyst who excels in real-time analysis of company performance. Responsible for tracking live stock data, "
            "earnings releases, and news to provide up-to-the-minute insights for short-term investment decisions."
        ),
        allow_delegation=False,
        tools=[scraper_website_tool],
        verbose=1,
    )

report_writer = Agent(
    llm=report_writer_llm,
    role="Investment Strategy Analyst",
    goal="Synthesize historical trends and real-time data into a detailed investment report for each company, providing strategic insights.",
    backstory=(
        "An experienced investment strategist with a reputation for creating in-depth reports that merge historical performance with "
        "current data to forecast future performance. Known for delivering actionable insights to guide investment decisions."
    ),
    allow_delegation=False,
    verbose=1,
)

# Define data collection tasks for each company
data_collection_tasks = []
for company_name, stock_symbol in companies.items():
    # Define directory for the company and ensure it exists
    company_dir = f"data/{company_name.lower()}"

    # Historical data task (last two years)
    historical_file_path = os.path.join(company_dir, "historical_data_output.md")
    data_collection_tasks.append(Task(
        description=f"Collect detailed financial performance data for {company_name} from {start_date} to {end_date}. Focus on revenue, growth rate, profitability, and any notable strategic moves.",
        description=f"Collect financial data for {company_name} from {start_date} to {end_date}, focusing on revenue, growth rate, and profitability.",
        expected_output=(
            f"A markdown report of {company_name}'s financial performance over the last two years, "
            "covering annual and quarterly revenue, growth rate, and profitability trends.\n\n"
            "Include the following table format for quarterly and annual revenue:\n\n"
            "| Year | Quarter 1       | Quarter 2       | Quarter 3       | Quarter 4       | Annual         |\n"
            "|------|-----------------|-----------------|-----------------|-----------------|----------------|\n"
            "| 2022 | $18.1 billion   | $18.9 billion   | $19.2 billion   | $20.1 billion   | $76.3 billion  |\n"
            "| 2023 | $19.5 billion   | $20.2 billion   | $20.5 billion   | $21.3 billion   | $81.5 billion  |\n"
            "| 2024 | $20.8 billion   | $21.5 billion   | $21.8 billion   | $22.6 billion   | $86.7 billion  |\n\n"
            "Provide an analysis section below the table that discusses revenue trends, seasonal variations, and annual growth."
        ),
        output_file=historical_file_path,
        agent=historical_data_collector,
    ))

    # Current data task
    current_file_path = os.path.join(company_dir, "current_data_output.md")
    data_collection_tasks.append(Task(
       description=(
            f"Fetch the latest stock price, quarterly earnings, press releases, and important news articles for {company_name} from Yahoo Finance. "
            "Identify recent trends, strategic moves, HR changes, or any issues impacting company growth."
        ),
        expected_output=(
            f"A markdown summary of {company_name}'s current stock performance, recent earnings, latest press releases, and key news articles, "
            "formatted as follows:\n\n"
            "### {company_name} - Latest Stock and News Report\n\n"
            "| Date       | Information Type        | Details                                                                                           |\n"
            "|------------|------------------------|---------------------------------------------------------------------------------------------------|\n"
            "| 2023-09-15 | Stock Price             | **$2000**                                                                                        |\n"
            "| 2023-09-15 | Quarterly Earnings      | Reported **$15 billion** net income, **$60 billion** revenue, with a growth rate of **20%**.     |\n"
            "| 2023-09-10 | Press Release           | Announced acquisition of XYZ company to expand market reach in AI technology.                     |\n"
            "| 2023-08-25 | News Article            | [Source: Financial Times] Discussed potential HR restructuring to improve operational efficiency. |\n\n"
            "Ensure each entry is up-to-date, with dates listed for each record and links to sources when applicable."
        ),
        output_file=current_file_path,
        agent=current_data_collectors[company_name],
    ))

# Define report generation tasks for each company
report_tasks = []
for company_name in companies.keys():
    company_dir = f"data/{company_name.lower()}"
    report_file_path = os.path.join(company_dir, "investment_report_output.md")
    report_tasks.append(Task(
        description=f"Generate a comprehensive investment report for {company_name}, integrating both historical and current data.",
        expected_output=(
            "A markdown report that includes a detailed executive summary, sections on historical trends and recent performance, strategic analysis of the company, "
            "and a final investment recommendation with a 5-star rating. Also have a detailed report what are the main top 3 drivers for the current business in the future."
        ),
        output_file=report_file_path,
        agent=report_writer,
    ))

# Define a comparison task to summarize all companies
comparison_output_file = "data/comparison_output.md"
comparison_task = Task(
    description=(
        "Using all collected data, create a detailed markdown table comparing revenue, growth rate, profitability, market share, and additional financial indicators. "
        "Provide a strategic recommendation and rating for each company based on recent performance, historical trends, and current market position."
    ),
    expected_output=(
        "A markdown table comparing key metrics for each company, followed by a summary recommendation and rating. Each rating reflects overall investment potential based on "
        "financial stability, growth prospects, and market position.\n\n"
        "| Company       |Stock Price| Revenue (USD) | Growth Rate | Profitability | Market Share | P/E Ratio | Debt-to-Equity | Dividend Yield | Rating     |\n"
        "|---------------|---------------|---------------|-------------|---------------|--------------|-----------|----------------|----------------|------------|\n"
        "| IBM           |216 USD        | $75B          | 3.2%        | 12%           | 10%          | 24.5      | 1.2            | 4.8%           | *****      |\n"
        "| Infosys       |35 USD         | $15B          | 8.5%        | 15%           | 2%           | 28.1      | 0.3            | 1.5%           | ****       |\n"
        "| Accenture     |50 USD         | $50B          | 6.0%        | 10%           | 7%           | 30.2      | 0.8            | 1.6%           | *****      |\n"
        "| Google        |500 USD        | $180B         | 10.0%       | 20%           | 25%          | 35.0      | 0.1            | 0%             | *****      |\n\n"
        "### Rating Explanation\n"
        "- **★★★★★**: Strong buy – Excellent financial stability, growth potential, and market position.\n"
        "- **★★★★**: Buy – Good investment potential, with strong financials and growth prospects.\n"
        "- **★★★**: Hold – Moderate investment potential with average growth and profitability.\n\n"
        "Each recommendation is based on historical and current performance data and strategic trends for each company."
    ),
    output_file=comparison_output_file,
    agent=report_writer,
)

# Assemble all tasks and agents into a Crew
crew = Crew(
    agents=[historical_data_collector] + list(current_data_collectors.values()) + [report_writer],
    tasks=data_collection_tasks + report_tasks + [comparison_task],
    verbose=1
)

# Execute the tasks and print the results
print(crew.kickoff())
