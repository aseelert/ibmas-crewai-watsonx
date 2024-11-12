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

# Date range (last two years)
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

# Tool for online search and direct scraping from Yahoo Finance
search_tool = SerperDevTool(api_key=os.getenv("SERPER_API_KEY"))

# Helper function to set up directory structure
def prepare_directory(directory_path):
    if os.path.exists(directory_path):
        shutil.rmtree(directory_path)
    os.makedirs(directory_path, exist_ok=True)

# Prepare directories for output
prepare_directory("data")
for company_name in companies.keys():
    prepare_directory(f"data/{company_name.lower()}")

# Historical data collection agent
historical_data_collector = Agent(
    llm=historical_data_llm,
    role="Historical Data Specialist",
    goal="Collect a two-year financial trend analysis on revenue growth, market share, profitability, and other key metrics for each company.",
    backstory="An analyst focusing on long-term trend identification and strategic insights based on historical data.",
    allow_delegation=False,
    tools=[search_tool],
    verbose=1,
)

# Current data collection agents
current_data_collectors = {}
for company_name, stock_symbol in companies.items():
    scraper_tool = ScrapeWebsiteTool(website_url=f"https://finance.yahoo.com/quote/{stock_symbol}")
    current_data_collectors[company_name] = Agent(
        llm=current_data_llm,
        role="Real-Time Market Data Analyst",
        goal="Capture the latest stock price, quarterly earnings, P/E ratio, ROE, EPS, debt levels, and important news articles for each company.",
        backstory="An analyst who specializes in real-time performance tracking and provides short-term insights.",
        allow_delegation=False,
        tools=[scraper_tool],
        verbose=1,
    )

# Investment strategy report writer agent
report_writer = Agent(
    llm=report_writer_llm,
    role="Investment Strategy Analyst",
    goal="Generate a comprehensive investment report based on historical trends and real-time data, with strategic recommendations.",
    backstory="An experienced strategist providing insights by synthesizing historical and real-time data.",
    allow_delegation=False,
    verbose=1,
)

# Define historical and current data collection tasks
data_collection_tasks = []
for company_name, stock_symbol in companies.items():
    company_dir = f"data/{company_name.lower()}"
    
    # Historical Data Task
    historical_file_path = os.path.join(company_dir, "historical_data_output.md")
    data_collection_tasks.append(Task(
        description=f"Retrieve comprehensive historical financial data for {company_name} from {start_date} to {end_date}. Include metrics such as revenue, growth rate, profitability, and strategic trends.",
        expected_output=(
            f"## {company_name} Historical Financial Performance\n"
            "| Year | Quarter 1       | Quarter 2       | Quarter 3       | Quarter 4       | Annual         |\n"
            "|------|-----------------|-----------------|-----------------|-----------------|----------------|\n"
            "| 2022 | $18.1B          | $18.9B          | $19.2B          | $20.1B          | $76.3B         |\n"
            "| 2023 | $19.5B          | $20.2B          | $20.5B          | $21.3B          | $81.5B         |\n"
            "| 2024 | $20.8B          | $21.5B          | $21.8B          | $22.6B          | $86.7B         |\n\n"
            "### Analysis\n- Discuss trends, seasonal variations, and growth comparisons."
        ),
        output_file=historical_file_path,
        agent=historical_data_collector,
    ))
    
    # Current Data Task
    current_file_path = os.path.join(company_dir, "current_data_output.md")
    data_collection_tasks.append(Task(
        description=(
            f"Collect latest stock price, quarterly earnings, P/E ratio, ROE, EPS, and recent news articles for {company_name}. Identify recent trends, HR changes, or acquisitions impacting the company's growth."
        ),
        expected_output=(
            f"### {company_name} - Latest Financial Overview\n\n"
            "| Date       | Metric               | Value/Details                                                                                       |\n"
            "|------------|----------------------|-----------------------------------------------------------------------------------------------------|\n"
            "| 2023-09-15 | Stock Price          | $2000                                                                                               |\n"
            "| 2023-09-15 | Quarterly Earnings   | **$15 billion** net income, **$60 billion** revenue, **20%** growth rate                           |\n"
            "| 2023-09-10 | P/E Ratio            | 24.5                                                                                               |\n"
            "| 2023-09-10 | ROE                  | 15%                                                                                                |\n"
            "| 2023-08-25 | News Article         | [Source: Financial Times] HR restructuring to improve efficiency                                    |\n\n"
            "Ensure each entry is up-to-date and includes links to sources when applicable."
        ),
        output_file=current_file_path,
        agent=current_data_collectors[company_name],
    ))

# Report generation tasks
report_tasks = []
for company_name in companies.keys():
    company_dir = f"data/{company_name.lower()}"
    report_file_path = os.path.join(company_dir, "investment_report_output.md")
    report_tasks.append(Task(
        description=f"Generate an investment report for {company_name} using historical and current data. Provide a recommendation based on recent trends.",
        expected_output=(
            "A markdown report with a summary, sections for historical trends, current performance, and a 5-star recommendation."
        ),
        output_file=report_file_path,
        agent=report_writer,
    ))

# Comparison task
comparison_output_file = "data/comparison_output.md"
comparison_task = Task(
    description="Create a markdown table comparing metrics for each company.",
    expected_output="A table with detailed comparisons and recommendations based on collected data.",
    output_file=comparison_output_file,
    agent=report_writer,
)

# Assemble and run tasks
crew = Crew(
    agents=[historical_data_collector] + list(current_data_collectors.values()) + [report_writer],
    tasks=data_collection_tasks + report_tasks + [comparison_task],
    verbose=1
)

# Execute tasks
print(crew.kickoff())
