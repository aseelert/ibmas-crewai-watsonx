
# Investment Data Collection & Analysis Project

This project utilizes IBM's Watsonx LLM models and CrewAI agents to perform a comprehensive investment analysis of four major companies: IBM, Infosys, Accenture, and Google (Alphabet). The analysis focuses on historical data from the past two years and current market data to produce in-depth investment reports and comparisons for each company.

## Project Overview

The code is designed to:
1. **Collect Historical Data**: Analyze financial trends such as revenue growth, profitability, and market share over the last two years.
2. **Gather Current Data**: Scrape real-time stock prices, quarterly earnings, and relevant news articles from Yahoo Finance.
3. **Generate Investment Reports**: Synthesize historical and current data into comprehensive investment reports with strategic insights.
4. **Compare Companies**: Create a comparison report with a markdown table summarizing revenue, growth rate, profitability, and market position, followed by a 5-star investment rating.

## Project Structure

The project uses three main Watsonx models:
- **Granite** (ibm/granite-3-8b-instruct): For historical data analysis
- **Llama** (meta-llama/llama-3-405b-instruct): For current data scraping and real-time insights
- **Mistral** (mistralai/mistral-large): For generating the final investment report and comparative analysis

## Code Structure

The code defines the following roles and agents:

1. **Historical Data Specialist**: Collects financial trend data for the past two years for each company.
2. **Real-Time Market Data Analyst**: Fetches the latest stock price, earnings, and key news articles for each company.
3. **Investment Strategy Analyst**: Combines historical and current data to create investment reports for each company.
4. **Comparison Analyst**: Summarizes and compares all companies in a markdown table and provides a 5-star investment rating.

## Installation

1. **Clone the Repository**:
   ```bash
   git clone <repo-url>
   cd <repo-directory>
   ```

2. **Install Dependencies**:
   Ensure you have the necessary libraries by running:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set Environment Variables**:
   Export your Watsonx and Serper API keys and project ID:
   ```bash
   export WATSONX_APIKEY="your-api-key"
   export WATSONX_PROJECT_ID="your-project-id"
   export SERPER_API_KEY="your-serper-api-key"
   ```

## Folder Structure

The code creates a `data/` directory, with each company having its own subdirectory (e.g., `data/ibm/`). For each company, the following output files are generated:
- `historical_data_output.md`: Historical financial data
- `current_data_output.md`: Current stock, earnings, and news summary
- `investment_report_output.md`: Comprehensive investment report
- `comparison_output.md`: A final report comparing all companies

## Execution

To execute the analysis and generate reports:
```bash
python agent.py
```

The program will create and populate the `data/` directory with markdown files for each company, containing historical, current, and investment reports, as well as a final comparative analysis.

## Example Output

### Company Investment Report (Markdown)

Each company's report includes a detailed analysis with sections on revenue, growth rate, market share, and a 5-star investment rating.

```markdown
# IBM Investment Report

- **Revenue Growth**: 3.2%
- **Profitability**: 12%
- **Market Share**: 10%
- **Investment Rating**: *****
```

### Comparison Report

The final report compares the companies side-by-side.

```markdown
| Company       | Revenue (USD) | Growth Rate | Profitability | Market Share | Rating |
|---------------|---------------|-------------|---------------|--------------|--------|
| IBM           | $75B          | 3.2%        | 12%           | 10%          | *****  |
| Infosys       | $15B          | 8.5%        | 15%           | 2%           | ****   |
```

## Future Improvements

1. **Advanced Data Collection**: Implement a broader search for relevant news beyond Yahoo Finance.
2. **Enhanced Model Selection**: Consider model fine-tuning for specific industry trends.
3. **Multi-Year Analysis**: Expand analysis capabilities for 3–5 year financial trends.

## License

MIT License

---

This project helps investment analysts and strategists quickly gain actionable insights on multiple companies, integrating historical trends and real-time data.
