# Autonomous Tool Integration Agent (ATIA)

ATIA is designed to enhance AI agent capabilities by autonomously discovering, integrating, and utilizing external APIs as tools. The system can identify when a tool is needed, search for appropriate APIs, comprehend documentation, handle account registration, securely manage credentials, and create reusable function definitions for future use.

## Project Structure

```
atia/
├── agent_core/      # Agent Core component
├── need_identifier/ # Need Identifier component
├── api_discovery/   # API Discovery component
├── doc_processor/   # Documentation Processor component
├── utils/           # Utility functions
├── config.py        # Configuration settings
├── __init__.py      # Package initialization
└── __main__.py      # CLI entry point
```

## Setup in Replit

To set up ATIA in Replit, follow these steps:

1. **Create a new Replit project**
   - Choose "Python" as the template
   - Name your project (e.g., "ATIA")

2. **Copy the code**
   - Copy all the provided code files into your Replit project, maintaining the directory structure

3. **Environment Variables**
   - Create a `.env` file in the root directory and add your API keys:
   ```
   OPENAI_API_KEY=your_openai_api_key
   RAPID_API_KEY=your_rapidapi_key  # Optional
   SERP_API_KEY=your_serpapi_key    # Optional
   GITHUB_TOKEN=your_github_token   # Optional
   ```

4. **Install Dependencies**
   - Replit will automatically install the dependencies from `pyproject.toml`
   - If you need to manually install them, run:
   ```bash
   pip install -r requirements.txt
   ```

5. **Run the Project**
   - To run in interactive mode:
   ```bash
   python -m atia --interactive
   ```
   - To process a single query:
   ```bash
   python -m atia "your query here"
   ```

## Phase 1 Implementation

This is the initial implementation of ATIA, focusing on:

- Setting up project infrastructure and development environment
- Implementing Agent Core with OpenAI integration
- Developing Need Identification with accuracy on test queries
- Creating basic API Discovery with search capabilities
- Setting up testing framework and CI pipeline

## Running Tests

To run the tests, execute:

```bash
python -m pytest
```

## Environment Variables

The following environment variables can be set in the `.env` file:

- `OPENAI_API_KEY`: Your OpenAI API key (required)
- `OPENAI_MODEL`: The OpenAI model to use (default: "gpt-4o")
- `OPENAI_MAX_TOKENS`: Maximum number of tokens for OpenAI API calls (default: 2000)
- `OPENAI_TEMPERATURE`: Temperature for OpenAI API calls (default: 0.2)
- `RAPID_API_KEY`: Your RapidAPI key (optional)
- `SERP_API_KEY`: Your SERP API key (optional)
- `GITHUB_TOKEN`: Your GitHub API token (optional)
- `DEBUG`: Set to "True" for debug mode (default: "False")
- `NEED_IDENTIFIER_THRESHOLD`: Confidence threshold for identifying tool needs (default: 0.75)

## Future Phases

- **Phase 2**: Key Integration & Security
- **Phase 3**: Advanced Processing
- **Phase 4**: Integration and Optimization

See the Technical Specification Document for more details on the future phases.