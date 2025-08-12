from crewai import Agent, Task, Crew, LLM
from crewai_tools import MCPServerAdapter
from mcp import StdioServerParameters
import sys
import platform
from pathlib import Path
import os
import warnings
from pydantic import PydanticDeprecatedSince20
from dotenv import load_dotenv
import traceback
import subprocess
from pydantic import BaseModel,Field

FILESYSTEM_SERVER_DIR_PATH = os.getenv("FILESYSTEM_SERVER_DIR_PATH", os.path.join(os.path.expanduser("~"), "Downloads"))

class Summary(BaseModel):
    summary: str = Field(description="A detailed summary of the research findings")

# Load environment variables
load_dotenv()

def get_available_llm():
    """Get the first available LLM from environment variables"""
    llm_configs = [
        {
            "name": "OpenRouter",
            "model": "openrouter/deepseek/deepseek-r1-0528:free",
            #"model": "openrouter/deepseek/deepseek-chat-v3-0324:free",
            #"model": "openrouter/qwen/qwen3-coder:free",
            "base_url": "https://openrouter.ai/api/v1",
            "api_key_env": "OPEN_ROUTER_API_KEY",
            "temperature": 0.7
        },
        {
            "name": "Gemini",
            "model": "gemini/gemini-2.5-flash",
            "api_key_env": "GEMINI_API_KEY",
            "temperature": 0.7
        },
        {
            "name": "Ollama Local",
            "model": "ollama/llama3.2",
            "api_key_env": None,  # No API key needed for local
            "temperature": 0.7
        }
    ]
    
    print("🔍 Checking available LLM providers...")
    
    for config in llm_configs:
        if config["api_key_env"] is None:
            # For local models like Ollama, try without API key
            print(f"⚡ Trying {config['name']} (Local)...")
            llm = LLM(
                model=config["model"],
                temperature=config["temperature"],
                max_tokens=1000,
            )
            print(f"✅ Using {config['name']}: {config['model']}")
            return llm
        else:
            api_key = os.getenv(config["api_key_env"])
            if api_key:
                print(f"⚡ Trying {config['name']}...")
                llm = LLM(
                    model=config["model"],
                    temperature=config["temperature"],
                    api_key=api_key
                )
                print(f"✅ Using {config['name']}: {config['model']}")
                return llm
            else:
                print(f"⚠️  {config['name']} API key not found in environment")


llm = get_available_llm()

# Suppress warnings
warnings.filterwarnings("ignore", category=PydanticDeprecatedSince20)

# Get current directory
base_dir = Path(__file__).parent.resolve()

print(f"Python executable: {sys.executable}")
print(f"Current directory: {os.getcwd()}")
print(f"Base directory: {base_dir}")

# Determine correct npx command for Windows
npx_cmd = "npx.cmd" if platform.system() == "Windows" else "npx"

def check_npx_availability():
    """Check if npx is available and working"""
    try:
        result = subprocess.run([npx_cmd, "--version"], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print(f"✓ NPX is available: {result.stdout.strip()}")
            return True
        else:
            print(f"✗ NPX check failed: {result.stderr}")
            return False
    except Exception as e:
        print(f"✗ NPX not available: {e}")
        return False

SEARCH_SERVER_FILE_PATH = Path(os.getenv("SEARCH_SERVER_FILE_PATH", base_dir / "servers" / "search_server.py"))
def check_search_server():
    """Check if the Python search server exists"""
    server_path = SEARCH_SERVER_FILE_PATH
    if server_path.exists():
        print(f"✓ Python search server found: {server_path}")
        return True
    else:
        print(f"✗ Python search server not found: {server_path}")
        return False

def get_working_servers():
    """Get list of working server configurations"""
    working_servers = []
    
    print("\n" + "="*50)
    print("DIAGNOSING MCP SERVERS")
    print("="*50)
    

    # Check Python search server
    search_server_available = check_search_server()
    if search_server_available:
        #search_server_params = StdioServerParameters(
        #    command="python", 
        #    args=[
        #        str(SEARCH_SERVER_FILE_PATH),
        #    ],
        #    env={"UV_PYTHON": "3.12", **os.environ},
        #
        search_server_params = {
            "url": "http://localhost:8081/mcp",
            "transport": "streamable-http"
        }
        working_servers.append(("Python Search Server", search_server_params))
        print("✓ Python search server configured")
    else:
        print("✗ Skipping Python search server (server file not found)")

    # Check NPX availability for filesystem server only
    npx_available = check_npx_availability()
    
    # Only add NPX servers if Node.js version is recent enough
    if npx_available:
        node_version_check = check_node_version()
        if node_version_check:
            # Filesystem server configuration
            filesystem_server_params = StdioServerParameters(
                command=npx_cmd,
                args=[
                    "-y",
                    "@modelcontextprotocol/server-filesystem",
                    FILESYSTEM_SERVER_DIR_PATH
                ],
            )
            working_servers.append(("Filesystem Server", filesystem_server_params))
            print(f"✓ Filesystem server configured {FILESYSTEM_SERVER_DIR_PATH}")
        else:
            print("⚠️  Skipping NPX filesystem server due to Node.js version compatibility issues")
            print("💡 To enable filesystem server, update Node.js to version 18+ or 20+")
            print("   Visit: https://nodejs.org/en/download/")
    else:
        print("✗ Skipping NPX filesystem server (NPX not available)")

    print(f"\nFound {len(working_servers)} server configurations")
    return working_servers

def check_node_version():
    """Check if Node.js version is compatible"""
    try:
        result = subprocess.run(["node", "--version"], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            version = result.stdout.strip()
            print(f"Node.js version: {version}")
            # Extract major version number
            major_version = int(version.lstrip('v').split('.')[0])
            if major_version >= 18:
                print("✓ Node.js version is compatible")
                return True
            else:
                print(f"⚠️  Node.js version {version} may be too old (recommend v18+)")
                return False
        return False
    except Exception as e:
        print(f"✗ Cannot check Node.js version: {e}")
        return False

class CustomMCPServerAdapter(MCPServerAdapter):
    """Custom MCP Server Adapter with increased timeout"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.timeout = 90  # Increase timeout to 90 seconds

def test_servers_individually(server_configs):
    """Test each server individually to identify problematic ones"""
    working_servers = []
    
    print("\n" + "="*50)
    print("TESTING SERVERS INDIVIDUALLY")
    print("="*50)
    
    for name, server_params in server_configs:
        print(f"\nTesting {name}...")
        try:
            with CustomMCPServerAdapter([server_params]) as tools:
                print(f"✓ {name} connected successfully!")
                print(f"  Available tools: {[tool.name for tool in tools]}")
                working_servers.append(server_params)
        except Exception as e:
            print(f"✗ {name} failed: {str(e)[:100]}...")
            continue
    
    return working_servers

def create_agent_and_tasks(tools=None):
    """Create agent and tasks with or without tools"""
    tools_list = tools or []
    
    # Adjust role and tasks based on available tools
    if tools_list:
        tool_names = [getattr(tool, 'name', 'unknown') for tool in tools_list]
        print(f"Agent will have access to: {tool_names}")
        
        role = "AI Research Creator with Tools"
        goal = "Research topics thoroughly using available MCP tools, create comprehensive diagrams, and save summaries"
        backstory = "An AI researcher and creator that specializes in using MCP tools to gather information, create visual representations, and save findings."
    else:
        role = "AI Research Creator"
        goal = "Research topics using built-in knowledge and create comprehensive analysis"
        backstory = "An AI researcher that specializes in analyzing topics and providing detailed insights using available knowledge."
    
    agent = Agent(
        role=role,
        goal=goal,
        backstory=backstory,
        tools=tools_list,
        llm=llm,
        verbose=True,
    )
    
    if tools_list:
        research_task = Task(
            description="Research the topic '{topic}' thoroughly using available MCP tools.",
            expected_output="A comprehensive research summary.",
            agent=agent,
        )
        
        summary_task = Task(
            description=f"Create a detailed summary of your research findings. If filesystem tools are available, save it as a text file in the {FILESYSTEM_SERVER_DIR_PATH} folder. Include key insights, important details, and references to any websites queried.",
            expected_output="A detailed summary of research findings, preferably saved as a text file if filesystem access is available.The final response should be in the format of a pydantic model Summary",
            agent=agent,
            output_pydantic=Summary
        )
    else:
        research_task = Task(
            description="Research and analyze the topic '{topic}' thoroughly using your knowledge. Provide detailed insights about how it works, including key components, processes, and relationships.",
            expected_output="A comprehensive analysis and explanation of the topic with detailed insights.",
            agent=agent,
        )
        
        summary_task = Task(
            description="Create a detailed summary of your analysis, highlighting the most important aspects, key insights, and practical implications of the topic.",
            expected_output="A well-structured summary with key findings and insights about the topic.The final response should be in the format of a pydantic model Summary",
            agent=agent,
            output_pydantic=Summary,
            markdown=True,  # Enable markdown formatting for the final output
            output_file="report.md"
        )
    
    return agent, [research_task, summary_task]

def main():
    """Main function to run the CrewAI application"""
    # Get available server configurations
    server_configs = get_working_servers()
    
    if not server_configs:
        raise ValueError("\n⚠️  No MCP servers available!")
    
    # Test servers individually to find working ones
    working_server_params = test_servers_individually(server_configs)
    
    if not working_server_params:
        raise ValueError("\n⚠️  No MCP servers are working!")
    
    try:
        print(f"\n✓ Using {len(working_server_params)} working MCP server(s)")
        print("Initializing MCP Server Adapter...")
        
        with CustomMCPServerAdapter(working_server_params) as tools:
            print(f"Successfully connected to MCP servers!")
            print(f"Available tools: {[tool.name for tool in tools]}")
            
            # Create agent and tasks with MCP tools
            agent, tasks = create_agent_and_tasks(tools)
            
            # Create crew with error handling
            crew = Crew(
                agents=[agent],
                tasks=tasks,
                verbose=True,
                reasoning=True,
            )
            
            # Get user input
            topic = input("\nPlease provide a topic to research: ").strip()
            if not topic:
                topic = "artificial intelligence"
                print(f"No topic provided, using default: {topic}")
            
            # Execute crew with retry mechanism
            max_retries = 2
            for attempt in range(max_retries + 1):
                try:
                    print(f"\nStarting research on: {topic} (Attempt {attempt + 1})")
                    result = crew.kickoff(inputs={"topic": topic})
                    # print("\n" + "="*50)
                    # print("FINAL RESULT FROM THE AGENT")
                    # print("="*50)
                   
                    response = result["summary"]
                    print(response)
                    print(f"Summary task output :{tasks[1].output}")
                    return response
                except Exception as e:
                    if attempt < max_retries:
                        print(f"⚠️  Attempt {attempt + 1} failed: {str(e)[:100]}...")
                        print(f"🔄 Retrying... ({attempt + 2}/{max_retries + 1})")
                        continue
                    else:
                        print(f"❌ All attempts failed. Error: {e}")
                        raise e
            
    except Exception as e:
        print(f"Error running with MCP tools: {e}")
        traceback.print_exc()
        raise e


if __name__ == "__main__":
    print("🚀 Starting CrewAI MCP Demo")
    print("\n📋 Setup Instructions:")
    print("   For more MCP servers, update Node.js to v18+: https://nodejs.org")
    print("   Add API keys to .env file for additional LLM providers")
    print("   Supported: GROQ_API_KEY, OPENAI_API_KEY, ANTHROPIC_API_KEY, BRAVE_API_KEY")
    result = main()
    #print(result)
